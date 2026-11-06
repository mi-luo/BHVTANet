import copy
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from bhvtanet.geometry import lorentz as L

VALID_MCTF_FEATURE_MODES = {'feature_kinds', 'multilevel_cache'}

DEFAULT_MCTF_CFG: Dict[str, Any] = {
    'enabled': False,
    'consensus_supervision_enabled': True,
    'depth_structure_enabled': True,
    'feature_mode': 'feature_kinds',
    'level_feature_kinds': ['perceptual', 'clean', 'semantic'],
    'level_feature_paths': [],
    'allow_missing_multilevel_fallback': True,
    'fallback_feature_kinds': ['perceptual', 'clean', 'semantic'],
    'level_names': None,
    'normalize_levels': None,
    'use_consensus': True,
    'main_target': 'baseline',
    'target_mix': 0.10,
    'target_mix_warmup_epochs': 0,
    'consensus_weight': 0.15,
    'learnable_alpha': True,
    'alpha_init': 'uniform',
    'aux_weight': 0.35,
    'use_aux_distill': True,
    'anchor_weight': 0.0,
    'anchor_consensus_weight': 0.0,
    'anchor_start_epoch': 0,
    'dcc_weight': 0.02,
    'mpc_weight': 0.005,
    'progressive': {
        'enabled': True,
        'mode': 'soft',
        'warmup_epochs': 12,
        'bias_scale': 1.0,
        'start_levels': 1,
        'release_every': 4,
        'hybrid_soft_bias_scale': 0.5,
    },
    'subject_conditioning': {
        'enabled': False,
        'num_subjects': 1,
        'start_epoch': 0,
        'scale': 1.0,
    },
    'sample_depth_bias': {
        'enabled': False,
        'mode': 'scalar_prior',
        'hidden_dim': 16,
        'scale': 0.35,
        'start_epoch': 0,
        'detach': True,
        'num_stats': 4,
    },
    'schedule': {
        'consensus_start_epoch': 0,
        'aux_start_epoch': 0,
        'regularization_start_epoch': 0,
    },
    'single_level_override': None,
    'save_alpha_history': True,
    'auto_extract': {
        'enabled': False,
        'image_root': None,
        'levels': None,
        'glob_pattern': '**/*.*',
        'batch_size': 256,
        'num_workers': 4,
        'use_dataset_image_paths': True,
    },
}


def _to_plain_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    if hasattr(obj, 'items'):
        return {k: v for k, v in obj.items()}
    return {}


def resolve_mctf_config(config: Dict[str, Any]) -> Dict[str, Any]:
    config = _to_plain_dict(config)
    raw = _to_plain_dict(config.get('mctf', {}))
    resolved = copy.deepcopy(DEFAULT_MCTF_CFG)
    resolved.update({
        k: v for k, v in raw.items()
        if k not in {'progressive', 'subject_conditioning', 'sample_depth_bias', 'schedule', 'auto_extract'} and v is not None
    })

    progressive = copy.deepcopy(DEFAULT_MCTF_CFG['progressive'])
    progressive.update(_to_plain_dict(raw.get('progressive', {})))
    progressive['enabled'] = bool(progressive.get('enabled', False))
    progressive['mode'] = str(progressive.get('mode', 'soft')).lower()
    if progressive['mode'] not in {'soft', 'hard', 'hybrid'}:
        raise ValueError(f"Unsupported mctf.progressive.mode: {progressive['mode']}")
    progressive['warmup_epochs'] = max(1, int(progressive.get('warmup_epochs', 12)))
    progressive['bias_scale'] = float(progressive.get('bias_scale', 1.0))
    progressive['start_levels'] = max(1, int(progressive.get('start_levels', 1)))
    progressive['release_every'] = max(1, int(progressive.get('release_every', 4)))
    progressive['hybrid_soft_bias_scale'] = float(progressive.get('hybrid_soft_bias_scale', 0.5))
    resolved['progressive'] = progressive

    subject_conditioning = copy.deepcopy(DEFAULT_MCTF_CFG['subject_conditioning'])
    subject_conditioning.update(_to_plain_dict(raw.get('subject_conditioning', {})))
    subject_conditioning['enabled'] = bool(subject_conditioning.get('enabled', False))
    subject_conditioning['num_subjects'] = max(1, int(subject_conditioning.get('num_subjects', 1)))
    subject_conditioning['start_epoch'] = max(0, int(subject_conditioning.get('start_epoch', 0)))
    subject_conditioning['scale'] = float(subject_conditioning.get('scale', 1.0))
    resolved['subject_conditioning'] = subject_conditioning

    sample_depth_bias = copy.deepcopy(DEFAULT_MCTF_CFG['sample_depth_bias'])
    sample_depth_bias.update(_to_plain_dict(raw.get('sample_depth_bias', {})))
    sample_depth_bias['enabled'] = bool(sample_depth_bias.get('enabled', False))
    sample_depth_bias['mode'] = str(sample_depth_bias.get('mode', 'scalar_prior')).lower()
    if sample_depth_bias['mode'] not in {'scalar_prior', 'logit_bias'}:
        raise ValueError(f"Unsupported mctf.sample_depth_bias.mode: {sample_depth_bias['mode']}")
    sample_depth_bias['hidden_dim'] = max(4, int(sample_depth_bias.get('hidden_dim', 16)))
    sample_depth_bias['scale'] = float(sample_depth_bias.get('scale', 0.35))
    sample_depth_bias['start_epoch'] = max(0, int(sample_depth_bias.get('start_epoch', 0)))
    sample_depth_bias['detach'] = bool(sample_depth_bias.get('detach', True))
    sample_depth_bias['num_stats'] = max(1, int(sample_depth_bias.get('num_stats', 4)))
    resolved['sample_depth_bias'] = sample_depth_bias


    auto_extract = copy.deepcopy(DEFAULT_MCTF_CFG['auto_extract'])
    auto_extract.update(_to_plain_dict(raw.get('auto_extract', {})))
    auto_extract['enabled'] = bool(auto_extract.get('enabled', False))
    auto_extract['image_root'] = auto_extract.get('image_root')
    levels = auto_extract.get('levels')
    if isinstance(levels, str):
        if levels.strip().lower() in {'', 'auto', 'none'}:
            levels = None
        else:
            levels = [token.strip() for token in levels.split(',') if token.strip()]
    elif levels is not None:
        levels = [str(token).strip() for token in levels if str(token).strip()]
        if not levels:
            levels = None
    auto_extract['levels'] = levels
    auto_extract['glob_pattern'] = str(auto_extract.get('glob_pattern', '**/*.*'))
    auto_extract['batch_size'] = max(1, int(auto_extract.get('batch_size', 256)))
    auto_extract['num_workers'] = max(0, int(auto_extract.get('num_workers', 4)))
    auto_extract['use_dataset_image_paths'] = bool(auto_extract.get('use_dataset_image_paths', True))
    resolved['auto_extract'] = auto_extract

    schedule = copy.deepcopy(DEFAULT_MCTF_CFG['schedule'])
    schedule.update(_to_plain_dict(raw.get('schedule', {})))
    schedule['consensus_start_epoch'] = max(0, int(schedule.get('consensus_start_epoch', 0)))
    schedule['aux_start_epoch'] = max(0, int(schedule.get('aux_start_epoch', 0)))
    schedule['regularization_start_epoch'] = max(0, int(schedule.get('regularization_start_epoch', 0)))
    resolved['schedule'] = schedule

    resolved['enabled'] = bool(resolved.get('enabled', False))
    resolved['consensus_supervision_enabled'] = bool(resolved.get('consensus_supervision_enabled', True))
    resolved['depth_structure_enabled'] = bool(resolved.get('depth_structure_enabled', True))
    resolved['feature_mode'] = str(resolved.get('feature_mode', 'feature_kinds')).lower()
    if resolved['feature_mode'] not in VALID_MCTF_FEATURE_MODES:
        raise ValueError(f"Unsupported mctf.feature_mode: {resolved['feature_mode']}")

    resolved['level_feature_kinds'] = list(resolved.get('level_feature_kinds', []))
    resolved['level_feature_paths'] = list(resolved.get('level_feature_paths', []))
    resolved['allow_missing_multilevel_fallback'] = bool(resolved.get('allow_missing_multilevel_fallback', True))
    resolved['fallback_feature_kinds'] = list(resolved.get('fallback_feature_kinds', ['perceptual', 'clean', 'semantic']))

    if resolved['feature_mode'] == 'feature_kinds':
        if not resolved['level_feature_kinds']:
            raise ValueError('mctf.level_feature_kinds must not be empty when feature_mode=feature_kinds')
        if len(set(resolved['level_feature_kinds'])) != len(resolved['level_feature_kinds']):
            raise ValueError('mctf.level_feature_kinds must be unique and ordered')
    elif resolved['enabled'] and not resolved['level_feature_paths']:
        raise ValueError('mctf.level_feature_paths must not be empty when feature_mode=multilevel_cache')

    resolved['use_consensus'] = bool(resolved.get('use_consensus', True))
    resolved['main_target'] = str(resolved.get('main_target', 'baseline')).lower()
    if resolved['main_target'] not in {'baseline', 'consensus', 'blend'}:
        raise ValueError(f"Unsupported mctf.main_target: {resolved['main_target']}")
    resolved['consensus_weight'] = float(resolved.get('consensus_weight', 0.15))
    resolved['target_mix'] = max(0.0, min(1.0, float(resolved.get('target_mix', 0.10))))
    resolved['target_mix_warmup_epochs'] = max(0, int(resolved.get('target_mix_warmup_epochs', 0)))
    resolved['learnable_alpha'] = bool(resolved.get('learnable_alpha', True))
    resolved['use_aux_distill'] = bool(resolved.get('use_aux_distill', True))
    resolved['save_alpha_history'] = bool(resolved.get('save_alpha_history', True))
    resolved['aux_weight'] = float(resolved.get('aux_weight', 0.0))
    resolved['anchor_weight'] = float(resolved.get('anchor_weight', 0.0))
    resolved['anchor_consensus_weight'] = float(resolved.get('anchor_consensus_weight', 0.0))
    resolved['anchor_start_epoch'] = max(0, int(resolved.get('anchor_start_epoch', 0)))
    resolved['dcc_weight'] = float(resolved.get('dcc_weight', 0.0))
    resolved['mpc_weight'] = float(resolved.get('mpc_weight', 0.0))
    single_level_override = resolved.get('single_level_override')
    if isinstance(single_level_override, str) and single_level_override.lower() in {'', 'none'}:
        single_level_override = None
    resolved['single_level_override'] = None if single_level_override is None else int(single_level_override)

    if not resolved['consensus_supervision_enabled']:
        resolved['use_consensus'] = False
        resolved['use_aux_distill'] = False
        resolved['aux_weight'] = 0.0
        resolved['consensus_weight'] = 0.0
        resolved['anchor_weight'] = 0.0
        resolved['anchor_consensus_weight'] = 0.0
        if resolved['main_target'] in {'consensus', 'blend'}:
            resolved['main_target'] = 'baseline'

    if not resolved['depth_structure_enabled']:
        resolved['dcc_weight'] = 0.0
        resolved['mpc_weight'] = 0.0
        resolved['progressive']['enabled'] = False
        resolved['subject_conditioning']['enabled'] = False
        resolved['sample_depth_bias']['enabled'] = False

    level_names = resolved.get('level_names')
    if level_names is not None:
        level_names = list(level_names)
    if resolved['feature_mode'] == 'feature_kinds':
        expected_levels = len(resolved['level_feature_kinds'])
        if level_names is not None and len(level_names) != expected_levels:
            raise ValueError(f'mctf.level_names must match number of levels ({expected_levels})')
        if level_names is None:
            level_names = [f'level_{i}' for i in range(expected_levels)]
    else:
        expected_levels = len(resolved['level_feature_paths'])
        if level_names is None:
            level_names = [f'level_{i}' for i in range(expected_levels)]
    resolved['level_names'] = level_names

    normalize_levels = resolved.get('normalize_levels')
    if normalize_levels is None:
        normalize_levels = [True] * expected_levels
    elif isinstance(normalize_levels, bool):
        normalize_levels = [bool(normalize_levels)] * expected_levels
    else:
        normalize_levels = [bool(v) for v in normalize_levels]
    if resolved['feature_mode'] == 'feature_kinds' and len(normalize_levels) != expected_levels:
        raise ValueError(f'mctf.normalize_levels must match number of levels ({expected_levels})')
    resolved['normalize_levels'] = normalize_levels
    return resolved


from .mdtp import MDTP
from .dacam import DACAM
from .dswr import DSWR


class MCTF(nn.Module):
    """Multi-Depth Consensus Target Fusion (paper Section 3.2).

    MDTP creates ordered targets, DACAM estimates depth posteriors, and DSWR
    exposes the two structural regularization terms consumed by the trainer.
    """
    def __init__(self, level_input_dims: Sequence[int], z_dim: int,
                 mctf_cfg: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.cfg = resolve_mctf_config({'mctf': mctf_cfg})
        self.level_input_dims = [int(dim) for dim in level_input_dims]
        self.num_levels = len(self.level_input_dims)

        level_names = list(self.cfg.get('level_names', []) or [])
        if len(level_names) != self.num_levels:
            level_names = [f'level_{i}' for i in range(self.num_levels)]
        self.level_names = level_names
        self.cfg['level_names'] = level_names

        normalize_levels = self.cfg.get('normalize_levels')
        if normalize_levels is None:
            normalize_levels = [True] * self.num_levels
        elif isinstance(normalize_levels, bool):
            normalize_levels = [bool(normalize_levels)] * self.num_levels
        else:
            normalize_levels = [bool(v) for v in normalize_levels]
        if len(normalize_levels) != self.num_levels:
            if len(normalize_levels) == 1:
                normalize_levels = normalize_levels * self.num_levels
            else:
                normalize_levels = [True] * self.num_levels
        self.normalize_levels = normalize_levels
        self.cfg['normalize_levels'] = normalize_levels

        self.mdtp = MDTP(self.level_input_dims, z_dim, self.normalize_levels)
        self.dacam = DACAM()
        self.dswr = DSWR()
        init_logits = self._init_global_depth_logits(self.cfg['alpha_init'])
        if self.cfg['learnable_alpha']:
            self.global_depth_logits = nn.Parameter(init_logits)
        else:
            self.register_buffer('global_depth_logits', init_logits, persistent=True)

        depth_positions = torch.linspace(0.0, 1.0, self.num_levels)
        self.register_buffer('depth_positions', depth_positions, persistent=True)
        self.register_buffer('depth_centered', depth_positions - depth_positions.mean(), persistent=True)

        subj_cfg = self.cfg['subject_conditioning']
        if subj_cfg['enabled']:
            self.subject_depth_bias = nn.Embedding(subj_cfg['num_subjects'], self.num_levels)
            nn.init.zeros_(self.subject_depth_bias.weight)
        else:
            self.subject_depth_bias = None

        bias_cfg = self.cfg['sample_depth_bias']
        if bias_cfg['enabled']:
            self.sample_depth_scalar = nn.Sequential(
                nn.Linear(bias_cfg['num_stats'], bias_cfg['hidden_dim']),
                nn.GELU(),
                nn.Linear(bias_cfg['hidden_dim'], 1),
            )
            self.sample_depth_bias = nn.Sequential(
                nn.Linear(bias_cfg['num_stats'], bias_cfg['hidden_dim']),
                nn.GELU(),
                nn.Linear(bias_cfg['hidden_dim'], self.num_levels),
            )
            for module in (self.sample_depth_scalar[-1], self.sample_depth_bias[-1]):
                nn.init.zeros_(module.weight)
                nn.init.zeros_(module.bias)
        else:
            self.sample_depth_scalar = None
            self.sample_depth_bias = None

    def _init_global_depth_logits(self, mode: str) -> torch.Tensor:
        mode = str(mode).lower()
        if mode == 'uniform':
            return torch.zeros(self.num_levels)
        if mode == 'middle':
            center = (self.num_levels - 1) / 2.0
            logits = -0.5 * (torch.arange(self.num_levels) - center).float().pow(2)
            return logits
        if mode == 'shallow':
            return -torch.arange(self.num_levels).float()
        if mode in {'deep', 'final'}:
            return torch.arange(self.num_levels).float()
        raise ValueError(f'Unsupported alpha_init: {mode}')

    @staticmethod
    def _normalize_if_needed(x: torch.Tensor, enabled: bool) -> torch.Tensor:
        if not enabled:
            return x
        return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    def _progressive_log_prior(self, current_epoch: int) -> torch.Tensor:
        """Return the log-prior offset; its common normalizer cancels in softmax."""
        progressive = self.cfg['progressive']
        if not progressive['enabled']:
            return torch.zeros_like(self.global_depth_logits)
        mode = str(progressive.get('mode', 'soft')).lower()
        warmup = max(1, int(progressive['warmup_epochs']))
        phase = min(max(float(current_epoch), 0.0) / float(warmup), 1.0)
        if mode == 'hybrid':
            strength = (1.0 - phase) * float(progressive.get('hybrid_soft_bias_scale', progressive.get('bias_scale', 1.0)))
        else:
            strength = (1.0 - phase) * float(progressive.get('bias_scale', 1.0))
        if strength <= 0:
            return torch.zeros_like(self.global_depth_logits)
        log_prior = -self.depth_positions * strength
        return log_prior.to(self.global_depth_logits.device, self.global_depth_logits.dtype)

    def _progressive_hard_mask(self, batch_size: int, device: torch.device, dtype: torch.dtype, current_epoch: int) -> torch.Tensor:
        progressive = self.cfg['progressive']
        if not progressive['enabled']:
            return torch.ones((batch_size, self.num_levels), device=device, dtype=dtype)
        mode = str(progressive.get('mode', 'soft')).lower()
        if mode not in {'hard', 'hybrid'}:
            return torch.ones((batch_size, self.num_levels), device=device, dtype=dtype)
        start_levels = min(max(1, int(progressive.get('start_levels', 1))), self.num_levels)
        release_every = max(1, int(progressive.get('release_every', 4)))
        released = start_levels + max(0, int(current_epoch)) // release_every
        active_levels = min(self.num_levels, released)
        mask = torch.zeros((batch_size, self.num_levels), device=device, dtype=dtype)
        mask[:, :active_levels] = 1.0
        return mask

    def _subject_depth_bias_logits(self, subject_ids: Optional[torch.Tensor], current_epoch: int, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        subj_cfg = self.cfg['subject_conditioning']
        if self.subject_depth_bias is None or not subj_cfg['enabled'] or current_epoch < subj_cfg['start_epoch']:
            return torch.zeros((batch_size, self.num_levels), device=device, dtype=dtype)
        if subject_ids is None:
            return torch.zeros((batch_size, self.num_levels), device=device, dtype=dtype)
        subject_ids = subject_ids.to(device=device).long().clamp(min=0, max=self.subject_depth_bias.num_embeddings - 1)
        return self.subject_depth_bias(subject_ids).to(dtype=dtype) * float(subj_cfg.get('scale', 1.0))

    def _sample_depth_bias_logits(self, sample_depth_stats: Optional[torch.Tensor], current_epoch: int, batch_size: int, device: torch.device, dtype: torch.dtype):
        bias_cfg = self.cfg['sample_depth_bias']
        zero_logits = torch.zeros((batch_size, self.num_levels), device=device, dtype=dtype)
        zero_score = torch.full((batch_size, 1), 0.5, device=device, dtype=dtype)
        if self.sample_depth_scalar is None or self.sample_depth_bias is None or not bias_cfg['enabled'] or current_epoch < bias_cfg['start_epoch']:
            return zero_logits, zero_score
        if sample_depth_stats is None:
            return zero_logits, zero_score
        stats = sample_depth_stats.to(device=device, dtype=dtype)
        if bias_cfg.get('detach', True):
            stats = stats.detach()
        score = torch.sigmoid(self.sample_depth_scalar(stats))
        if bias_cfg['mode'] == 'scalar_prior':
            logits = (2.0 * score - 1.0) * self.depth_centered.view(1, -1) * float(bias_cfg.get('scale', 0.35))
        else:
            logits = self.sample_depth_bias(stats) * float(bias_cfg.get('scale', 0.35))
        return logits.to(dtype=dtype), score.to(dtype=dtype)

    def compute_alpha(self, batch_size: int, device: torch.device, dtype: torch.dtype, current_epoch: int, context: Optional[Dict[str, Any]] = None):
        if self.cfg['single_level_override'] is not None:
            index = int(self.cfg['single_level_override'])
            if index < 0 or index >= self.num_levels:
                raise ValueError(f'single_level_override out of range: {index}')
            alpha = torch.zeros((batch_size, self.num_levels), device=device, dtype=dtype)
            alpha[:, index] = 1.0
            return {
                'alpha': alpha,
                'global_depth_logits': self.global_depth_logits.to(device=device, dtype=dtype).view(1, self.num_levels).expand(batch_size, -1),
                'subject_depth_bias': torch.zeros_like(alpha),
                'sample_depth_bias': torch.zeros_like(alpha),
                'progressive_log_prior': torch.zeros_like(alpha),
                'sample_depth_score': torch.ones((batch_size, 1), device=device, dtype=dtype),
            }

        context = context or {}
        global_depth_logits = self.global_depth_logits.to(device=device, dtype=dtype).view(1, self.num_levels).expand(batch_size, -1)
        progressive_log_prior = self._progressive_log_prior(current_epoch).to(device=device, dtype=dtype).view(1, self.num_levels).expand(batch_size, -1)
        subject_depth_bias = self._subject_depth_bias_logits(context.get('subject_ids'), current_epoch, batch_size, device, dtype)
        sample_depth_stats = context.get('sample_depth_stats')
        sample_depth_bias, sample_depth_score = self._sample_depth_bias_logits(sample_depth_stats, current_epoch, batch_size, device, dtype)
        hard_mask = self._progressive_hard_mask(batch_size=batch_size, device=device, dtype=dtype, current_epoch=current_epoch)
        alpha = self.dacam(global_depth_logits, progressive_log_prior, sample_depth_bias,
                           hard_mask=hard_mask, subject_depth_bias=subject_depth_bias)
        return {
            'alpha': alpha,
            'global_depth_logits': global_depth_logits,
            'subject_depth_bias': subject_depth_bias,
            'sample_depth_bias': sample_depth_bias,
            'progressive_log_prior': progressive_log_prior,
            'sample_depth_score': sample_depth_score,
        }

    def forward(self, level_features: Sequence[torch.Tensor], visual_alpha: torch.Tensor, curv: torch.Tensor, current_epoch: int, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if len(level_features) != self.num_levels:
            raise ValueError(f'Expected {self.num_levels} level feature tensors, got {len(level_features)}')

        depth_targets = self.mdtp(level_features)
        hyperbolic_depth_targets = [L.exp_map0(visual_alpha * target, curv) for target in depth_targets]

        batch_size = depth_targets[0].shape[0]
        alpha_out = self.compute_alpha(batch_size=batch_size, device=depth_targets[0].device, dtype=depth_targets[0].dtype, current_epoch=current_epoch, context=context)
        alpha = alpha_out['alpha']
        consensus_target = self.dacam.aggregate(depth_targets, alpha)
        hyperbolic_consensus_target = L.exp_map0(visual_alpha * consensus_target, curv)

        regularization = self.dswr(alpha)
        dcc_loss = regularization['dcc_loss']
        mpc_loss = regularization['mpc_loss']
        mean_depth_posterior = alpha.mean(dim=0)
        alpha_components = {
            'global_depth_logits': alpha_out['global_depth_logits'].detach(),
            'subject_depth_bias': alpha_out['subject_depth_bias'].detach(),
            'sample_depth_bias': alpha_out['sample_depth_bias'].detach(),
            'progressive_log_prior': alpha_out['progressive_log_prior'].detach(),
        }

        return {
            'alpha': alpha,
            'mean_depth_posterior': mean_depth_posterior,
            'depth_targets': depth_targets,
            'hyperbolic_depth_targets': hyperbolic_depth_targets,
            'consensus_target': consensus_target,
            'hyperbolic_consensus_target': hyperbolic_consensus_target,
            'dcc_loss': dcc_loss,
            'mpc_loss': mpc_loss,
            'sample_depth_score': alpha_out['sample_depth_score'],
            'alpha_components': alpha_components,
        }

    def aux_loss(self, retrieval_loss_fn, batch: Dict[str, Any], eeg_z: torch.Tensor,
                 hyperbolic_depth_targets: Sequence[torch.Tensor],
                 curv: torch.Tensor, *, use_memory: bool, update_memory: bool,
                 mean_depth_posterior: torch.Tensor):
        depth_losses = []
        for index, depth_target in enumerate(hyperbolic_depth_targets):
            depth_loss = retrieval_loss_fn(batch, eeg_z, depth_target, curv, use_memory=use_memory and index == len(hyperbolic_depth_targets) - 1, update_memory=update_memory and index == len(hyperbolic_depth_targets) - 1)
            depth_losses.append(depth_loss)
        stacked_losses = torch.stack(depth_losses)
        depth_weights = mean_depth_posterior.to(device=stacked_losses.device, dtype=stacked_losses.dtype)
        return (depth_weights * stacked_losses).sum(), stacked_losses.detach()

__all__ = ['MCTF', 'MDTP', 'DACAM', 'DSWR', 'resolve_mctf_config',
           'DEFAULT_MCTF_CFG', 'VALID_MCTF_FEATURE_MODES']
