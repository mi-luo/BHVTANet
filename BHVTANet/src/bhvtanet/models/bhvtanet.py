"""BHVTANet model and loss implementation."""

import argparse
import glob
import json
import os
import shutil
import platform
from pathlib import Path
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from omegaconf import OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim import Adam, AdamW, SGD

from bhvtanet.models.ablation import resolve_ablation_config
from bhvtanet.models.mctf import MCTF, resolve_mctf_config
from bhvtanet.data.common import infer_channel_count
from bhvtanet.geometry import lorentz as L
from bhvtanet.models.retrieval import RetrievalObjective, resolve_retrieval_config
from bhvtanet.training.config import build_warmup_cosine_scheduler, resolve_train_config, to_jsonable
from bhvtanet.training.utils import instantiate_from_config, update_config
from bhvtanet.training.paths import PROJECT_ROOT, register_path_resolvers, resolve_project_paths
from bhvtanet.training.checkpoint_io import LocalAtomicCheckpointIO



PRETRAIN_MAP = {
    'RN50': {'pretrained': 'openai', 'resize': (224, 224), 'z_dim': 1024},
    'RN101': {'pretrained': 'openai', 'resize': (224, 224), 'z_dim': 512},
    'ViT-B-16': {'pretrained': 'openai', 'resize': (224, 224), 'z_dim': 512},
    'ViT-B-32': {'pretrained': 'openai', 'resize': (224, 224), 'z_dim': 512},
    'OpenAI-ViT-L-14': {'pretrained': 'openai', 'resize': (224, 224), 'z_dim': 768},
    'OpenAI-ViT-L-14-336': {'pretrained': 'openai', 'resize': (336, 336), 'z_dim': 768},
    'ViT-L-14': {'pretrained': 'laion2b_s32b_b82k', 'resize': (224, 224), 'z_dim': 768},
    'DFN-ViT-L-14': {'pretrained': 'dfn2b_s39b', 'resize': (224, 224), 'z_dim': 768},
    'DataComp-ViT-L-14': {'pretrained': 'datacomp_xl_s13b_b90k', 'resize': (224, 224), 'z_dim': 768},
    'SigLIP-B-16-256': {'pretrained': 'webli', 'resize': (256, 256), 'z_dim': 768},
    'SigLIP2-B-16-256': {'pretrained': 'webli', 'resize': (256, 256), 'z_dim': 768},
    'SigLIP2-L-16-256': {'pretrained': 'webli', 'resize': (256, 256), 'z_dim': 1024},
    'Timm-SigLIP2-L-16-256': {'pretrained': 'webli-image-tower', 'resize': (256, 256), 'z_dim': 1024},
    'SigLIP-SO400M-14-384': {'pretrained': 'webli', 'resize': (384, 384), 'z_dim': 1152},
    'Timm-DINOv2-L-14': {'pretrained': 'lvd142m', 'resize': (518, 518), 'z_dim': 1024},
    'SigLIP-L-16-256': {'pretrained': 'webli', 'resize': (256, 256), 'z_dim': 1024},
    'PE-Core-B-16': {'pretrained': 'meta', 'resize': (224, 224), 'z_dim': 1024},
    'PE-Core-L-14-336': {'pretrained': 'meta', 'resize': (336, 336), 'z_dim': 1024},
    'PE-Core-bigG-14-448': {'pretrained': 'meta', 'resize': (448, 448), 'z_dim': 1280},
    'ViT-H-14': {'pretrained': 'laion2b_s32b_b79k', 'resize': (224, 224), 'z_dim': 1024},
    'MetaCLIP-H-14': {'pretrained': 'metaclip_fullcc', 'resize': (224, 224), 'z_dim': 1024},
    'ViT-H-14-CLIPA-336': {'pretrained': 'datacomp1b', 'resize': (336, 336), 'z_dim': 1024},
    'DFN5B-ViT-H-14-378': {'pretrained': 'dfn5b', 'resize': (378, 378), 'z_dim': 1024},
    'ViT-g-14': {'pretrained': 'laion2b_s34b_b88k', 'resize': (224, 224), 'z_dim': 1024},
    'EVA02-B-16': {'pretrained': 'merged2b_s8b_b131k', 'resize': (224, 224), 'z_dim': 512},
    'EVA02-L-14': {'pretrained': 'merged2b_s4b_b131k', 'resize': (224, 224), 'z_dim': 768},
    'ConvNeXt-BW': {'pretrained': 'laion2b_s13b_b82k_augreg', 'resize': (224, 224), 'z_dim': 640},
    'MobileCLIP2-B': {'pretrained': 'dfndr2b', 'resize': (224, 224), 'z_dim': 512},
    'MetaCLIP-B-16': {'pretrained': 'metaclip_fullcc', 'resize': (224, 224), 'z_dim': 512},
    'InternViT-L28': {'pretrained': 'cache', 'resize': (448, 448), 'z_dim': 512, 'feature_dim': 3200},
}


HYPERBOLIC_LOGIT_SCALE = 100.0029525756836


def _unwrap_dataset(dataset):
    base = dataset
    while hasattr(base, 'dataset'):
        base = base.dataset
    return base


def load_model(config, train_loader, test_loader):
    model = {}
    for key, value in config['models'].items():
        print(f'init {key}')
        model[key] = instantiate_from_config(value)
    return BHVTANet(model, config, train_loader, test_loader)



class BHVTANet(pl.LightningModule):
    def __init__(self, model, config, train_loader, test_loader, model_type='RN50'):
        super().__init__()
        self.config = config
        self.ablation = resolve_ablation_config(config)
        self.mctf_cfg = resolve_mctf_config(config)
        self.retrieval_cfg = resolve_retrieval_config(config)
        self.train_cfg = resolve_train_config(config)
        for key, value in model.items():
            setattr(self, f'{key}', value)

        self.retrieval_objective = RetrievalObjective(self.retrieval_cfg)
        self.all_predicted_classes = []
        self.all_true_labels = []
        self.z_dim = self.config['z_dim']
        self.sim = np.ones(len(train_loader.dataset))
        self.mAP_total = 0
        self.match_similarities = []
        self.score = torch.tensor(0.0)
        self.last_depth_posterior = None
        self.depth_posterior_history = []
        self.depth_posterior_test_batches = []
        self.sample_depth_score_history = []
        self.depth_logit_component_history = []

        train_dataset = _unwrap_dataset(train_loader.dataset)
        self.config['mctf'] = self.mctf_cfg
        self.config['mctf']['subject_conditioning']['num_subjects'] = max(1, len(getattr(train_dataset, 'subjects', []) or [0]))
        self.mctf_cfg = resolve_mctf_config(self.config)
        self.level_names = []
        if self.mctf_cfg['enabled']:
            level_dims = list(getattr(train_dataset, 'mctf_level_dims', []))
            if not level_dims:
                raise ValueError('MCTF is enabled but the dataset did not expose mctf_level_dims.')
            self.level_names = list(getattr(train_dataset, 'mctf_level_names', self.mctf_cfg['level_names']))
            if len(self.level_names) != len(level_dims):
                self.level_names = [f'level_{i}' for i in range(len(level_dims))]
                self.mctf_cfg['level_names'] = self.level_names
            self.mctf = MCTF(level_dims, self.z_dim, self.mctf_cfg)
        else:
            self.mctf = None

        self.public_teacher_cfg = dict(self.config.get('aux_teacher', {}) or {})
        self.public_teacher_enabled = bool(self.public_teacher_cfg.get('enabled', False))
        self.public_teacher_proj = None
        if self.public_teacher_enabled:
            aux_dim = int(self.public_teacher_cfg.get('feature_dim') or getattr(train_dataset, 'aux_teacher_dim', 0))
            if aux_dim <= 0:
                raise ValueError('aux_teacher.enabled requires a resolvable feature dimension.')
            dropout = float(self.public_teacher_cfg.get('dropout', 0.0))
            if str(self.public_teacher_cfg.get('projector', 'linear')).lower() == 'mlp':
                hidden_dim = int(self.public_teacher_cfg.get('hidden_dim', max(aux_dim, self.z_dim)))
                self.public_teacher_proj = nn.Sequential(
                    nn.LayerNorm(aux_dim),
                    nn.Linear(aux_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, self.z_dim),
                )
            else:
                self.public_teacher_proj = nn.Sequential(
                    nn.LayerNorm(aux_dim),
                    nn.Dropout(dropout),
                    nn.Linear(aux_dim, self.z_dim),
                )
        self.target_adapter_cfg = dict(self.config.get('target_adapter', {}) or {})
        self.target_adapter_enabled = bool(self.target_adapter_cfg.get('enabled', False))
        self.target_anchor_proj = None
        self.target_anchor_projs = None
        if self.target_adapter_enabled:
            anchor_dims = list(getattr(train_dataset, 'target_anchor_dims', []) or [])
            if not anchor_dims:
                anchor_dims = [int(getattr(train_dataset, 'target_anchor_dim', 0))]
            target_dim = int(self.config.get('visual_feature_dim', self.z_dim))
            projectors = []
            for anchor_dim in anchor_dims:
                projectors.append(self._build_target_anchor_projector(int(anchor_dim), target_dim))
            if len(projectors) == 1:
                self.target_anchor_proj = projectors[0]
            elif len(projectors) > 1:
                self.target_anchor_projs = nn.ModuleList(projectors)

    def _build_target_anchor_projector(self, anchor_dim: int, target_dim: int) -> nn.Module:
        if anchor_dim <= 0:
            return nn.Identity()
        projector = str(self.target_adapter_cfg.get('projector', 'none')).lower()
        if anchor_dim == target_dim and projector not in {'linear', 'mlp'}:
            return nn.Identity()
        if projector not in {'linear', 'mlp'}:
            raise ValueError(
                f'target_adapter anchor dim {anchor_dim} does not match primary feature dim {target_dim}; '
                'set target_adapter.projector to linear or mlp.'
            )
        dropout = float(self.target_adapter_cfg.get('dropout', 0.0))
        if projector == 'mlp':
            hidden_dim = int(self.target_adapter_cfg.get('hidden_dim', max(anchor_dim, target_dim)))
            return nn.Sequential(
                nn.LayerNorm(anchor_dim),
                nn.Linear(anchor_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, target_dim),
            )
        return nn.Sequential(
            nn.LayerNorm(anchor_dim),
            nn.Dropout(dropout),
            nn.Linear(anchor_dim, target_dim),
        )

    @staticmethod
    def _normalize_if_needed(x: torch.Tensor, enabled: bool) -> torch.Tensor:
        if not enabled:
            return x
        return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    def _mix_score(self, feature_for_score: torch.Tensor) -> torch.Tensor:
        batch_size = feature_for_score.shape[0]
        if not self.ablation['use_feature_interpolation']:
            return torch.zeros((batch_size, 1), device=feature_for_score.device, dtype=feature_for_score.dtype)
        if self.ablation['learnable_mix_score']:
            return self.brain.score(feature_for_score)
        weight = float(self.ablation['fixed_mix_weight'])
        return torch.full((batch_size, 1), weight, device=feature_for_score.device, dtype=feature_for_score.dtype)

    def _adapt_primary_feature(self, batch, primary_raw: torch.Tensor) -> torch.Tensor:
        has_single_anchor = 'target_anchor_feature' in batch
        has_multi_anchor = 'target_anchor_features' in batch
        if not self.target_adapter_enabled or (not has_single_anchor and not has_multi_anchor):
            return primary_raw

        if has_multi_anchor:
            anchors = batch['target_anchor_features']
            if not isinstance(anchors, (list, tuple)):
                raise TypeError(f'target_anchor_features must be a list/tuple, got {type(anchors)}')
            projected_anchors = []
            projectors = list(self.target_anchor_projs) if self.target_anchor_projs is not None else []
            if len(projectors) != len(anchors):
                raise ValueError(f'Expected {len(projectors)} target anchor projectors, got {len(anchors)} anchors.')
            weights_cfg = self.target_adapter_cfg.get('anchor_weights', None)
            if weights_cfg is None:
                weights = torch.ones(len(anchors), device=primary_raw.device, dtype=primary_raw.dtype)
            else:
                weights = torch.as_tensor(list(weights_cfg), device=primary_raw.device, dtype=primary_raw.dtype)
                if weights.numel() != len(anchors):
                    raise ValueError(f'target_adapter.anchor_weights length {weights.numel()} does not match {len(anchors)} anchors.')
            weights = weights / weights.sum().clamp_min(1e-6)
            for anchor_tensor, projector, weight in zip(anchors, projectors, weights):
                projected = projector(anchor_tensor.to(primary_raw.device).float())
                if projected.shape[-1] != primary_raw.shape[-1]:
                    raise ValueError(
                        f'target_adapter anchor dim {projected.shape[-1]} does not match primary feature dim {primary_raw.shape[-1]}.'
                    )
                projected_anchors.append(weight * F.normalize(projected, dim=-1))
            anchor = torch.stack(projected_anchors, dim=0).sum(dim=0)
        else:
            anchor = batch['target_anchor_feature'].to(primary_raw.device).float()
            if self.target_anchor_proj is not None:
                anchor = self.target_anchor_proj(anchor)
        if anchor.shape[-1] != primary_raw.shape[-1]:
            raise ValueError(
                f'target_adapter anchor dim {anchor.shape[-1]} does not match primary feature dim {primary_raw.shape[-1]}.'
            )

        mix_key = 'train_anchor_mix' if self.training else 'eval_anchor_mix'
        mix_value = self.target_adapter_cfg.get(mix_key, self.target_adapter_cfg.get('anchor_mix', 0.0))
        max_mix = max(0.0, min(1.0, float(mix_value)))
        if max_mix <= 0:
            return primary_raw
        mix_mode = str(self.target_adapter_cfg.get('mix_mode', 'fixed')).lower()
        if mix_mode in {'score', 'inverse_score'}:
            score = self.brain.score(primary_raw.float())
            if bool(self.target_adapter_cfg.get('score_detach', True)):
                score = score.detach()
            if mix_mode == 'inverse_score':
                score = 1.0 - score
            mix = max_mix * score.clamp(0.0, 1.0)
        else:
            mix = torch.full(
                (primary_raw.shape[0], 1),
                max_mix,
                device=primary_raw.device,
                dtype=primary_raw.dtype,
            )

        primary_for_mix = primary_raw.float()
        if bool(self.target_adapter_cfg.get('normalize_before_mix', True)):
            primary_for_mix = F.normalize(primary_for_mix, dim=-1)
            anchor = F.normalize(anchor, dim=-1)
        adapted = (1.0 - mix) * primary_for_mix + mix * anchor
        if bool(self.target_adapter_cfg.get('normalize_after_mix', True)):
            adapted = F.normalize(adapted, dim=-1)
        return adapted.to(dtype=primary_raw.dtype)


    @staticmethod
    def _reduce_brain_stat(x: torch.Tensor) -> torch.Tensor:
        if x.ndim <= 1:
            return x.float().view(-1, 1)
        dims = tuple(range(1, x.ndim))
        return x.float().mean(dim=dims, keepdim=False).unsqueeze(-1)

    def _build_mctf_context(self, batch) -> dict:
        if not self.mctf_cfg['enabled']:
            return {}
        ctx = {}
        if 'subject' in batch and torch.is_tensor(batch['subject']):
            ctx['subject_ids'] = batch['subject'].to(self.device).long()
        eeg = batch.get('eeg')
        eeg_mean = batch.get('eeg_mean')
        if torch.is_tensor(eeg):
            eeg_f = eeg.float()
            dims = tuple(range(1, eeg_f.ndim)) if eeg_f.ndim > 1 else ()
            signal_std = eeg_f.std(dim=-1).mean(dim=-1, keepdim=True) if eeg_f.ndim >= 3 else eeg_f.std(dim=-1, keepdim=True)
            signal_energy = eeg_f.abs().mean(dim=dims, keepdim=False).unsqueeze(-1) if dims else eeg_f.abs().view(-1, 1)
            if torch.is_tensor(eeg_mean):
                eeg_mean_f = eeg_mean.float()
                diff = (eeg_f - eeg_mean_f).abs()
                consistency = diff.mean(dim=dims, keepdim=False).unsqueeze(-1) if dims else diff.view(-1, 1)
                mean_energy = eeg_mean_f.abs().mean(dim=dims, keepdim=False).unsqueeze(-1) if dims else eeg_mean_f.abs().view(-1, 1)
            else:
                consistency = torch.zeros_like(signal_energy)
                mean_energy = signal_energy
            stats = torch.cat([signal_std, signal_energy, consistency, mean_energy], dim=-1)
            ctx['sample_depth_stats'] = stats.to(self.device)
        return ctx

    def _maybe_meg_mixup(self, batch):
        if str(self.config.get('dataset', '')).lower() != 'meg':
            return batch

        mix_cfg = self.train_cfg.get('mixup', {})
        if not bool(mix_cfg.get('enabled', False)):
            return batch

        current_epoch = int(getattr(self, 'current_epoch', 0))
        start_epoch = int(mix_cfg.get('start_epoch', 0))
        if current_epoch < start_epoch:
            return batch

        stop_epoch = mix_cfg.get('stop_epoch', None)
        stop_ratio = mix_cfg.get('stop_ratio', None)
        total_epochs = int(self.train_cfg.get('epoch', 0))
        if stop_epoch is None and stop_ratio is not None and total_epochs > 0:
            stop_epoch = max(start_epoch + 1, int(round(total_epochs * float(stop_ratio))))
        if stop_epoch is not None and current_epoch >= int(stop_epoch):
            return batch

        eeg = batch.get('eeg')
        if eeg is None or eeg.shape[0] < 2:
            return batch

        p = float(mix_cfg.get('p', 1.0))
        if torch.rand(1, device=eeg.device).item() > p:
            return batch

        alpha = float(mix_cfg.get('alpha', 0.2))
        lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 0.5

        min_lam = mix_cfg.get('min_lam', None)
        max_lam = mix_cfg.get('max_lam', None)
        if min_lam is not None or max_lam is not None:
            lo = 0.0 if min_lam is None else float(min_lam)
            hi = 1.0 if max_lam is None else float(max_lam)
            if lo > hi:
                lo, hi = hi, lo
            lam = min(max(lam, lo), hi)

        same_subject_only = bool(mix_cfg.get('same_subject_only', False))
        if same_subject_only and 'subject' in batch and torch.is_tensor(batch['subject']):
            subject_ids = batch['subject'].to(eeg.device).view(-1)
            perm = torch.arange(eeg.shape[0], device=eeg.device)

            for subject_id in torch.unique(subject_ids):
                idx = torch.nonzero(subject_ids == subject_id, as_tuple=False).flatten()
                if idx.numel() <= 1:
                    continue
                shuffled = idx[torch.randperm(idx.numel(), device=eeg.device)]
                perm[shuffled] = shuffled.roll(1)
        else:
            perm = torch.randperm(eeg.shape[0], device=eeg.device)

        mixed = dict(batch)
        for key in ('eeg', 'eeg_mean', 'visual_feature_primary', 'visual_feature_secondary', 'public_teacher_feature', 'target_anchor_feature'):
            if key in batch and torch.is_tensor(batch[key]):
                mixed[key] = lam * batch[key] + (1.0 - lam) * batch[key][perm]

        if 'visual_feature_levels' in batch and isinstance(batch['visual_feature_levels'], (list, tuple)):
            mixed['visual_feature_levels'] = [
                lam * feat + (1.0 - lam) * feat[perm]
                for feat in batch['visual_feature_levels']
            ]

        mixed['_multi_positive_valid'] = False
        return mixed

    def _similarity_fn(self, curv: torch.Tensor):
        return lambda query, target: -L.pairwise_dist(query, target, curv)

    def _apply_eval_rerank(self, batch, distance: torch.Tensor) -> torch.Tensor:
        cfg = dict(self.config.get('eval_rerank', {}) or {})
        if not bool(cfg.get('enabled', False)):
            return distance

        feature = None
        if 'target_anchor_feature' in batch:
            feature = batch['target_anchor_feature']
        elif 'public_teacher_feature' in batch:
            feature = batch['public_teacher_feature']
        elif bool(cfg.get('fallback_to_target', False)):
            feature = batch.get('visual_feature_primary')
        if feature is None:
            return distance

        feature = feature.to(distance.device, dtype=distance.dtype)
        if feature.ndim != 2 or feature.shape[0] != distance.shape[1]:
            return distance

        feature = F.normalize(feature, dim=-1)
        graph = (feature @ feature.T) / max(float(cfg.get('temperature', 0.08)), 1e-6)

        topk = int(cfg.get('topk', 0) or 0)
        if 0 < topk < graph.shape[-1]:
            values, indices = graph.topk(topk, dim=-1)
            masked = torch.full_like(graph, -float('inf'))
            graph = masked.scatter(-1, indices, values)

        transition = torch.softmax(graph, dim=-1)
        score = -distance
        smoothed_score = score @ transition
        weight = min(max(float(cfg.get('weight', 0.0)), 0.0), 1.0)
        return -((1.0 - weight) * score + weight * smoothed_score)

    def _apply_eval_calibration(self, batch, distance: torch.Tensor) -> torch.Tensor:
        cfg = dict(self.config.get('eval_calibration', {}) or {})
        if not bool(cfg.get('enabled', False)):
            return distance
        if distance.ndim != 2 or min(distance.shape) <= 1:
            return distance

        method = str(cfg.get('method', 'col_zmix')).lower()
        weight = min(max(float(cfg.get('weight', 0.25)), 0.0), 1.0)
        col_mean = distance.mean(dim=0, keepdim=True)
        col_std = distance.std(dim=0, keepdim=True).clamp_min(1e-6)

        if method == 'col_zmix':
            calibrated = (distance - col_mean) / col_std
        elif method == 'bizmix':
            row_mean = distance.mean(dim=1, keepdim=True)
            row_std = distance.std(dim=1, keepdim=True).clamp_min(1e-6)
            calibrated = 0.5 * ((distance - row_mean) / row_std + (distance - col_mean) / col_std)
        elif method == 'col_center':
            global_mean = distance.mean()
            calibrated = distance - (col_mean - global_mean)
        else:
            raise ValueError(f"Unsupported eval_calibration.method: {method}")

        return (1.0 - weight) * distance + weight * calibrated

    def _build_positive_mask(self, batch, device: torch.device) -> torch.Tensor | None:
        if not bool(self.retrieval_cfg.get('multi_positive_enabled', False)):
            return None
        if batch.get('_multi_positive_valid', True) is False:
            return None
        img_paths = batch.get('img_path')
        if img_paths is None:
            return None
        if torch.is_tensor(img_paths):
            return None
        img_paths = [str(path) for path in img_paths]
        if len(img_paths) == 0:
            return None
        mask = torch.zeros((len(img_paths), len(img_paths)), device=device, dtype=torch.bool)
        for row, path in enumerate(img_paths):
            for col, other in enumerate(img_paths):
                if path == other:
                    mask[row, col] = True
        diag = torch.arange(len(img_paths), device=device)
        mask[diag, diag] = True
        return mask

    def _compute_loss(self, batch, eeg_z: torch.Tensor, img_z: torch.Tensor, curv: torch.Tensor, *, use_memory: bool, update_memory: bool) -> torch.Tensor:
        scale = torch.as_tensor(
            float(self.config.get('retrieval', {}).get('logit_scale', HYPERBOLIC_LOGIT_SCALE)),
            device=eeg_z.device,
            dtype=eeg_z.dtype,
        )
        sample_ids = batch['idx'].to(eeg_z.device).long() if 'idx' in batch else None
        positive_mask = self._build_positive_mask(batch, eeg_z.device)
        return self.retrieval_objective(
            eeg_z,
            img_z,
            similarity_fn=self._similarity_fn(curv),
            sample_ids=sample_ids,
            positive_mask=positive_mask,
            scale=scale,
            use_memory=use_memory,
            update_memory=update_memory,
        )

    def _public_teacher_loss(self, batch, eeg_z: torch.Tensor, curv: torch.Tensor) -> torch.Tensor:
        zero = torch.tensor(0.0, device=eeg_z.device, dtype=eeg_z.dtype)
        if not self._scheduled_cfg_enabled(self.public_teacher_cfg):
            return zero
        if self.public_teacher_proj is None or 'public_teacher_feature' not in batch:
            return zero

        feat = batch['public_teacher_feature'].to(eeg_z.device).float()
        if bool(self.public_teacher_cfg.get('normalize', True)):
            feat = F.normalize(feat, dim=-1)
        proj = self.public_teacher_proj(feat)
        visual_alpha = torch.tensor(1.0, device=eeg_z.device, dtype=eeg_z.dtype)
        teacher_z = L.exp_map0(visual_alpha * proj, curv)
        return self._compute_loss(batch, eeg_z, teacher_z, curv, use_memory=False, update_memory=False)

    def _resolve_mctf(self, batch, visual_alpha: torch.Tensor, curv: torch.Tensor):
        if not self.mctf_cfg['enabled']:
            return None
        level_features = batch.get('visual_feature_levels')
        if level_features is None:
            raise ValueError('MCTF is enabled but batch does not contain visual_feature_levels.')
        if not isinstance(level_features, (list, tuple)):
            raise TypeError(f'visual_feature_levels must be a list/tuple, got {type(level_features)}')
        context = self._build_mctf_context(batch)
        return self.mctf(level_features, visual_alpha=visual_alpha, curv=curv, current_epoch=int(self.current_epoch), context=context)

    def _compute_mctf_loss(self, batch, eeg_z: torch.Tensor, img_z: torch.Tensor, curv: torch.Tensor, mctf_output: dict, *, use_memory: bool, update_memory: bool) -> dict:
        schedule = self.mctf_cfg.get('schedule', {})
        current_epoch = int(self.current_epoch)
        main_target = str(self.mctf_cfg.get('main_target', 'baseline')).lower()
        use_consensus_main = current_epoch >= int(schedule.get('consensus_start_epoch', 0)) and main_target in {'consensus', 'blend'}
        target_img_z = self._select_mctf_target_img_z(img_z, mctf_output, curv) if use_consensus_main else img_z
        main_loss = self._compute_loss(batch, eeg_z, target_img_z, curv, use_memory=use_memory, update_memory=update_memory)
        total = main_loss

        consensus_loss = torch.tensor(0.0, device=main_loss.device, dtype=main_loss.dtype)
        if self.mctf_cfg.get('use_consensus', True) and not use_consensus_main and float(self.mctf_cfg.get('consensus_weight', 0.0)) > 0:
            consensus_loss = self._compute_loss(batch, eeg_z, mctf_output['hyperbolic_consensus_target'], curv, use_memory=False, update_memory=False)
            total = total + float(self.mctf_cfg['consensus_weight']) * consensus_loss

        aux_loss = torch.tensor(0.0, device=main_loss.device, dtype=main_loss.dtype)
        per_level_loss = None
        if current_epoch >= int(schedule.get('aux_start_epoch', 0)) and self.mctf_cfg['use_aux_distill'] and self.mctf_cfg['aux_weight'] > 0:
            aux_loss, per_level_loss = self.mctf.aux_loss(
                self._compute_loss,
                batch,
                eeg_z,
                mctf_output['hyperbolic_depth_targets'],
                curv,
                use_memory=False,
                update_memory=False,
                mean_depth_posterior=mctf_output['mean_depth_posterior'],
            )
            total = total + float(self.mctf_cfg['aux_weight']) * aux_loss

        anchor_loss = torch.tensor(0.0, device=main_loss.device, dtype=main_loss.dtype)
        anchor_consensus_loss = torch.tensor(0.0, device=main_loss.device, dtype=main_loss.dtype)
        anchor_ready = current_epoch >= int(self.mctf_cfg.get('anchor_start_epoch', 0))
        if anchor_ready and (float(self.mctf_cfg.get('anchor_weight', 0.0)) > 0 or float(self.mctf_cfg.get('anchor_consensus_weight', 0.0)) > 0):
            target_proj = L.log_map0(img_z, curv).detach()
            target_norm = F.normalize(target_proj, dim=-1)
            level_anchor_losses = []
            for level_proj in mctf_output.get('depth_targets', []):
                level_norm = F.normalize(level_proj, dim=-1)
                level_anchor_losses.append(1.0 - F.cosine_similarity(level_norm, target_norm, dim=-1).mean())
            if level_anchor_losses:
                stacked_anchor = torch.stack(level_anchor_losses)
                anchor_weights = mctf_output['mean_depth_posterior'].detach().to(device=stacked_anchor.device, dtype=stacked_anchor.dtype)
                anchor_loss = (anchor_weights * stacked_anchor).sum()
                total = total + float(self.mctf_cfg.get('anchor_weight', 0.0)) * anchor_loss
            consensus_norm = F.normalize(mctf_output['consensus_target'], dim=-1)
            anchor_consensus_loss = 1.0 - F.cosine_similarity(consensus_norm, target_norm, dim=-1).mean()
            total = total + float(self.mctf_cfg.get('anchor_consensus_weight', 0.0)) * anchor_consensus_loss

        dcc_loss = torch.tensor(0.0, device=main_loss.device, dtype=main_loss.dtype)
        mpc_loss = torch.tensor(0.0, device=main_loss.device, dtype=main_loss.dtype)
        regularization_start_epoch = int(schedule.get('regularization_start_epoch', 0))
        dcc_start_epoch = int(schedule.get('dcc_start_epoch', regularization_start_epoch))
        mpc_start_epoch = int(schedule.get('mpc_start_epoch', regularization_start_epoch))
        if self.mctf_cfg['dcc_weight'] > 0 and current_epoch >= dcc_start_epoch:
            dcc_loss = mctf_output['dcc_loss']
            total = total + float(self.mctf_cfg['dcc_weight']) * dcc_loss
        if self.mctf_cfg['mpc_weight'] > 0 and current_epoch >= mpc_start_epoch:
            mpc_loss = mctf_output['mpc_loss']
            total = total + float(self.mctf_cfg['mpc_weight']) * mpc_loss

        return {
            'loss': total,
            'main_loss': main_loss.detach(),
            'consensus_loss': consensus_loss.detach(),
            'aux_loss': aux_loss.detach(),
            'anchor_loss': anchor_loss.detach(),
            'anchor_consensus_loss': anchor_consensus_loss.detach(),
            'dcc_loss': dcc_loss.detach(),
            'mpc_loss': mpc_loss.detach(),
            'per_level_loss': per_level_loss,
            'sample_depth_score': mctf_output.get('sample_depth_score'),
            'alpha_components': mctf_output.get('alpha_components'),
        }

    def _mctf_main_target_active(self) -> bool:
        if not self.mctf_cfg.get('enabled', False):
            return False
        schedule = self.mctf_cfg.get('schedule', {})
        current_epoch = int(self.current_epoch)
        main_target = str(self.mctf_cfg.get('main_target', 'baseline')).lower()
        return current_epoch >= int(schedule.get('consensus_start_epoch', 0)) and main_target in {'consensus', 'blend'}

    def _select_mctf_target_img_z(self, img_z: torch.Tensor, mctf_output: Optional[dict], curv: torch.Tensor) -> torch.Tensor:
        if mctf_output is None or not self._mctf_main_target_active():
            return img_z
        main_target = str(self.mctf_cfg.get('main_target', 'baseline')).lower()
        if main_target == 'consensus':
            return mctf_output['hyperbolic_consensus_target']
        if main_target == 'blend':
            mix = max(0.0, min(1.0, float(self.mctf_cfg.get('target_mix', 0.10))))
            warmup_epochs = max(0, int(self.mctf_cfg.get('target_mix_warmup_epochs', 0)))
            if warmup_epochs > 0:
                start_epoch = int(self.mctf_cfg.get('schedule', {}).get('consensus_start_epoch', 0))
                phase = (int(self.current_epoch) - start_epoch + 1) / float(warmup_epochs)
                mix = mix * max(0.0, min(1.0, phase))
            if mix <= 0:
                return img_z
            target_proj = L.log_map0(img_z, curv)
            blended_proj = (1.0 - mix) * target_proj + mix * mctf_output['consensus_target']
            return L.exp_map0(blended_proj, curv)
        return img_z

    def forward(self, batch, sample_posterior=False):
        eeg = batch['eeg']
        primary_raw = batch['visual_feature_primary']
        secondary_raw = batch['visual_feature_secondary']
        primary_for_target = self._adapt_primary_feature(batch, primary_raw)

        eeg_alpha = self.brain.eeg_alpha.exp()
        # Visual embeddings use a fixed scale of 1.0.
        visual_alpha = torch.tensor(1.0, device=eeg.device, dtype=eeg.dtype)
        curv = self.brain.curv.exp()

        eeg_z = self.brain(eeg)
        eeg_z = L.exp_map0(eeg_alpha * eeg_z, curv)

        primary_input = self._normalize_if_needed(primary_for_target, self.ablation['normalize_primary_feature'])
        primary_proj = self.brain.image_projection1(primary_input)
        primary_z = L.exp_map0(visual_alpha * primary_proj, curv)

        score = self._mix_score(primary_for_target)
        if self.ablation['use_feature_interpolation']:
            secondary_input = self._normalize_if_needed(secondary_raw, self.ablation['normalize_secondary_feature'])
            secondary_proj = self.brain.image_projection2(secondary_input)
            secondary_z = L.exp_map0(visual_alpha * secondary_proj, curv)
            img_z = L.geodesic_interploation(primary_z, secondary_z, score, curv)
        else:
            img_z = primary_z

        mctf_output = self._resolve_mctf(batch, visual_alpha=visual_alpha, curv=curv)
        return eeg_z, img_z, curv, score, mctf_output

    def _compute_step_loss(self, batch, eeg_z: torch.Tensor, img_z: torch.Tensor, curv: torch.Tensor, mctf_output: dict):
        if mctf_output is None:
            loss = self._compute_loss(batch, eeg_z, img_z, curv, use_memory=self.retrieval_cfg['xbm_enabled'], update_memory=True)
            zero = torch.tensor(0.0, device=loss.device, dtype=loss.dtype)
            return {
                'loss': loss,
                'main_loss': loss.detach(),
                'aux_loss': zero,
                'anchor_loss': zero,
                'anchor_consensus_loss': zero,
                'dcc_loss': zero,
                'mpc_loss': zero,
                'per_level_loss': None,
            }
        return self._compute_mctf_loss(batch, eeg_z, img_z, curv, mctf_output, use_memory=self.retrieval_cfg['xbm_enabled'], update_memory=True)

    def _scheduled_cfg_enabled(self, cfg: dict) -> bool:
        if not bool(cfg.get('enabled', False)):
            return False

        current_epoch = int(getattr(self, 'current_epoch', 0))
        start_epoch = int(cfg.get('start_epoch', 0))
        if current_epoch < start_epoch:
            return False

        stop_epoch = cfg.get('stop_epoch', None)
        stop_ratio = cfg.get('stop_ratio', None)
        total_epochs = int(self.train_cfg.get('epoch', 0))
        if stop_epoch is None and stop_ratio is not None and total_epochs > 0:
            stop_epoch = max(start_epoch + 1, int(round(total_epochs * float(stop_ratio))))
        if stop_epoch is not None and current_epoch >= int(stop_epoch):
            return False
        return True

    def _subject_balanced_loss(self, batch, eeg_z: torch.Tensor, img_z: torch.Tensor, curv: torch.Tensor, mctf_output: dict) -> torch.Tensor:
        cfg = self.train_cfg.get('subject_balanced_loss', {})
        zero = torch.tensor(0.0, device=eeg_z.device, dtype=eeg_z.dtype)
        if not self._scheduled_cfg_enabled(cfg):
            return zero
        if mctf_output is not None:
            return zero

        subject_ids = batch.get('subject')
        if subject_ids is None or not torch.is_tensor(subject_ids):
            return zero
        subject_ids = subject_ids.to(eeg_z.device).view(-1)
        unique_subjects = torch.unique(subject_ids)
        if unique_subjects.numel() < 2:
            return zero

        min_count = int(cfg.get('min_count', 16))
        losses = []
        for subject_id in unique_subjects:
            mask = subject_ids == subject_id
            if int(mask.sum().item()) < min_count:
                continue
            sub_batch = {}
            if 'idx' in batch and torch.is_tensor(batch['idx']):
                sub_batch['idx'] = batch['idx'].to(eeg_z.device)[mask]
            losses.append(
                self._compute_loss(
                    sub_batch,
                    eeg_z[mask],
                    img_z[mask],
                    curv,
                    use_memory=False,
                    update_memory=False,
                )
            )
        if len(losses) < 2:
            return zero
        return torch.stack(losses).mean()

    def _top1_margin_loss(self, batch, eeg_z: torch.Tensor, img_z: torch.Tensor, curv: torch.Tensor, mctf_output: dict) -> torch.Tensor:
        cfg = self.train_cfg.get('top1_margin_loss', {})
        zero = torch.tensor(0.0, device=eeg_z.device, dtype=eeg_z.dtype)
        if not self._scheduled_cfg_enabled(cfg):
            return zero

        target_z = self._select_mctf_target_img_z(img_z, mctf_output, curv)
        distance = L.pairwise_dist(eeg_z, target_z, curv)
        batch_size = int(distance.shape[0])
        if batch_size <= 1:
            return zero

        diag = torch.arange(batch_size, device=distance.device)
        positive = distance[diag, diag]
        masked = distance.masked_fill(torch.eye(batch_size, device=distance.device, dtype=torch.bool), float('inf'))
        hard_topk = min(int(cfg.get('hard_topk', 1)), batch_size - 1)
        margin = float(cfg.get('margin', 0.02))
        temperature = float(cfg.get('temperature', 0.05))

        hard_q = masked.topk(hard_topk, dim=1, largest=False).values
        loss_q = F.softplus((positive.unsqueeze(1) - hard_q + margin) / temperature).mean() * temperature
        if not bool(cfg.get('symmetric', True)):
            return loss_q

        hard_t = masked.t().topk(hard_topk, dim=1, largest=False).values
        loss_t = F.softplus((positive.unsqueeze(1) - hard_t + margin) / temperature).mean() * temperature
        return 0.5 * (loss_q + loss_t)

    def _rank_calibration_direction(self, logits: torch.Tensor, positive_mask: torch.Tensor, cfg: dict) -> torch.Tensor:
        positive_mask = positive_mask.to(device=logits.device, dtype=torch.bool)
        pos_weight = positive_mask.to(dtype=logits.dtype)
        pos_weight = pos_weight / pos_weight.sum(dim=1, keepdim=True).clamp_min(1.0)
        pos_score = (logits * pos_weight).sum(dim=1)
        neg_logits = logits.masked_fill(positive_mask, -torch.inf)
        max_neg = neg_logits.max(dim=1).values
        max_neg = torch.where(torch.isfinite(max_neg), max_neg, pos_score.detach())

        rank_temperature = max(1e-6, float(cfg.get('rank_temperature', 1.0)))
        beta = max(1e-6, float(cfg.get('beta', 5.0)))
        margin = float(cfg.get('margin', 0.0))
        comparisons = torch.sigmoid((neg_logits - pos_score[:, None] + margin) / rank_temperature)
        comparisons = comparisons.masked_fill(positive_mask, 0.0)
        soft_rank = 1.0 + comparisons.sum(dim=1)
        topk_risk = F.softplus((soft_rank - float(cfg.get('topk', 5))) / beta) * beta
        listwise_risk = -(F.log_softmax(logits, dim=1) * pos_weight).sum(dim=1)
        loss_vec = float(cfg.get('topk_weight', 1.0)) * topk_risk + float(cfg.get('listwise_weight', 0.10)) * listwise_risk

        if bool(cfg.get('reliability_enabled', True)):
            with torch.no_grad():
                rel_temp = max(1e-6, float(cfg.get('reliability_temperature', 1.0)))
                margin_conf = torch.sigmoid((pos_score - max_neg) / rel_temp)
                entropy_weight = float(cfg.get('reliability_entropy_weight', 0.35))
                if entropy_weight > 0:
                    probs = F.softmax(logits, dim=1)
                    entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=1)
                    entropy = entropy / torch.log(torch.tensor(float(logits.shape[1]), device=logits.device, dtype=logits.dtype))
                    reliability = (1.0 - entropy_weight) * margin_conf + entropy_weight * (1.0 - entropy.clamp(0.0, 1.0))
                else:
                    reliability = margin_conf
                reliability = reliability.clamp(float(cfg.get('reliability_min_weight', 0.35)), 1.0)
            loss_vec = loss_vec * reliability
        return loss_vec.mean()

    def _rank_calibration_loss(self, batch, eeg_z: torch.Tensor, img_z: torch.Tensor, curv: torch.Tensor, mctf_output: dict) -> torch.Tensor:
        cfg = self.train_cfg.get('rank_calibration_loss', {})
        zero = torch.tensor(0.0, device=eeg_z.device, dtype=eeg_z.dtype)
        if not self._scheduled_cfg_enabled(cfg):
            return zero

        target_z = self._select_mctf_target_img_z(img_z, mctf_output, curv)
        eeg_t = F.normalize(L.log_map0(eeg_z, curv), dim=-1)
        target_t = F.normalize(L.log_map0(target_z, curv), dim=-1)
        logits = float(self.retrieval_cfg.get('logit_scale', HYPERBOLIC_LOGIT_SCALE)) * eeg_t @ target_t.t()
        positive_mask = self._build_positive_mask(batch, logits.device)
        if positive_mask is None:
            positive_mask = torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
        loss_q = self._rank_calibration_direction(logits, positive_mask, cfg)
        if not bool(cfg.get('symmetric', True)):
            return loss_q
        loss_t = self._rank_calibration_direction(logits.t(), positive_mask.t(), cfg)
        return 0.5 * (loss_q + loss_t)

    def _rank_calibration_weight(self) -> float:
        cfg = self.train_cfg.get('rank_calibration_loss', {})
        if not self._scheduled_cfg_enabled(cfg):
            return 0.0
        weight = float(cfg.get('weight', 0.0))
        warmup_epochs = int(cfg.get('warmup_epochs', 0))
        if warmup_epochs > 0:
            current_epoch = int(getattr(self, 'current_epoch', 0))
            start_epoch = int(cfg.get('start_epoch', 0))
            progress = min(1.0, max(0.0, (current_epoch - start_epoch + 1) / float(warmup_epochs)))
            weight *= progress
        return weight

    def _domain_alignment_loss(self, batch, eeg_z: torch.Tensor, img_z: torch.Tensor, curv: torch.Tensor) -> torch.Tensor:
        cfg = self.train_cfg.get('domain_alignment', {})
        zero = torch.tensor(0.0, device=eeg_z.device, dtype=eeg_z.dtype)
        if not self._scheduled_cfg_enabled(cfg):
            return zero

        subject_ids = batch.get('subject')
        if subject_ids is None or not torch.is_tensor(subject_ids):
            return zero
        subject_ids = subject_ids.to(eeg_z.device).view(-1)
        unique_subjects = torch.unique(subject_ids)
        if unique_subjects.numel() < 2:
            return zero

        eeg_t = L.log_map0(eeg_z, curv)
        img_t = L.log_map0(img_z, curv)
        target = str(cfg.get('target', 'residual')).lower()
        if target == 'eeg':
            features = eeg_t
        elif target == 'image':
            features = img_t
        elif target == 'pair':
            features = torch.cat([eeg_t, img_t], dim=-1)
        else:
            features = eeg_t - img_t
        if bool(cfg.get('normalize', True)):
            features = F.normalize(features, dim=-1)

        min_count = int(cfg.get('min_count', 8))
        means = []
        variances = []
        for subject_id in unique_subjects:
            mask = subject_ids == subject_id
            if int(mask.sum().item()) < min_count:
                continue
            group = features[mask]
            means.append(group.mean(dim=0))
            if float(cfg.get('variance_weight', 0.0)) > 0:
                variances.append(group.var(dim=0, unbiased=False))
        if len(means) < 2:
            return zero

        means = torch.stack(means, dim=0)
        mean_center = means.mean(dim=0, keepdim=True)
        loss = (means - mean_center).pow(2).mean()

        variance_weight = float(cfg.get('variance_weight', 0.0))
        if variance_weight > 0 and len(variances) == len(means):
            variances = torch.stack(variances, dim=0)
            var_center = variances.mean(dim=0, keepdim=True)
            loss = loss + variance_weight * (variances - var_center).pow(2).mean()
        return loss

    def _consistency_weight(self) -> float:
        cfg = self.train_cfg.get('consistency_loss', {})
        if not self._scheduled_cfg_enabled(cfg):
            return 0.0
        weight = float(cfg.get('weight', 0.0))
        warmup_epochs = int(cfg.get('warmup_epochs', 0))
        if warmup_epochs > 0:
            current_epoch = int(getattr(self, 'current_epoch', 0))
            start_epoch = int(cfg.get('start_epoch', 0))
            progress = min(1.0, max(0.0, (current_epoch - start_epoch + 1) / float(warmup_epochs)))
            weight *= progress
        return weight

    def _make_consistency_view(self, eeg: torch.Tensor, cfg: dict) -> torch.Tensor:
        x = eeg.clone()
        if x.ndim < 3:
            return x
        batch_size, c_num, t_num = int(x.shape[0]), int(x.shape[-2]), int(x.shape[-1])
        device = x.device

        scale_std = float(cfg.get('scale_std', 0.0))
        if scale_std > 0:
            shape = [batch_size] + [1] * (x.ndim - 1)
            scale = 1.0 + torch.randn(shape, device=device, dtype=x.dtype) * scale_std
            x = x * scale

        noise_std = float(cfg.get('noise_std', 0.0))
        if noise_std > 0:
            x = x + torch.randn_like(x) * noise_std

        max_shift = min(int(cfg.get('time_shift', 0)), max(1, t_num - 1))
        if max_shift > 0:
            shifted = x.clone()
            shifts = torch.randint(-max_shift, max_shift + 1, (batch_size,), device=device)
            for row, shift in enumerate(shifts.tolist()):
                if shift != 0:
                    shifted[row] = torch.roll(x[row], shifts=int(shift), dims=-1)
            x = shifted

        channel_mask_num = min(int(cfg.get('channel_mask_num', 0)), c_num)
        if channel_mask_num > 0:
            for row in range(batch_size):
                idx = torch.randperm(c_num, device=device)[:channel_mask_num]
                x[row, idx] = 0.0

        width = min(int(cfg.get('time_mask_width', 0)), t_num)
        mask_num = int(cfg.get('time_mask_num', 0))
        if width > 0 and mask_num > 0:
            max_start = max(1, t_num - width + 1)
            for row in range(batch_size):
                for _ in range(mask_num):
                    start = int(torch.randint(0, max_start, (1,), device=device).item())
                    x[row, :, start:start + width] = 0.0
        return x

    def _consistency_loss(self, batch, eeg_z: torch.Tensor, img_z: torch.Tensor, curv: torch.Tensor, mctf_output: dict) -> torch.Tensor:
        cfg = self.train_cfg.get('consistency_loss', {})
        zero = torch.tensor(0.0, device=eeg_z.device, dtype=eeg_z.dtype)
        if not self._scheduled_cfg_enabled(cfg):
            return zero
        if float(cfg.get('p', 1.0)) < 1.0 and torch.rand(1, device=eeg_z.device).item() > float(cfg.get('p', 1.0)):
            return zero

        aug_eeg = self._make_consistency_view(batch['eeg'], cfg)
        eeg_alpha = self.brain.eeg_alpha.exp()
        aug_eeg_z = self.brain(aug_eeg)
        aug_eeg_z = L.exp_map0(eeg_alpha * aug_eeg_z, curv)

        target_z = self._select_mctf_target_img_z(img_z, mctf_output, curv)
        if bool(cfg.get('detach_target', True)):
            target_z = target_z.detach()

        total = zero
        retrieval_weight = float(cfg.get('retrieval_weight', 1.0))
        if retrieval_weight > 0:
            total = total + retrieval_weight * self._compute_loss(
                batch,
                aug_eeg_z,
                target_z,
                curv,
                use_memory=False,
                update_memory=False,
            )

        embedding_weight = float(cfg.get('embedding_weight', 0.0))
        if embedding_weight > 0:
            aug_t = F.normalize(L.log_map0(aug_eeg_z, curv), dim=-1)
            clean_t = F.normalize(L.log_map0(eeg_z.detach(), curv.detach()), dim=-1)
            embed_loss = 1.0 - F.cosine_similarity(aug_t, clean_t, dim=-1).mean()
            total = total + embedding_weight * embed_loss
        return total

    def training_step(self, batch, batch_idx):
        batch = self._maybe_meg_mixup(batch)
        batch_size = batch['idx'].shape[0]
        eeg_z, img_z, curv, score, mctf_output = self(batch, sample_posterior=True)
        loss_dict = self._compute_step_loss(batch, eeg_z, img_z, curv, mctf_output)
        domain_align_loss = self._domain_alignment_loss(batch, eeg_z, img_z, curv)
        top1_margin_loss = self._top1_margin_loss(batch, eeg_z, img_z, curv, mctf_output)
        rank_calibration_loss = self._rank_calibration_loss(batch, eeg_z, img_z, curv, mctf_output)
        consistency_loss = self._consistency_loss(batch, eeg_z, img_z, curv, mctf_output)
        subject_balanced_loss = self._subject_balanced_loss(batch, eeg_z, img_z, curv, mctf_output)
        subject_balanced_weight = float(self.train_cfg.get('subject_balanced_loss', {}).get('weight', 0.0))
        if subject_balanced_weight > 0 and subject_balanced_loss.requires_grad:
            loss = (1.0 - subject_balanced_weight) * loss_dict['loss'] + subject_balanced_weight * subject_balanced_loss
        else:
            loss = loss_dict['loss']
        loss = loss + float(self.train_cfg.get('domain_alignment', {}).get('weight', 0.0)) * domain_align_loss
        loss = loss + float(self.train_cfg.get('top1_margin_loss', {}).get('weight', 0.0)) * top1_margin_loss
        rank_calibration_weight = self._rank_calibration_weight()
        loss = loss + rank_calibration_weight * rank_calibration_loss
        consistency_weight = self._consistency_weight()
        loss = loss + consistency_weight * consistency_loss
        public_teacher_loss = self._public_teacher_loss(batch, eeg_z, curv)
        public_teacher_weight = float(self.public_teacher_cfg.get('weight', 0.0)) if self.public_teacher_enabled else 0.0
        loss = loss + public_teacher_weight * public_teacher_loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)
        if self.public_teacher_enabled:
            self.log('train_public_teacher_loss', public_teacher_loss.detach(), on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True, batch_size=batch_size)
        if bool(self.train_cfg.get('domain_alignment', {}).get('enabled', False)):
            self.log('train_domain_align_loss', domain_align_loss.detach(), on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True, batch_size=batch_size)
        if bool(self.train_cfg.get('top1_margin_loss', {}).get('enabled', False)):
            self.log('train_top1_margin_loss', top1_margin_loss.detach(), on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True, batch_size=batch_size)
        if bool(self.train_cfg.get('rank_calibration_loss', {}).get('enabled', False)):
            self.log('train_rank_calibration_loss', rank_calibration_loss.detach(), on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True, batch_size=batch_size)
        if bool(self.train_cfg.get('consistency_loss', {}).get('enabled', False)):
            self.log('train_consistency_loss', consistency_loss.detach(), on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True, batch_size=batch_size)
            self.log('train_consistency_weight', torch.tensor(consistency_weight, device=loss.device), on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True, batch_size=batch_size)
        if bool(self.train_cfg.get('subject_balanced_loss', {}).get('enabled', False)):
            self.log('train_subject_balanced_loss', subject_balanced_loss.detach(), on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True, batch_size=batch_size)
        if mctf_output is not None:
            self.log('train_main_loss', loss_dict['main_loss'], on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True, batch_size=batch_size)
            self.log('train_consensus_loss', loss_dict.get('consensus_loss', torch.tensor(0.0, device=loss.device, dtype=loss.dtype)), on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True, batch_size=batch_size)
            self.log('train_aux_loss', loss_dict['aux_loss'], on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True, batch_size=batch_size)
            self.log('train_anchor_loss', loss_dict['anchor_loss'], on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True, batch_size=batch_size)
            self.log('train_anchor_consensus_loss', loss_dict['anchor_consensus_loss'], on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True, batch_size=batch_size)
            self.log('train_dcc_loss', loss_dict['dcc_loss'], on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True, batch_size=batch_size)
            self.log('train_mpc_loss', loss_dict['mpc_loss'], on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True, batch_size=batch_size)
            sample_depth_score = loss_dict.get('sample_depth_score')
            if sample_depth_score is not None:
                self.log('train_sample_depth_score', sample_depth_score.mean().detach(), on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True, batch_size=batch_size)
            for idx, alpha_value in enumerate(mctf_output['mean_depth_posterior']):
                self.log(f'train_alpha/{self.level_names[idx]}', alpha_value.detach(), on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True, batch_size=batch_size)

        eval_img_z = self._select_mctf_target_img_z(img_z, mctf_output, curv)
        similarity = L.pairwise_dist(eeg_z, eval_img_z, curv)
        _, top_k_indices = similarity.topk(5, largest=False, dim=-1)
        self.all_predicted_classes.append(top_k_indices.cpu().numpy())
        label = torch.arange(0, batch_size).to(self.device)
        self.all_true_labels.extend(label.cpu().numpy())

        if batch_idx == self.trainer.num_training_batches - 1:
            all_predicted_classes = np.concatenate(self.all_predicted_classes, axis=0)
            all_true_labels = np.array(self.all_true_labels)
            top_1_predictions = all_predicted_classes[:, 0]
            top_1_correct = top_1_predictions == all_true_labels
            top_1_accuracy = sum(top_1_correct) / len(top_1_correct)
            top_k_correct = (all_predicted_classes == all_true_labels[:, np.newaxis]).any(axis=1)
            top_k_accuracy = sum(top_k_correct) / len(top_k_correct)
            self.log('train_top1_acc', top_1_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log('train_top5_acc', top_k_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.all_predicted_classes = []
            self.all_true_labels = []

        return loss

    def validation_step(self, batch, batch_idx):
        batch_size = batch['idx'].shape[0]
        eeg_z, img_z, curv, score, mctf_output = self(batch, sample_posterior=True)
        if mctf_output is None:
            loss = self._compute_loss(batch, eeg_z, img_z, curv, use_memory=False, update_memory=False)
        else:
            loss_dict = self._compute_mctf_loss(batch, eeg_z, img_z, curv, mctf_output, use_memory=False, update_memory=False)
            loss = loss_dict['loss']
            self.last_depth_posterior = mctf_output['mean_depth_posterior'].detach().cpu()
            if mctf_output.get('sample_depth_score') is not None:
                self.sample_depth_score_history.append({
                    'epoch': int(self.current_epoch),
                    'mean_sample_depth_score': float(mctf_output['sample_depth_score'].mean().detach().cpu().item())
                })
            if mctf_output.get('alpha_components') is not None:
                comp_entry = {'epoch': int(self.current_epoch)}
                for comp_name in ('global_depth_logits', 'subject_depth_bias', 'sample_depth_bias', 'progressive_log_prior'):
                    comp_tensor = mctf_output['alpha_components'][comp_name]
                    comp_entry[comp_name] = [float(v) for v in comp_tensor.mean(dim=0).detach().cpu().tolist()]
                self.depth_logit_component_history.append(comp_entry)
        eval_img_z = self._select_mctf_target_img_z(img_z, mctf_output, curv)
        similarity = L.pairwise_dist(eeg_z, eval_img_z, curv)
        similarity = self._apply_eval_calibration(batch, self._apply_eval_rerank(batch, similarity))
        _, top_k_indices = similarity.topk(5, largest=False, dim=-1)

        self.all_predicted_classes.append(top_k_indices.cpu().numpy())
        label = torch.arange(0, batch_size).to(self.device)
        self.all_true_labels.extend(label.cpu().numpy())

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)
        return loss

    def on_validation_epoch_end(self):
        all_predicted_classes = np.concatenate(self.all_predicted_classes, axis=0)
        all_true_labels = np.array(self.all_true_labels)
        top_1_predictions = all_predicted_classes[:, 0]
        top_1_correct = top_1_predictions == all_true_labels
        top_1_accuracy = sum(top_1_correct) / len(top_1_correct)
        top_k_correct = (all_predicted_classes == all_true_labels[:, np.newaxis]).any(axis=1)
        top_k_accuracy = sum(top_k_correct) / len(top_k_correct)
        self.log('val_top1_acc', top_1_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_top5_acc', top_k_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        val_score = self.train_cfg['val_top1_weight'] * top_1_accuracy + self.train_cfg['val_top5_weight'] * top_k_accuracy
        self.log('val_score', val_score, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        if self.mctf_cfg['enabled'] and self.last_depth_posterior is not None:
            alpha_entry = {'epoch': int(self.current_epoch)}
            for idx, alpha_value in enumerate(self.last_depth_posterior.tolist()):
                alpha_entry[self.level_names[idx]] = float(alpha_value)
                self.log(f'val_alpha/{self.level_names[idx]}', float(alpha_value), on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
            self.depth_posterior_history.append(alpha_entry)
        self.all_predicted_classes = []
        self.all_true_labels = []

    def on_test_epoch_start(self):
        self.all_predicted_classes = []
        self.all_true_labels = []
        self.depth_posterior_test_batches = []
        self.match_similarities = []
        self.mAP_total = 0

    def test_step(self, batch, batch_idx):
        batch_size = batch['idx'].shape[0]
        if batch_size != 200:
            raise ValueError(
                f'Published evaluation requires one batch of 200 averaged stimuli; got {batch_size}. '
                'Use test_avg=true and test_batch_size=200.'
            )
        eeg_z, img_z, curv, score, mctf_output = self(batch, sample_posterior=True)
        if mctf_output is None:
            loss = self._compute_loss(batch, eeg_z, img_z, curv, use_memory=False, update_memory=False)
            self.score = score.mean()
        else:
            loss_dict = self._compute_mctf_loss(batch, eeg_z, img_z, curv, mctf_output, use_memory=False, update_memory=False)
            loss = loss_dict['loss']
            self.score = mctf_output['mean_depth_posterior'].mean()
            self.depth_posterior_test_batches.append(mctf_output['mean_depth_posterior'].detach().cpu())

        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)

        label = torch.arange(0, batch_size).to(self.device)
        eval_img_z = self._select_mctf_target_img_z(img_z, mctf_output, curv)
        similarity = L.pairwise_dist(eeg_z, eval_img_z, curv)
        similarity = self._apply_eval_calibration(batch, self._apply_eval_rerank(batch, similarity))
        _, top_k_indices = similarity.topk(5, largest=False, dim=-1)

        self.all_true_labels.extend(label.cpu().numpy())
        self.all_predicted_classes.append(top_k_indices.cpu().numpy())
        self.match_similarities.extend(similarity.diag().detach().cpu().tolist())

        for i in range(similarity.shape[0]):
            true_index = i
            sims = similarity[i, :]
            sorted_indices = torch.argsort(sims)
            rank = (sorted_indices == true_index).nonzero()[0][0] + 1
            ap = 1 / rank
            self.mAP_total += ap

        return loss

    def _write_analysis_files(self):
        if not getattr(self, 'logger', None):
            return
        log_dir = getattr(self.logger, 'log_dir', None)
        if not log_dir:
            return
        analysis_dir = os.path.join(log_dir, 'analysis_mctf')
        os.makedirs(analysis_dir, exist_ok=True)
        if self.mctf_cfg['enabled'] and self.mctf_cfg.get('save_alpha_history', False) and self.depth_posterior_history:
            with open(os.path.join(analysis_dir, 'depth_posterior_history.json'), 'w', encoding='utf-8') as f:
                json.dump(self.depth_posterior_history, f, indent=2, ensure_ascii=False)
        if self.mctf_cfg['enabled'] and self.sample_depth_score_history:
            with open(os.path.join(analysis_dir, 'sample_depth_score_history.json'), 'w', encoding='utf-8') as f:
                json.dump(self.sample_depth_score_history, f, indent=2, ensure_ascii=False)
        if self.mctf_cfg['enabled'] and self.depth_logit_component_history:
            with open(os.path.join(analysis_dir, 'depth_logit_component_history.json'), 'w', encoding='utf-8') as f:
                json.dump(self.depth_logit_component_history, f, indent=2, ensure_ascii=False)
        if self.mctf_cfg['enabled'] and self.depth_posterior_test_batches:
            alpha_mean = torch.stack(self.depth_posterior_test_batches, dim=0).mean(dim=0)
            alpha_summary = {name: float(value) for name, value in zip(self.level_names, alpha_mean.tolist())}
            with open(os.path.join(analysis_dir, 'test_depth_posterior_summary.json'), 'w', encoding='utf-8') as f:
                json.dump(alpha_summary, f, indent=2, ensure_ascii=False)

    def on_fit_end(self):
        self._write_analysis_files()

    def on_test_epoch_end(self):
        all_predicted_classes = np.concatenate(self.all_predicted_classes, axis=0)
        all_true_labels = np.array(self.all_true_labels)

        top_1_predictions = all_predicted_classes[:, 0]
        top_1_correct = top_1_predictions == all_true_labels
        top_1_accuracy = sum(top_1_correct) / len(top_1_correct)

        top_k_correct = (all_predicted_classes == all_true_labels[:, np.newaxis]).any(axis=1)
        top_k_accuracy = sum(top_k_correct) / len(top_k_correct)

        self.mAP = (self.mAP_total / len(all_true_labels)).item()
        self.match_similarities = np.mean(self.match_similarities) if self.match_similarities else 0

        self.log('test_top1_acc', top_1_accuracy, sync_dist=True)
        self.log('test_top5_acc', top_k_accuracy, sync_dist=True)
        test_score = self.train_cfg['val_top1_weight'] * top_1_accuracy + self.train_cfg['val_top5_weight'] * top_k_accuracy
        self.log('test_score', test_score, sync_dist=True)
        self.log('mAP', self.mAP, sync_dist=True)
        self.log('similarity', self.match_similarities, sync_dist=True)
        self.log('score', self.score, sync_dist=True)
        self._write_analysis_files()

        self.all_predicted_classes = []
        self.all_true_labels = []

        avg_test_loss = self.trainer.callback_metrics['test_loss']
        result = {
            'test_loss': avg_test_loss.item(),
            'test_top1_acc': top_1_accuracy.item(),
            'test_top5_acc': top_k_accuracy.item(),
            'mAP': self.mAP,
            'similarity': self.match_similarities,
            'score': self.score,
        }
        if self.mctf_cfg['enabled'] and self.depth_posterior_test_batches:
            alpha_mean = torch.stack(self.depth_posterior_test_batches, dim=0).mean(dim=0)
            result['alpha_mean'] = {name: float(value) for name, value in zip(self.level_names, alpha_mean.tolist())}
        if self.mctf_cfg['enabled'] and self.sample_depth_score_history:
            result['mean_sample_depth_score'] = float(np.mean([entry['mean_sample_depth_score'] for entry in self.sample_depth_score_history]))
        return result

    def configure_optimizers(self):
        optimizer = globals()[self.config['train']['optimizer']](
            self.parameters(),
            lr=self.config['train']['lr'],
            weight_decay=float(self.train_cfg['weight_decay'])
        )
        scheduler_cfg = self.train_cfg.get('scheduler', {})
        if not bool(scheduler_cfg.get('enabled', False)):
            return [optimizer]

        total_steps = int(getattr(self.trainer, 'estimated_stepping_batches', 0))
        if total_steps <= 0:
            return [optimizer]

        scheduler = build_warmup_cosine_scheduler(
            optimizer,
            total_steps=total_steps,
            warmup_ratio=float(scheduler_cfg.get('warmup_ratio', 0.1)),
            min_lr_ratio=float(scheduler_cfg.get('min_lr_ratio', 0.05)),
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            },
        }

