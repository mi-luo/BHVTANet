import copy
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


DEFAULT_RETRIEVAL: Dict[str, Any] = {
    'preset': 'original',
    'objective': 'ce',
    'xbm_enabled': False,
    'xbm_size': 4096,
    'xbm_min_size': 64,
    'vsepp_margin': 0.2,
    'vsepp_topk': 1,
    'normalize_query': False,
    'normalize_target': False,
    'multi_positive_enabled': False,
    'multi_positive_mode': 'positive',
}

RETRIEVAL_PRESETS: Dict[str, Dict[str, Any]] = {
    'original': {
        'objective': 'ce',
        'xbm_enabled': False,
        'normalize_query': False,
        'normalize_target': False,
    },
    'xbm': {
        'objective': 'ce',
        'xbm_enabled': True,
        'normalize_query': False,
        'normalize_target': False,
    },
    'siglip': {
        'objective': 'siglip',
        'xbm_enabled': False,
        'normalize_query': True,
        'normalize_target': True,
    },
    'vsepp': {
        'objective': 'vsepp',
        'xbm_enabled': False,
        'normalize_query': True,
        'normalize_target': True,
    },
    'xbm_siglip': {
        'objective': 'siglip',
        'xbm_enabled': True,
        'normalize_query': True,
        'normalize_target': True,
    },
    'xbm_vsepp': {
        'objective': 'vsepp',
        'xbm_enabled': True,
        'normalize_query': True,
        'normalize_target': True,
    },
}

_VALID_OBJECTIVES = {'ce', 'siglip', 'vsepp'}


# Source inspirations:
# - SigLIP pairwise sigmoid loss follows the public paper/code pattern from Google DeepMind's SigLIP.
# - XBM queue structure follows the public Cross-Batch Memory implementations in research-xbm and pytorch-metric-learning.
# - VSE++ hardest-negative ranking follows the public PyTorch logic from fartashf/vsepp.


def _to_plain_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    if hasattr(obj, 'items'):
        return {k: v for k, v in obj.items()}
    return {}


def resolve_retrieval_config(config: Dict[str, Any]) -> Dict[str, Any]:
    raw = _to_plain_dict(config.get('retrieval', {}))
    preset = str(raw.get('preset', DEFAULT_RETRIEVAL['preset'])).lower()

    resolved = copy.deepcopy(DEFAULT_RETRIEVAL)
    if preset in RETRIEVAL_PRESETS:
        resolved.update(copy.deepcopy(RETRIEVAL_PRESETS[preset]))
    resolved.update({k: v for k, v in raw.items() if v is not None})
    resolved['preset'] = preset

    objective = str(resolved.get('objective', 'ce')).lower()
    if objective not in _VALID_OBJECTIVES:
        raise ValueError(f'Unsupported retrieval objective: {objective}')
    resolved['objective'] = objective
    resolved['xbm_enabled'] = bool(resolved.get('xbm_enabled', False))
    resolved['xbm_size'] = max(1, int(resolved.get('xbm_size', 4096)))
    resolved['xbm_min_size'] = max(1, int(resolved.get('xbm_min_size', 64)))
    resolved['vsepp_margin'] = float(resolved.get('vsepp_margin', 0.2))
    resolved['vsepp_topk'] = max(1, int(resolved.get('vsepp_topk', 1)))
    resolved['normalize_query'] = bool(resolved.get('normalize_query', False))
    resolved['normalize_target'] = bool(resolved.get('normalize_target', False))
    resolved['label_smoothing'] = float(resolved.get('label_smoothing', 0.0))
    resolved['multi_positive_enabled'] = bool(resolved.get('multi_positive_enabled', False))
    resolved['multi_positive_mode'] = str(resolved.get('multi_positive_mode', 'positive')).lower()
    if resolved['multi_positive_mode'] not in {'positive', 'mask_false_negatives'}:
        raise ValueError(f"Unsupported multi_positive_mode: {resolved['multi_positive_mode']}")
    return resolved


class CrossBatchMemory:
    def __init__(self, size: int):
        self.size = int(size)
        self.query_queue: Optional[torch.Tensor] = None
        self.target_queue: Optional[torch.Tensor] = None
        self.id_queue: Optional[torch.Tensor] = None
        self.ptr = 0
        self.valid = 0

    def reset(self) -> None:
        self.query_queue = None
        self.target_queue = None
        self.id_queue = None
        self.ptr = 0
        self.valid = 0

    def has_items(self, min_size: int = 1) -> bool:
        return self.valid >= int(min_size) and self.query_queue is not None and self.target_queue is not None

    def _ensure_storage(self, query: torch.Tensor, target: torch.Tensor, ids: Optional[torch.Tensor]) -> None:
        if self.query_queue is not None:
            return
        qdim = query.shape[-1]
        tdim = target.shape[-1]
        device = query.device
        self.query_queue = torch.zeros((self.size, qdim), device=device, dtype=query.dtype)
        self.target_queue = torch.zeros((self.size, tdim), device=device, dtype=target.dtype)
        self.id_queue = torch.full((self.size,), -1, device=device, dtype=torch.long)
        if ids is None:
            self.id_queue = None

    def enqueue(self, query: torch.Tensor, target: torch.Tensor, ids: Optional[torch.Tensor]) -> None:
        batch_size = int(query.shape[0])
        if batch_size == 0:
            return

        if ids is not None:
            ids = ids.detach().long().to(query.device)
        self._ensure_storage(query, target, ids)
        assert self.query_queue is not None and self.target_queue is not None

        query = query.detach()
        target = target.detach()

        if batch_size >= self.size:
            self.query_queue.copy_(query[-self.size:])
            self.target_queue.copy_(target[-self.size:])
            if self.id_queue is not None and ids is not None:
                self.id_queue.copy_(ids[-self.size:])
            self.ptr = 0
            self.valid = self.size
            return

        first = min(self.size - self.ptr, batch_size)
        second = batch_size - first

        self.query_queue[self.ptr:self.ptr + first] = query[:first]
        self.target_queue[self.ptr:self.ptr + first] = target[:first]
        if self.id_queue is not None and ids is not None:
            self.id_queue[self.ptr:self.ptr + first] = ids[:first]

        if second > 0:
            self.query_queue[0:second] = query[first:]
            self.target_queue[0:second] = target[first:]
            if self.id_queue is not None and ids is not None:
                self.id_queue[0:second] = ids[first:]

        self.ptr = (self.ptr + batch_size) % self.size
        self.valid = min(self.size, self.valid + batch_size)

    def get(self) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if not self.has_items():
            raise RuntimeError('CrossBatchMemory is empty.')
        assert self.query_queue is not None and self.target_queue is not None

        # Clone queue values so enqueue() cannot mutate tensors used by autograd.
        if self.valid < self.size:
            query = self.query_queue[:self.valid].detach().clone()
            target = self.target_queue[:self.valid].detach().clone()
            ids = None if self.id_queue is None else self.id_queue[:self.valid].detach().clone()
            return query, target, ids

        query = self.query_queue.detach().clone()
        target = self.target_queue.detach().clone()
        ids = None if self.id_queue is None else self.id_queue.detach().clone()
        return query, target, ids


class RetrievalObjective(nn.Module):
    def __init__(self, retrieval_cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = resolve_retrieval_config({'retrieval': retrieval_cfg})
        self.memory = CrossBatchMemory(self.cfg['xbm_size']) if self.cfg['xbm_enabled'] else None
        self.neg_inf = -1e4

    @staticmethod
    def normalize(x: torch.Tensor) -> torch.Tensor:
        return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    def maybe_normalize_query(self, x: torch.Tensor) -> torch.Tensor:
        return self.normalize(x) if self.cfg['normalize_query'] else x

    def maybe_normalize_target(self, x: torch.Tensor) -> torch.Tensor:
        return self.normalize(x) if self.cfg['normalize_target'] else x

    def _build_memory_mask(self, sample_ids: Optional[torch.Tensor], memory_ids: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if sample_ids is None or memory_ids is None:
            return None
        return sample_ids.view(-1, 1).long().eq(memory_ids.view(1, -1).long())

    def _extend_similarity(self, current: torch.Tensor, memory: Optional[torch.Tensor], memory_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if memory is None:
            return current
        if memory_mask is None:
            return torch.cat([current, memory], dim=1)
        memory = memory.clone()
        memory = memory.masked_fill(memory_mask, self.neg_inf)
        return torch.cat([current, memory], dim=1)

    def _multi_positive_direction_loss(self, logits: torch.Tensor, positive_mask: torch.Tensor) -> torch.Tensor:
        positive_mask = positive_mask.to(device=logits.device, dtype=torch.bool)
        if positive_mask.shape != logits.shape:
            raise ValueError(f'positive_mask shape {tuple(positive_mask.shape)} does not match logits shape {tuple(logits.shape)}')
        positive_mask = positive_mask.clone()
        row_has_positive = positive_mask.any(dim=1)
        if not bool(row_has_positive.all().item()):
            diag = torch.arange(min(logits.shape[0], logits.shape[1]), device=logits.device)
            positive_mask[diag, diag] = True
        log_denom = torch.logsumexp(logits, dim=1)
        positive_logits = logits.masked_fill(~positive_mask, self.neg_inf)
        log_pos = torch.logsumexp(positive_logits, dim=1)
        return (log_denom - log_pos).mean()

    def _cross_entropy_loss(
        self,
        logits_qt: torch.Tensor,
        logits_tq: torch.Tensor,
        positive_mask_qt: Optional[torch.Tensor] = None,
        positive_mask_tq: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if positive_mask_qt is not None and positive_mask_tq is not None and self.cfg['multi_positive_enabled']:
            if self.cfg['multi_positive_mode'] == 'mask_false_negatives':
                labels = torch.arange(logits_qt.shape[0], device=logits_qt.device, dtype=torch.long)
                label_smoothing = float(self.cfg.get('label_smoothing', 0.0))
                false_neg_qt = positive_mask_qt.clone()
                false_neg_tq = positive_mask_tq.clone()
                diag_q = torch.arange(min(false_neg_qt.shape[0], false_neg_qt.shape[1]), device=logits_qt.device)
                diag_t = torch.arange(min(false_neg_tq.shape[0], false_neg_tq.shape[1]), device=logits_tq.device)
                false_neg_qt[diag_q, diag_q] = False
                false_neg_tq[diag_t, diag_t] = False
                loss_q = F.cross_entropy(logits_qt.masked_fill(false_neg_qt, self.neg_inf), labels, label_smoothing=label_smoothing)
                loss_t = F.cross_entropy(logits_tq.masked_fill(false_neg_tq, self.neg_inf), labels, label_smoothing=label_smoothing)
                return 0.5 * (loss_q + loss_t)
            loss_q = self._multi_positive_direction_loss(logits_qt, positive_mask_qt)
            loss_t = self._multi_positive_direction_loss(logits_tq, positive_mask_tq)
            return 0.5 * (loss_q + loss_t)

        labels = torch.arange(logits_qt.shape[0], device=logits_qt.device, dtype=torch.long)
        label_smoothing = float(self.cfg.get('label_smoothing', 0.0))
        loss_q = F.cross_entropy(logits_qt, labels, label_smoothing=label_smoothing)
        loss_t = F.cross_entropy(logits_tq, labels, label_smoothing=label_smoothing)
        return 0.5 * (loss_q + loss_t)

    def _siglip_direction_loss(self, logits: torch.Tensor, current_size: int, memory_mask: Optional[torch.Tensor]) -> torch.Tensor:
        labels = -torch.ones_like(logits)
        diag_index = torch.arange(current_size, device=logits.device)
        labels[diag_index, diag_index] = 1.0

        weights = torch.ones_like(logits)
        if memory_mask is not None:
            weights[:, current_size:] = (~memory_mask).to(weights.dtype)

        loss = -F.logsigmoid(labels * logits) * weights
        denom = weights.sum().clamp_min(1.0)
        return loss.sum() / denom

    def _siglip_loss(self, logits_qt: torch.Tensor, logits_tq: torch.Tensor, current_size: int, memory_mask_qt: Optional[torch.Tensor], memory_mask_tq: Optional[torch.Tensor]) -> torch.Tensor:
        loss_q = self._siglip_direction_loss(logits_qt, current_size, memory_mask_qt)
        loss_t = self._siglip_direction_loss(logits_tq, current_size, memory_mask_tq)
        return 0.5 * (loss_q + loss_t)

    def _vsepp_direction_loss(self, sims: torch.Tensor, current_size: int, margin: float, memory_mask: Optional[torch.Tensor]) -> torch.Tensor:
        positive = sims[:, :current_size].diagonal().unsqueeze(1)
        cost = (margin + sims - positive).clamp_min(0.0)
        diag_mask = torch.eye(current_size, device=sims.device, dtype=torch.bool)
        cost[:, :current_size] = cost[:, :current_size].masked_fill(diag_mask, 0.0)
        if memory_mask is not None:
            cost[:, current_size:] = cost[:, current_size:].masked_fill(memory_mask, 0.0)

        topk = min(int(self.cfg.get('vsepp_topk', 1)), int(cost.shape[1]))
        if topk <= 1:
            return cost.max(dim=1)[0].mean()

        # Average several hard negatives to reduce noisy single-negative dominance
        # in cross-subject retrieval while preserving the VSE++ ranking signal.
        return cost.topk(topk, dim=1).values.mean(dim=1).mean()

    def _vsepp_loss(self, logits_qt: torch.Tensor, logits_tq: torch.Tensor, current_size: int, memory_mask_qt: Optional[torch.Tensor], memory_mask_tq: Optional[torch.Tensor]) -> torch.Tensor:
        margin = float(self.cfg['vsepp_margin'])
        loss_q = self._vsepp_direction_loss(logits_qt, current_size, margin, memory_mask_qt)
        loss_t = self._vsepp_direction_loss(logits_tq, current_size, margin, memory_mask_tq)
        return 0.5 * (loss_q + loss_t)

    def forward(
        self,
        query_embed: torch.Tensor,
        target_embed: torch.Tensor,
        *,
        similarity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        sample_ids: Optional[torch.Tensor] = None,
        positive_mask: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
        use_memory: bool = False,
        update_memory: bool = False,
    ) -> torch.Tensor:
        current_qt = similarity_fn(query_embed, target_embed)
        current_tq = similarity_fn(target_embed, query_embed)

        memory_qt = None
        memory_tq = None
        memory_mask_qt = None
        memory_mask_tq = None

        if use_memory and self.memory is not None and self.memory.has_items(self.cfg['xbm_min_size']):
            mem_query, mem_target, mem_ids = self.memory.get()
            memory_qt = similarity_fn(query_embed, mem_target)
            memory_tq = similarity_fn(target_embed, mem_query)
            memory_mask_qt = self._build_memory_mask(sample_ids, mem_ids)
            memory_mask_tq = self._build_memory_mask(sample_ids, mem_ids)

        logits_qt = self._extend_similarity(current_qt, memory_qt, memory_mask_qt)
        logits_tq = self._extend_similarity(current_tq, memory_tq, memory_mask_tq)
        positive_mask_qt = None
        positive_mask_tq = None
        if positive_mask is not None:
            positive_mask = positive_mask.to(dtype=torch.bool, device=logits_qt.device)
            positive_mask_qt = positive_mask
            positive_mask_tq = positive_mask.t()

        if positive_mask_qt is not None and positive_mask_qt.shape[1] != logits_qt.shape[1]:
            pad = logits_qt.shape[1] - positive_mask_qt.shape[1]
            if pad < 0:
                raise ValueError(f'positive_mask has too many columns: {positive_mask_qt.shape[1]} > {logits_qt.shape[1]}')
            positive_mask_qt = F.pad(positive_mask_qt, (0, pad), value=False)
        if positive_mask_tq is not None and positive_mask_tq.shape[1] != logits_tq.shape[1]:
            pad = logits_tq.shape[1] - positive_mask_tq.shape[1]
            if pad < 0:
                raise ValueError(f'positive_mask has too many columns: {positive_mask_tq.shape[1]} > {logits_tq.shape[1]}')
            positive_mask_tq = F.pad(positive_mask_tq, (0, pad), value=False)

        objective = self.cfg['objective']
        if objective in {'ce', 'siglip'} and scale is not None:
            logits_qt = logits_qt * scale
            logits_tq = logits_tq * scale

        current_size = int(query_embed.shape[0])
        if objective == 'ce':
            loss = self._cross_entropy_loss(logits_qt, logits_tq, positive_mask_qt=positive_mask_qt, positive_mask_tq=positive_mask_tq)
        elif objective == 'siglip':
            loss = self._siglip_loss(logits_qt, logits_tq, current_size, memory_mask_qt, memory_mask_tq)
        elif objective == 'vsepp':
            loss = self._vsepp_loss(logits_qt, logits_tq, current_size, memory_mask_qt, memory_mask_tq)
        else:
            raise ValueError(f'Unsupported retrieval objective: {objective}')

        if update_memory and self.memory is not None:
            self.memory.enqueue(query_embed, target_embed, sample_ids)
        return loss
