"""DACAM: paper component implementation."""

from typing import Optional, Sequence
import torch
import torch.nn as nn


class DACAM(nn.Module):
    """Depth-Adaptive Consensus Aggregation Module (paper Eqs. 13-15)."""

    def forward(self, global_depth_logits: torch.Tensor,
                progressive_log_prior: torch.Tensor,
                sample_depth_bias: torch.Tensor,
                hard_mask: Optional[torch.Tensor] = None,
                subject_depth_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        logits = global_depth_logits + progressive_log_prior
        if subject_depth_bias is not None:
            logits = logits + subject_depth_bias
        logits = logits + sample_depth_bias
        if hard_mask is not None:
            if hard_mask.shape != logits.shape:
                raise RuntimeError(f'Progressive mask shape {tuple(hard_mask.shape)} does not match logits {tuple(logits.shape)}')
            if (hard_mask.sum(dim=-1) <= 0).any():
                raise RuntimeError('Progressive depth prior produced an empty active set.')
            logits = logits.masked_fill(hard_mask <= 0, torch.finfo(logits.dtype).min)
        return torch.softmax(logits, dim=-1)

    @staticmethod
    def aggregate(depth_targets: Sequence[torch.Tensor],
                  alpha: torch.Tensor) -> torch.Tensor:
        return sum(alpha[:, index:index + 1] * target for index, target in enumerate(depth_targets))
