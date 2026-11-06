"""MDTP: paper component implementation."""

from typing import List, Sequence
import torch
import torch.nn as nn


class MDTP(nn.ModuleList):
    """Multi-Depth Target Projection Module (paper Section 3.2.1).

    Each ordered descriptor is optionally normalized and projected by its own
    linear layer. The projections follow the visual encoder's depth order.
    """

    def __init__(self, level_input_dims: Sequence[int], z_dim: int, normalize_levels: Sequence[bool]):
        super().__init__([nn.Linear(int(dim), z_dim) for dim in level_input_dims])
        self.normalize_levels = [bool(flag) for flag in normalize_levels]

    def forward(self, level_features: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        if len(level_features) != len(self):
            raise ValueError(f'Expected {len(self)} depth descriptors, got {len(level_features)}')
        depth_targets = []
        for depth_descriptor, depth_projector, normalize in zip(level_features, self, self.normalize_levels):
            if normalize:
                depth_descriptor = depth_descriptor / depth_descriptor.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            depth_targets.append(depth_projector(depth_descriptor))
        return depth_targets
