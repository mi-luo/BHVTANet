"""DSWR: paper component implementation."""

from typing import Dict
import torch
import torch.nn as nn


class DSWR(nn.Module):
    """Unweighted MPC and DCC terms of Depth-Structured Weight Regularization."""

    def forward(self, alpha: torch.Tensor) -> Dict[str, torch.Tensor]:
        dcc_loss = alpha.new_zeros(())
        if alpha.shape[-1] > 1:
            dcc_loss = (alpha[:, 1:] - alpha[:, :-1]).pow(2).mean()
        mpc_loss = (alpha.clamp_min(1e-8) * alpha.clamp_min(1e-8).log()).sum(dim=-1).mean()
        return {'mpc_loss': mpc_loss, 'dcc_loss': dcc_loss}
