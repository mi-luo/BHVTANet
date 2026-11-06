"""EEGNet brain encoder."""

import numpy as np
import torch
import torch.nn as nn
from .encoder_layers import BaseModel


class EEGNet(BaseModel):
    def __init__(self,  z_dim, c_num, timesteps, drop_proj=0.5, image_dim=None):
        super().__init__(z_dim, c_num, timesteps, proj_dropout=drop_proj, image_dim=image_dim)
        self.backbone = nn.Sequential(
                nn.Conv2d(1, 8, (1, 64), (1, 1)),
                nn.BatchNorm2d(8),
                nn.Conv2d(8, 16, (c_num, 1), (1, 1)),
                nn.BatchNorm2d(16),
                nn.ELU(),
                nn.AvgPool2d((1, 2), (1, 2)),
                nn.Dropout(0.5),
                nn.Conv2d(16, 16, (1, 16), (1, 1)),
                nn.BatchNorm2d(16), 
                nn.ELU(),
                # nn.AvgPool2d((1, 2), (1, 2)),
                nn.Dropout2d(0.5)
            )
        self._build_project()
