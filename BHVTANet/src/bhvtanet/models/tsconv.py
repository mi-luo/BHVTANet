"""TSConv brain encoder."""

import numpy as np
import torch
import torch.nn as nn
from .encoder_layers import BaseModel


class TSConv(BaseModel):
    def __init__(self, z_dim, c_num, timesteps, drop_proj=0.5, image_dim=None):
        super().__init__(z_dim, c_num, timesteps, proj_dropout=drop_proj, image_dim=image_dim)
        self.backbone = nn.Sequential(
                nn.Conv2d(1, 40, (1, 25), (1, 1)),
                nn.AvgPool2d((1, 51), (1, 5)),
                nn.BatchNorm2d(40),
                nn.ELU(),
                nn.Conv2d(40, 40, (c_num, 1), (1, 1)),
                nn.BatchNorm2d(40),
                nn.ELU(),
                nn.Dropout(0.5),
            )
        self._build_project()
