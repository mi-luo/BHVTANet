"""STAE brain encoder."""

import numpy as np
import torch
import torch.nn as nn
from .eegproject import EEGProject
BrainEncoder = EEGProject


class STAE(BrainEncoder):
    def __init__(self, z_dim, c_num, timesteps, drop_proj=0.3, image_dim=None):
        super().__init__(
            z_dim=z_dim,
            c_num=c_num,
            timesteps=timesteps,
            drop_proj=drop_proj,
            image_dim=image_dim
        )
        self.time_len = timesteps[1] - timesteps[0]
        self.temporal_attention = nn.Parameter(torch.ones(self.time_len))

    def forward(self, x):
        b = x.shape[0]
        x = x.reshape(b, self.c_num, self.time_len)
        w = torch.softmax(self.temporal_attention, dim=0)
        x = x * w.reshape(1, 1, self.time_len)
        x = x.reshape(b, self.input_dim)
        x = self.model(x)
        return x
