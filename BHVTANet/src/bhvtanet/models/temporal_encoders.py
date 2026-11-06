"""Optional temporal encoder variants."""

import numpy as np
import torch
import torch.nn as nn
from .eegproject import EEGProject
BrainEncoder = EEGProject


class EEGTemporalGateLayer(BrainEncoder):
    def __init__(self, z_dim, c_num, timesteps, drop_proj=0.3, image_dim=None, gate_scale=0.05):
        super().__init__(
            z_dim=z_dim,
            c_num=c_num,
            timesteps=timesteps,
            drop_proj=drop_proj,
            image_dim=image_dim
        )
        self.time_len = timesteps[1] - timesteps[0]
        self.gate_scale = float(gate_scale)
        self.temporal_gate = nn.Parameter(torch.zeros(self.time_len))

    def forward(self, x):
        b = x.shape[0]
        x = x.reshape(b, self.c_num, self.time_len)
        gate = 1.0 + self.gate_scale * torch.tanh(self.temporal_gate)
        x = x * gate.reshape(1, 1, self.time_len)
        x = x.reshape(b, self.input_dim)
        x = self.model(x)
        return x


class EEGTemporalReweightLayer(BrainEncoder):
    def __init__(self, z_dim, c_num, timesteps, drop_proj=0.3, image_dim=None, reweight_scale=0.10):
        super().__init__(
            z_dim=z_dim,
            c_num=c_num,
            timesteps=timesteps,
            drop_proj=drop_proj,
            image_dim=image_dim
        )
        self.time_len = timesteps[1] - timesteps[0]
        self.reweight_scale = float(reweight_scale)
        self.temporal_logits = nn.Parameter(torch.zeros(self.time_len))

    def forward(self, x):
        b = x.shape[0]
        x = x.reshape(b, self.c_num, self.time_len)
        attention = torch.softmax(self.temporal_logits, dim=0) * self.time_len
        weights = 1.0 + self.reweight_scale * (attention - 1.0)
        x = x * weights.reshape(1, 1, self.time_len)
        x = x.reshape(b, self.input_dim)
        x = self.model(x)
        return x
