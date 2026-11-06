"""EEGProject brain encoder."""

import numpy as np
import torch
import torch.nn as nn
from .encoder_layers import ResidualAdd


class EEGProject(nn.Module):
    """Linear residual brain encoder for EEG and MEG alignment."""

    def __init__(self,  z_dim,c_num, timesteps, drop_proj=0.3, image_dim=None):
        super().__init__()
        self.z_dim = z_dim
        self.c_num = c_num
        self.timesteps = timesteps
        self.image_dim = int(image_dim) if image_dim is not None else z_dim

        self.input_dim = self.c_num * (self.timesteps[1]-self.timesteps[0])
        proj_dim = z_dim

        self.model = nn.Sequential(nn.Linear(self.input_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim)
            )

        self.logit_scale = nn.Parameter(torch.tensor(100).log())
        self.softplus = nn.Softplus()
        self.eeg_alpha = nn.Parameter(torch.tensor(z_dim**-0.5).log())
        self.visual_alpha = nn.Parameter(torch.tensor(z_dim**-0.5).log())    
        self.curv = nn.Parameter(torch.tensor(1).log(), requires_grad=True)

        self.image_layer1 = nn.Linear(self.image_dim, proj_dim)
        self.image_layer2 = nn.Linear(self.image_dim, proj_dim)
        self.score_layer = nn.Linear(self.image_dim, 1)
        

    def image_projection1(self, x):
        return self.image_layer1(x)

    def image_projection2(self, x):
        return self.image_layer2(x)
    
    def score(self, x):
        x = self.score_layer(x)
        return torch.sigmoid(x)
    
    def forward(self, x):
        x = x.view(x.shape[0], self.input_dim)
        x = self.model(x)
        return x
