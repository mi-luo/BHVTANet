"""Shared encoder layers."""

import numpy as np
import torch
import torch.nn as nn


class ResidualAdd(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return  x + self.f(x)


class FlattenHead(nn.Sequential):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        return x


class BaseModel(nn.Module):
    def __init__(self, z_dim, c_num, timesteps, embedding_dim=None, proj_dropout=0.5, image_dim=None):
        super(BaseModel, self).__init__()
        self.z_dim = z_dim
        self.c_num = c_num
        self.timesteps = timesteps
        self.image_dim = int(image_dim) if image_dim is not None else z_dim
        self.time_len = timesteps[1] - timesteps[0]
        self.embedding_dim = embedding_dim
        self.proj_dropout = proj_dropout

        self.backbone = None
        self.project = None
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.softplus = nn.Softplus()
        self.curv = nn.Parameter(torch.tensor(1).log(), requires_grad=True)
        self.eeg_alpha = nn.Parameter(torch.tensor(z_dim**-0.5).log())
        self.visual_alpha = nn.Parameter(torch.tensor(z_dim**-0.5).log())

        self.image_layer1 = nn.Linear(self.image_dim, z_dim)
        self.image_layer2 = nn.Linear(self.image_dim, z_dim)
        self.score_layer = nn.Linear(self.image_dim, 1)

    def _infer_embedding_dim(self):
        if self.backbone is None:
            raise RuntimeError('backbone must be created before inferring embedding_dim')
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.c_num, self.time_len)
            feat = self.backbone(dummy)
            return int(feat.reshape(1, -1).shape[-1])

    def _build_project(self, embedding_dim=None):
        if embedding_dim is None:
            embedding_dim = self.embedding_dim
        if embedding_dim is None:
            embedding_dim = self._infer_embedding_dim()
        self.embedding_dim = int(embedding_dim)
        self.project = nn.Sequential(
            FlattenHead(),
            nn.Linear(self.embedding_dim, self.z_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(self.z_dim, self.z_dim),
                nn.Dropout(self.proj_dropout))),
            nn.LayerNorm(self.z_dim)
        )

    def score(self, x):
        x = self.score_layer(x)
        return torch.sigmoid(x)

    def image_projection1(self, x):
        return self.image_layer1(x)

    def image_projection2(self, x):
        return self.image_layer2(x)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.backbone(x)
        x = self.project(x)
        return x
