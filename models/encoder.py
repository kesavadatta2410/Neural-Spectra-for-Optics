"""
SpecCompress-India: Encoder  (v2 — ablation-ready)
1D CNN → MLP hybrid with temperature conditioning.
Supports: no-CNN (MLP-only), no-temp variants for ablation.
Input  : [B, n_points] spectrum + [B] temperature
Output : [B, latent_dim] compressed vector
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class TemperatureEmbedding(nn.Module):
    def __init__(self, embed_dim: int = 16, temp_min: float = 25.0, temp_max: float = 45.0):
        super().__init__()
        self.temp_min = temp_min
        self.temp_max = temp_max
        self.embed = nn.Sequential(
            nn.Linear(1, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, temperature: torch.Tensor) -> torch.Tensor:
        t = (temperature - self.temp_min) / (self.temp_max - self.temp_min + 1e-8)
        t = t.unsqueeze(-1).float()
        return self.embed(t)


class SpectralCNNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, dropout=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=kernel//2),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.proj = nn.Conv1d(in_ch, out_ch, 1, stride=stride) if (in_ch != out_ch or stride != 1) else None

    def forward(self, x):
        out = self.conv(x)
        res = self.proj(x) if self.proj is not None else x
        if res.shape[-1] != out.shape[-1]:
            res = F.adaptive_avg_pool1d(res, out.shape[-1])
        return out + res


class Encoder(nn.Module):
    def __init__(
        self,
        n_points=1000, latent_dim=10, temp_embed_dim=16,
        cnn_channels=[1,32,64,128], kernel_sizes=[7,5,3], strides=[2,2,2],
        mlp_hidden=[512,256], dropout=0.1,
        temp_min=25.0, temp_max=45.0,
        use_cnn=True, use_temp=True,
    ):
        super().__init__()
        self.use_cnn = use_cnn
        self.use_temp = use_temp
        self.latent_dim = latent_dim

        if use_temp:
            self.temp_embed = TemperatureEmbedding(temp_embed_dim, temp_min, temp_max)
        t_dim = temp_embed_dim if use_temp else 0

        if use_cnn:
            blocks = []
            for i in range(len(kernel_sizes)):
                blocks.append(SpectralCNNBlock(
                    cnn_channels[i], cnn_channels[i+1],
                    kernel=kernel_sizes[i], stride=strides[i], dropout=dropout
                ))
            self.cnn = nn.Sequential(*blocks)
            with torch.no_grad():
                feat_dim = self.cnn(torch.zeros(1,1,n_points)).flatten(1).shape[1]
        else:
            self.cnn = None
            feat_dim = n_points

        mlp_in = feat_dim + t_dim
        layers, prev = [], mlp_in
        for h in mlp_hidden:
            layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, latent_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, spectrum, temperature):
        x = self.cnn(spectrum.unsqueeze(1)).flatten(1) if self.use_cnn else spectrum
        if self.use_temp:
            x = torch.cat([x, self.temp_embed(temperature)], dim=-1)
        return self.mlp(x)
