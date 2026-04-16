"""
SpecCompress-India: Decoder  (v2 — ablation-ready)
Latent z + temperature → reconstructed spectrum.
use_temp=False for no-temperature-conditioning ablation.
"""

import torch
import torch.nn as nn
from typing import List
from models.encoder import TemperatureEmbedding


class Decoder(nn.Module):
    def __init__(
        self,
        n_points=1000, latent_dim=10, temp_embed_dim=16,
        mlp_hidden=[256,512,1024], dropout=0.1,
        temp_min=25.0, temp_max=45.0, use_temp=True,
    ):
        super().__init__()
        self.use_temp = use_temp
        if use_temp:
            self.temp_embed = TemperatureEmbedding(temp_embed_dim, temp_min, temp_max)
        t_dim = temp_embed_dim if use_temp else 0

        mlp_in = latent_dim + t_dim
        layers, prev = [], mlp_in
        for h in mlp_hidden:
            layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_points))
        self.mlp = nn.Sequential(*layers)

    def forward(self, z, temperature):
        if self.use_temp:
            t = self.temp_embed(temperature)
            z = torch.cat([z, t], dim=-1)
        return self.mlp(z)
