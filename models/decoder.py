"""
SpecCompress-India: Decoder
Latent z + temperature → reconstructed spectrum.
MLP-based: simple, stable, fast on Colab T4.
"""

import torch
import torch.nn as nn
from typing import List
from models.encoder import TemperatureEmbedding


class Decoder(nn.Module):
    """
    Architecture:
        z [B, latent_dim] + temp_embed [B, temp_embed_dim]
          → concat → MLP → [B, n_points]
    Temperature re-injected at decode step:
    allows decoder to "undo" temperature effects
    and reconstruct the canonical spectrum.
    """
    def __init__(
        self,
        n_points:       int = 1000,
        latent_dim:     int = 32,
        temp_embed_dim: int = 16,
        mlp_hidden:     List[int] = [256, 512, 1024],
        dropout:        float = 0.1,
        temp_min:       float = 25.0,
        temp_max:       float = 45.0
    ):
        super().__init__()
        self.n_points = n_points

        self.temp_embed = TemperatureEmbedding(temp_embed_dim, temp_min, temp_max)

        mlp_in  = latent_dim + temp_embed_dim
        layers  = []
        prev_d  = mlp_in
        for h in mlp_hidden:
            layers += [nn.Linear(prev_d, h), nn.GELU(), nn.Dropout(dropout)]
            prev_d = h
        layers.append(nn.Linear(prev_d, n_points))
        self.mlp = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z           : [B, latent_dim]
            temperature : [B]  (°C)
        Returns:
            x_hat       : [B, n_points]
        """
        t     = self.temp_embed(temperature)           # [B, temp_embed_dim]
        inp   = torch.cat([z, t], dim=-1)              # [B, latent_dim + temp_embed_dim]
        x_hat = self.mlp(inp)                          # [B, n_points]
        return x_hat
