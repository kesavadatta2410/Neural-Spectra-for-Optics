"""
SpecCompress-India: Encoder
1D CNN → MLP hybrid with temperature conditioning.
Input  : [B, n_points] spectrum + [B] temperature
Output : [B, latent_dim] compressed vector
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class TemperatureEmbedding(nn.Module):
    """
    Scalar temperature → dense embedding.
    Normalised to [0,1] over [25°C, 45°C] range.
    """
    def __init__(self, embed_dim: int = 16, temp_min: float = 25.0, temp_max: float = 45.0):
        super().__init__()
        self.temp_min  = temp_min
        self.temp_max  = temp_max
        self.embed     = nn.Sequential(
            nn.Linear(1, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, temperature: torch.Tensor) -> torch.Tensor:
        """
        Args:
            temperature: [B] raw °C values
        Returns:
            [B, embed_dim]
        """
        t_norm = (temperature - self.temp_min) / (self.temp_max - self.temp_min + 1e-8)
        t_norm = t_norm.unsqueeze(-1).float()          # [B, 1]
        return self.embed(t_norm)                       # [B, embed_dim]


class SpectralCNNBlock(nn.Module):
    """Conv1d + BN + GELU residual-style block."""
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1, dropout: float = 0.1):
        super().__init__()
        pad = kernel // 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=pad),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        # Projection shortcut if shape changes
        if in_ch != out_ch or stride != 1:
            self.proj = nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride)
        else:
            self.proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.proj is not None:
            x = self.proj(x)
            # Align lengths after strided conv
            if x.shape[-1] != out.shape[-1]:
                x = F.adaptive_avg_pool1d(x, out.shape[-1])
        return out + x


class Encoder(nn.Module):
    """
    Architecture:
        [B, 1, n_points]
          → SpectralCNNBlock × 3  (downsample via stride)
          → GlobalAvgPool + Flatten
          → Concat temperature embedding
          → MLP → [B, latent_dim]
    Temperature is injected AFTER CNN, before MLP bottleneck.
    This lets CNN learn shape-agnostic features; MLP learns
    temperature-conditioned projection to latent space.
    """
    def __init__(
        self,
        n_points:       int = 1000,
        latent_dim:     int = 32,
        temp_embed_dim: int = 16,
        cnn_channels:   List[int] = [1, 32, 64, 128],
        kernel_sizes:   List[int] = [7, 5, 3],
        strides:        List[int] = [2, 2, 2],
        mlp_hidden:     List[int] = [512, 256],
        dropout:        float = 0.1,
        temp_min:       float = 25.0,
        temp_max:       float = 45.0
    ):
        super().__init__()
        self.latent_dim = latent_dim

        # Temperature embedding
        self.temp_embed = TemperatureEmbedding(temp_embed_dim, temp_min, temp_max)

        # 1D CNN blocks
        assert len(cnn_channels) - 1 == len(kernel_sizes) == len(strides)
        cnn_blocks = []
        for i in range(len(kernel_sizes)):
            cnn_blocks.append(SpectralCNNBlock(
                cnn_channels[i], cnn_channels[i + 1],
                kernel=kernel_sizes[i], stride=strides[i], dropout=dropout
            ))
        self.cnn = nn.Sequential(*cnn_blocks)

        # Compute CNN output dim dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_points)
            cnn_out = self.cnn(dummy)
            cnn_flat_dim = cnn_out.shape[1] * cnn_out.shape[2]

        # MLP with temperature conditioning
        mlp_in = cnn_flat_dim + temp_embed_dim
        layers  = []
        prev_d  = mlp_in
        for h in mlp_hidden:
            layers += [nn.Linear(prev_d, h), nn.GELU(), nn.Dropout(dropout)]
            prev_d = h
        layers.append(nn.Linear(prev_d, latent_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, spectrum: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spectrum    : [B, n_points]
            temperature : [B]  (°C)
        Returns:
            z           : [B, latent_dim]
        """
        x = spectrum.unsqueeze(1)              # [B, 1, n_points]
        x = self.cnn(x)                        # [B, C, L']
        x = x.flatten(1)                       # [B, C*L']
        t = self.temp_embed(temperature)       # [B, temp_embed_dim]
        x = torch.cat([x, t], dim=-1)          # [B, C*L' + temp_embed_dim]
        z = self.mlp(x)                        # [B, latent_dim]
        return z
