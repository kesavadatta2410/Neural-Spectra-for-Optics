"""
SpecCompress-India: Full Autoencoder Model
Wires Encoder + Decoder with physics-informed loss.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

from models.encoder import Encoder
from models.decoder import Decoder
from models.physics_loss import PhysicsInformedLoss


class SpecCompress(nn.Module):
    """
    Physics-Informed Neural Spectral Compressor.

    encode(spectrum, temperature) → z [latent_dim]
    decode(z, temperature)        → spectrum_hat
    forward(spectrum, temperature) → (spectrum_hat, z, loss, components)
    """

    def __init__(self, cfg: dict):
        super().__init__()
        m   = cfg["model"]
        lss = cfg["loss"]

        self.encoder = Encoder(
            n_points       = m["input_dim"],
            latent_dim     = m["latent_dim"],
            temp_embed_dim = m["temp_embed_dim"],
            cnn_channels   = m["encoder"]["channels"],
            kernel_sizes   = m["encoder"]["kernel_sizes"],
            strides        = m["encoder"]["strides"],
            mlp_hidden     = m["bottleneck"]["hidden_dims"],
            dropout        = m["encoder"]["dropout"],
        )
        self.decoder = Decoder(
            n_points       = m["input_dim"],
            latent_dim     = m["latent_dim"],
            temp_embed_dim = m["temp_embed_dim"],
            mlp_hidden     = m["decoder"]["hidden_dims"],
            dropout        = m["decoder"]["dropout"],
        )
        self.loss_fn = PhysicsInformedLoss(
            reconstruction_weight     = lss["reconstruction_weight"],
            smoothness_weight         = lss["smoothness_weight"],
            power_conservation_weight = lss["power_conservation_weight"],
            osnr_penalty_weight       = lss["osnr_penalty_weight"],
        )

    def encode(self, spectrum: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
        return self.encoder(spectrum, temperature)

    def decode(self, z: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
        return self.decoder(z, temperature)

    def forward(
        self,
        spectrum:    torch.Tensor,
        temperature: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        z       = self.encode(spectrum, temperature)
        x_hat   = self.decode(z, temperature)
        loss, components = self.loss_fn(x_hat, spectrum)
        return x_hat, z, loss, components

    def compression_ratio(self) -> float:
        n_in  = self.encoder.mlp[-1].in_features   # approx
        n_lat = self.decoder.mlp[0].in_features
        return n_in / n_lat

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
