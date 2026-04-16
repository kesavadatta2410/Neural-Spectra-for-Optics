"""
SpecCompress-India: Full Autoencoder  (v2)
Factory supports ablation variants and latent_dim sweep.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple

from models.encoder import Encoder
from models.decoder import Decoder
from models.physics_loss import PhysicsInformedLoss


class SpecCompress(nn.Module):
    def __init__(self, cfg: dict, ablation: dict = None, ablation_variant: dict = None):
        super().__init__()
        m   = cfg["model"]
        lss = cfg["loss"]

        # Ablation overrides
        av = ablation or ablation_variant or {}
        use_cnn   = av.get("use_cnn",  True)
        use_temp  = av.get("use_temp_conditioning", True)
        use_sm    = av.get("use_smoothness_loss",   True)
        use_pwr   = av.get("use_power_loss",        True)
        use_osnr  = av.get("use_osnr_loss",         True)
        latent    = av.get("latent_dim", m["latent_dim"])

        self.encoder = Encoder(
            n_points=m["input_dim"], latent_dim=latent,
            temp_embed_dim=m["temp_embed_dim"],
            cnn_channels=m["encoder"]["channels"],
            kernel_sizes=m["encoder"]["kernel_sizes"],
            strides=m["encoder"]["strides"],
            mlp_hidden=m["bottleneck"]["hidden_dims"],
            dropout=m["encoder"]["dropout"],
            temp_min=m.get("temp_min", 25.0),
            temp_max=m.get("temp_max", 45.0),
            use_cnn=use_cnn, use_temp=use_temp,
        )
        self.decoder = Decoder(
            n_points=m["input_dim"], latent_dim=latent,
            temp_embed_dim=m["temp_embed_dim"],
            mlp_hidden=m["decoder"]["hidden_dims"],
            dropout=m["decoder"]["dropout"],
            temp_min=m.get("temp_min", 25.0),
            temp_max=m.get("temp_max", 45.0),
            use_temp=use_temp,
        )
        self.loss_fn = PhysicsInformedLoss(
            reconstruction_weight=lss["reconstruction_weight"],
            smoothness_weight=lss["smoothness_weight"],
            power_conservation_weight=lss["power_conservation_weight"],
            osnr_penalty_weight=lss["osnr_penalty_weight"],
            use_smoothness=use_sm, use_power=use_pwr, use_osnr=use_osnr,
        )
        self.latent_dim = latent
        self.n_points   = m["input_dim"]

    def encode(self, spectrum, temperature):
        return self.encoder(spectrum, temperature)

    def decode(self, z, temperature):
        return self.decoder(z, temperature)

    def forward(self, spectrum, temperature):
        z     = self.encode(spectrum, temperature)
        x_hat = self.decode(z, temperature)
        loss, comps = self.loss_fn(x_hat, spectrum)
        return x_hat, z, loss, comps

    @property
    def compression_ratio(self):
        return self.n_points / self.latent_dim

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
