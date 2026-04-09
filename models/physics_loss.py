"""
SpecCompress-India: Physics-Informed Loss
Four physics priors embedded in training objective:
  1. Reconstruction  — MSE(x_hat, x)
  2. Smoothness      — penalise spectral discontinuities
  3. Power conservation — integral(x_hat) ≈ integral(x)
  4. OSNR penalty    — low-power regions should stay smooth
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class PhysicsInformedLoss(nn.Module):
    """
    Combined physics-informed loss for spectral compression.

    L_total = w_rec  * L_recon
            + w_sm   * L_smooth
            + w_pwr  * L_power
            + w_osnr * L_osnr
    """

    def __init__(
        self,
        reconstruction_weight:    float = 1.0,
        smoothness_weight:        float = 0.1,
        power_conservation_weight: float = 0.05,
        osnr_penalty_weight:      float = 0.02,
    ):
        super().__init__()
        self.w_rec  = reconstruction_weight
        self.w_sm   = smoothness_weight
        self.w_pwr  = power_conservation_weight
        self.w_osnr = osnr_penalty_weight

    # ── Component losses ──────────────────────────

    @staticmethod
    def reconstruction_loss(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Standard MSE — primary fidelity term."""
        return F.mse_loss(x_hat, x)

    @staticmethod
    def smoothness_loss(x_hat: torch.Tensor) -> torch.Tensor:
        """
        TV-style smoothness: penalise first-order spectral differences.
        Physics rationale: EDFA gain spectra are smooth;
        high-frequency artefacts are non-physical.
        L_smooth = mean( |x_hat[i+1] - x_hat[i]|^2 )
        """
        diff = x_hat[:, 1:] - x_hat[:, :-1]   # [B, L-1]
        return (diff ** 2).mean()

    @staticmethod
    def power_conservation_loss(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Integral of reconstructed gain ≈ integral of true gain.
        Physics: total signal power (proportional to gain integral)
        must be conserved through compression/decompression.
        Uses trapezoidal sum approximation (uniform spacing).
        """
        # Trapezoid rule: sum of (a+b)/2 → proportional to mean for uniform spacing
        power_true = x.mean(dim=-1)              # [B]
        power_hat  = x_hat.mean(dim=-1)          # [B]
        return F.mse_loss(power_hat, power_true)

    @staticmethod
    def osnr_penalty_loss(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        OSNR-aware penalty:
        In low-power spectral regions (noise floor), reconstruction
        errors are amplified in dB-OSNR terms. Penalise errors
        weighted by inverse of true power (proxy for OSNR sensitivity).
        L_osnr = mean( |x_hat - x|^2 / (|x| + ε) )
        Approximation: avoids computing full OSNR from raw signal.
        """
        eps     = 0.1                            # Avoid div-by-zero
        weights = 1.0 / (torch.abs(x) + eps)    # Higher weight for low-power regions
        weights = weights / weights.mean()       # Normalise
        return (weights * (x_hat - x) ** 2).mean()

    # ─────────────────────────────────────────────

    def forward(
        self,
        x_hat: torch.Tensor,
        x:     torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            x_hat : [B, n_points]  reconstructed
            x     : [B, n_points]  ground truth
        Returns:
            total_loss : scalar tensor
            components : dict of individual losses (float, detached)
        """
        L_recon  = self.reconstruction_loss(x_hat, x)
        L_smooth = self.smoothness_loss(x_hat)
        L_power  = self.power_conservation_loss(x_hat, x)
        L_osnr   = self.osnr_penalty_loss(x_hat, x)

        total = (
            self.w_rec  * L_recon
          + self.w_sm   * L_smooth
          + self.w_pwr  * L_power
          + self.w_osnr * L_osnr
        )

        components = {
            "loss_total" : total.item(),
            "loss_recon" : L_recon.item(),
            "loss_smooth": L_smooth.item(),
            "loss_power" : L_power.item(),
            "loss_osnr"  : L_osnr.item(),
        }
        return total, components
