"""
SpecCompress-India: Physics-Informed Loss  (v2)
Four components, each independently toggleable for ablation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class PhysicsInformedLoss(nn.Module):
    def __init__(
        self,
        reconstruction_weight: float = 1.0,
        smoothness_weight: float = 0.1,
        power_conservation_weight: float = 0.05,
        osnr_penalty_weight: float = 0.02,
        use_smoothness: bool = True,
        use_power: bool = True,
        use_osnr: bool = True,
    ):
        super().__init__()
        self.w_rec   = reconstruction_weight
        self.w_sm    = smoothness_weight    if use_smoothness else 0.0
        self.w_pwr   = power_conservation_weight if use_power else 0.0
        self.w_osnr  = osnr_penalty_weight  if use_osnr    else 0.0

    @staticmethod
    def reconstruction_loss(x_hat, x):
        return F.mse_loss(x_hat, x)

    @staticmethod
    def smoothness_loss(x_hat):
        diff = x_hat[:, 1:] - x_hat[:, :-1]
        return (diff ** 2).mean()

    @staticmethod
    def power_conservation_loss(x_hat, x):
        return F.mse_loss(x_hat.mean(-1), x.mean(-1))

    @staticmethod
    def osnr_penalty_loss(x_hat, x):
        weights = 1.0 / (torch.abs(x) + 0.1)
        weights = weights / weights.mean()
        return (weights * (x_hat - x) ** 2).mean()

    def forward(self, x_hat, x) -> Tuple[torch.Tensor, Dict[str, float]]:
        L_rec   = self.reconstruction_loss(x_hat, x)
        L_sm    = self.smoothness_loss(x_hat)    if self.w_sm   > 0 else torch.tensor(0.)
        L_pwr   = self.power_conservation_loss(x_hat, x) if self.w_pwr  > 0 else torch.tensor(0.)
        L_osnr  = self.osnr_penalty_loss(x_hat, x)      if self.w_osnr > 0 else torch.tensor(0.)

        total = (self.w_rec * L_rec + self.w_sm * L_sm +
                 self.w_pwr * L_pwr + self.w_osnr * L_osnr)

        return total, {
            "loss_total" : total.item(),
            "loss_recon" : L_rec.item(),
            "loss_smooth": L_sm.item()  if isinstance(L_sm,  float) else L_sm.item(),
            "loss_power" : L_pwr.item() if isinstance(L_pwr, float) else L_pwr.item(),
            "loss_osnr"  : L_osnr.item()if isinstance(L_osnr,float) else L_osnr.item(),
        }
