"""
SpecCompress-India: Trainer
Encapsulates full training loop with:
  - LR scheduling (cosine / step)
  - Gradient clipping
  - Early stopping
  - Checkpoint saving
  - Metric logging
"""

import math
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple

from utils.logger import Logger
from utils.helpers import save_checkpoint


class Trainer:
    def __init__(
        self,
        model:       nn.Module,
        cfg:         dict,
        device:      torch.device,
        logger:      Logger,
        checkpoint_dir: str
    ):
        self.model    = model.to(device)
        self.cfg      = cfg
        self.device   = device
        self.logger   = logger
        self.ckpt_dir = checkpoint_dir
        self.t_cfg    = cfg["training"]

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.t_cfg["lr"],
            weight_decay=self.t_cfg["weight_decay"]
        )
        self.best_val_loss = float("inf")
        self.patience_ctr  = 0
        self.global_step   = 0

    def _build_scheduler(self, n_steps_per_epoch: int):
        total_steps   = self.t_cfg["epochs"] * n_steps_per_epoch
        warmup_steps  = self.t_cfg["warmup_epochs"] * n_steps_per_epoch
        sched_type    = self.t_cfg["lr_scheduler"]

        if sched_type == "cosine":
            def lr_lambda(step):
                if step < warmup_steps:
                    return step / max(warmup_steps, 1)
                progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
                return 0.5 * (1.0 + math.cos(math.pi * progress))
            return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        elif sched_type == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=20 * n_steps_per_epoch, gamma=0.5
            )
        return None

    def train_epoch(self, loader: DataLoader, scheduler=None) -> Dict[str, float]:
        self.model.train()
        totals: Dict[str, float] = {}
        n_batches = 0

        for batch in loader:
            spec = batch["spectrum"].to(self.device)
            temp = batch["temperature"].to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            _, _, loss, components = self.model(spec, temp)
            loss.backward()

            if self.t_cfg["grad_clip"] > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.t_cfg["grad_clip"])

            self.optimizer.step()
            if scheduler:
                scheduler.step()

            for k, v in components.items():
                totals[k] = totals.get(k, 0.0) + v
            n_batches += 1
            self.global_step += 1

            # Periodic batch logging
            if self.global_step % self.t_cfg["log_interval"] == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                self.logger.log(self.global_step, "train_batch", {
                    **{k: v / n_batches for k, v in totals.items()},
                    "lr": lr
                })

        return {k: v / n_batches for k, v in totals.items()}

    @torch.no_grad()
    def eval_epoch(self, loader: DataLoader) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
        self.model.eval()
        totals: Dict[str, float] = {}
        n_batches = 0
        all_hat, all_true = [], []

        for batch in loader:
            spec = batch["spectrum"].to(self.device)
            temp = batch["temperature"].to(self.device)
            x_hat, _, loss, components = self.model(spec, temp)

            for k, v in components.items():
                totals[k] = totals.get(k, 0.0) + v
            n_batches += 1

            all_hat.append(x_hat.cpu().numpy())
            all_true.append(spec.cpu().numpy())

        metrics  = {k: v / n_batches for k, v in totals.items()}
        x_hat_np = np.concatenate(all_hat, axis=0)
        x_np     = np.concatenate(all_true, axis=0)

        # RMSE in normalised space
        rmse = float(np.sqrt(np.mean((x_hat_np - x_np) ** 2)))
        metrics["rmse"] = rmse
        return metrics, x_hat_np, x_np

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        n_steps = len(train_loader)
        scheduler = self._build_scheduler(n_steps)
        patience  = self.t_cfg["early_stopping_patience"]

        self.logger.info(f"Training for {self.t_cfg['epochs']} epochs")
        self.logger.info(f"Scheduler: {self.t_cfg['lr_scheduler']}")

        for epoch in range(1, self.t_cfg["epochs"] + 1):
            t0 = time.time()
            train_metrics = self.train_epoch(train_loader, scheduler)
            val_metrics, _, _ = self.eval_epoch(val_loader)
            elapsed = time.time() - t0

            self.logger.log(epoch, "train_epoch", train_metrics)
            self.logger.log(epoch, "val_epoch",   val_metrics)
            self.logger.info(
                f"Epoch {epoch:3d}/{self.t_cfg['epochs']} | "
                f"val_loss={val_metrics['loss_total']:.4f} | "
                f"val_rmse={val_metrics['rmse']:.4f} | "
                f"{elapsed:.1f}s"
            )

            # Checkpoint + early stopping
            val_loss = val_metrics["loss_total"]
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_ctr  = 0
                save_checkpoint(
                    self.model, self.optimizer, epoch, val_loss,
                    path=f"{self.ckpt_dir}/best.pt"
                )
                self.logger.info(f"  ✓ New best val_loss={val_loss:.4f}  (saved checkpoint)")
            else:
                self.patience_ctr += 1
                if self.patience_ctr >= patience:
                    self.logger.info(f"  Early stopping after {patience} epochs without improvement.")
                    break
