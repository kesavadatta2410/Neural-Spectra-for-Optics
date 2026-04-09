"""
SpecCompress-India: Training Entry Point
Usage:
    python training/train.py --config configs/config.yaml
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from utils.helpers import set_seed, get_device, load_config, save_config, count_parameters
from utils.logger import Logger
from utils.dataloader import build_dataloaders
from models.speccompress import SpecCompress
from training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="SpecCompress-India Training")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["data"]["batch_size"] = args.batch_size

    set_seed(cfg["experiment"]["seed"])
    device = get_device(cfg["experiment"]["device"])

    exp_name = cfg["experiment"]["name"]
    ckpt_dir = cfg["training"]["checkpoint_dir"]
    log_dir  = f"experiments/logs"

    logger = Logger(log_dir, exp_name)
    logger.info(f"Experiment: {exp_name}")
    logger.info(f"Device: {device}")

    # ── Data ──────────────────────────────────────
    logger.info("Building dataloaders...")
    train_loader, val_loader, test_loader, stats = build_dataloaders(cfg)
    logger.info(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")

    # ── Model ─────────────────────────────────────
    model = SpecCompress(cfg)
    n_params = count_parameters(model)
    logger.info(f"Model parameters: {n_params:,}")

    # Save resolved config alongside experiment
    os.makedirs(ckpt_dir, exist_ok=True)
    save_config(cfg, f"{ckpt_dir}/{exp_name}_config.yaml")

    # ── Train ─────────────────────────────────────
    trainer = Trainer(model, cfg, device, logger, ckpt_dir)
    trainer.fit(train_loader, val_loader)

    # ── Final test eval ───────────────────────────
    logger.info("Running final test evaluation...")
    from utils.helpers import load_checkpoint
    ckpt = load_checkpoint(f"{ckpt_dir}/best.pt", model)
    test_metrics, x_hat, x_true = trainer.eval_epoch(test_loader)
    logger.log(ckpt["epoch"], "test_final", test_metrics)
    logger.info(f"Test RMSE: {test_metrics['rmse']:.4f}  |  loss: {test_metrics['loss_total']:.4f}")

    # Save reconstructions for eval script
    import numpy as np
    os.makedirs("experiments", exist_ok=True)
    np.save("experiments/test_predictions.npy", x_hat)
    np.save("experiments/test_ground_truth.npy", x_true)
    logger.info("Saved predictions → experiments/")


if __name__ == "__main__":
    main()
