"""
Gap 1: Compression Ratio Ablation
Sweeps latent_dim in [32, 16, 10, 8] → plots RMSE vs compression ratio.
Trains a lightweight model per dim and records metrics.
Usage: python evaluation/compression_ablation.py --config configs/config.yaml
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy

from utils.helpers import set_seed, get_device, load_config
from utils.dataloader import build_dataloaders
from models.speccompress import SpecCompress
from evaluation.metrics import rmse_db


def train_quick(model, train_loader, device, epochs=10):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    model.to(device).train()
    for ep in range(epochs):
        for batch in train_loader:
            spec = batch["spectrum"].to(device)
            temp = batch["temperature"].to(device)
            _, _, loss, _ = model(spec, temp)
            opt.zero_grad(); loss.backward(); opt.step()
    return model


@torch.no_grad()
def evaluate(model, test_loader, device):
    model.eval()
    all_hat, all_true = [], []
    for batch in test_loader:
        spec = batch["spectrum"].to(device)
        temp = batch["temperature"].to(device)
        x_hat, _, _, _ = model(spec, temp)
        all_hat.append(x_hat.cpu().numpy())
        all_true.append(spec.cpu().numpy())
    x_hat = np.concatenate(all_hat)
    x_true = np.concatenate(all_true)
    return rmse_db(x_hat, x_true)


def run_compression_ablation(cfg, device, train_loader, test_loader):
    dims   = cfg.get("compression_ablation", {}).get("latent_dims", [32, 16, 10, 8])
    n_pts  = cfg["model"]["input_dim"]
    results = []

    print("\n" + "="*55)
    print("  GAP 1: Compression Ratio Ablation")
    print("="*55)
    print(f"{'Latent Dim':<12} {'Compression':<14} {'RMSE':>8} {'Params':>12}")
    print("-"*55)

    for dim in dims:
        set_seed(cfg["experiment"]["seed"])
        cfg_copy = deepcopy(cfg)
        cfg_copy["model"]["latent_dim"] = dim

        model = SpecCompress(cfg_copy)
        n_params = model.count_parameters()
        cr = n_pts / dim

        model = train_quick(model, train_loader, device, epochs=15)
        rmse  = evaluate(model, test_loader, device)

        print(f"{dim:<12} {cr:<14.1f}× {rmse:>8.5f} {n_params:>12,}")
        results.append({
            "latent_dim"        : dim,
            "compression_ratio" : cr,
            "rmse"              : rmse,
            "n_params"          : n_params,
        })

    os.makedirs("experiments", exist_ok=True)
    with open("experiments/compression_ablation.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved → experiments/compression_ablation.json")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["data"]["num_workers"] = 0
    cfg["data"]["real_data"]["enabled"] = False
    set_seed(cfg["experiment"]["seed"])
    device = get_device(cfg["experiment"]["device"])
    train_loader, _, test_loader, _ = build_dataloaders(cfg)
    run_compression_ablation(cfg, device, train_loader, test_loader)
