"""
SpecCompress-India: Evaluation + Baseline Comparison
Compares SpecCompress against:
  1. PCA (32 components)
  2. Vanilla Autoencoder (no physics loss, no temp conditioning)

Usage:
    python evaluation/eval.py --config configs/config.yaml
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from utils.helpers import set_seed, get_device, load_config, load_checkpoint
from utils.dataloader import build_dataloaders
from models.speccompress import SpecCompress
from evaluation.metrics import compute_all_metrics


# ──────────────────────────────────────────────
# Vanilla AE Baseline (no physics, no temp)
# ──────────────────────────────────────────────
import torch.nn as nn

class VanillaAutoencoder(nn.Module):
    def __init__(self, n_points: int = 1000, latent_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_points, 512), nn.ReLU(),
            nn.Linear(512, 256),     nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 512),        nn.ReLU(),
            nn.Linear(512, n_points)
        )

    def forward(self, x, temp=None):
        z = self.encoder(x)
        return self.decoder(z), z


def train_vanilla_ae(train_loader, val_loader, device, n_points, latent_dim, epochs=20):
    model = VanillaAutoencoder(n_points, latent_dim).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            x = batch["spectrum"].to(device)
            x_hat, _ = model(x)
            loss = torch.nn.functional.mse_loss(x_hat, x)
            opt.zero_grad(); loss.backward(); opt.step()
    print(f"  [VanillaAE] Trained {epochs} epochs.")
    return model


def eval_model_on_loader(model, loader, device):
    model.eval()
    all_hat, all_true = [], []
    with torch.no_grad():
        for batch in loader:
            x    = batch["spectrum"].to(device)
            temp = batch["temperature"].to(device)
            out  = model(x, temp)
            if isinstance(out, tuple):
                x_hat = out[0]
            else:
                x_hat = out
            all_hat.append(x_hat.cpu().numpy())
            all_true.append(x.cpu().numpy())
    return np.concatenate(all_hat), np.concatenate(all_true)


# ──────────────────────────────────────────────
# Main Eval
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   type=str, default="configs/config.yaml")
    parser.add_argument("--ckpt",     type=str, default=None)
    parser.add_argument("--baselines",action="store_true", default=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["experiment"]["seed"])
    device    = get_device(cfg["experiment"]["device"])
    n_points  = cfg["model"]["input_dim"]
    latent_dim = cfg["model"]["latent_dim"]
    ckpt_path = args.ckpt or f"{cfg['training']['checkpoint_dir']}/best.pt"
    eval_cfg  = cfg["evaluation"]

    print("=" * 60)
    print("  SpecCompress-India | Evaluation Suite")
    print("=" * 60)

    _, val_loader, test_loader, stats = build_dataloaders(cfg)

    # ── SpecCompress ──────────────────────────────
    model = SpecCompress(cfg).to(device)
    load_checkpoint(ckpt_path, model)
    model.eval()

    print("\n[1/3] Evaluating SpecCompress...")
    x_hat_sc, x_true = eval_model_on_loader(model, test_loader, device)
    sc_metrics = compute_all_metrics(x_hat_sc, x_true, label="SpecCompress")
    _print_metrics(sc_metrics)

    results = {"SpecCompress": sc_metrics}

    # ── PCA Baseline ─────────────────────────────
    if args.baselines and "pca" in eval_cfg.get("baselines", []):
        print("\n[2/3] Evaluating PCA baseline...")
        # Gather train data
        train_arr = []
        for batch in val_loader:
            train_arr.append(batch["spectrum"].numpy())
        X_train = np.concatenate(train_arr[:20], axis=0)   # Use subset for speed
        X_test  = x_true

        pca = PCA(n_components=eval_cfg.get("pca_components", latent_dim))
        pca.fit(X_train)
        X_pca_recon = pca.inverse_transform(pca.transform(X_test))
        pca_metrics = compute_all_metrics(X_pca_recon, X_test, label="PCA")
        _print_metrics(pca_metrics)
        results["PCA"] = pca_metrics

    # ── Vanilla AE Baseline ───────────────────────
    if args.baselines and "vanilla_ae" in eval_cfg.get("baselines", []):
        print("\n[3/3] Training + evaluating VanillaAE baseline...")
        _, val_loader2, test_loader2, _ = build_dataloaders(cfg)
        vae_model = train_vanilla_ae(
            val_loader2, test_loader2, device, n_points, latent_dim, epochs=15
        )
        x_hat_vae, x_true_vae = eval_model_on_loader(vae_model, test_loader2, device)
        vae_metrics = compute_all_metrics(x_hat_vae, x_true_vae, label="VanillaAE")
        _print_metrics(vae_metrics)
        results["VanillaAE"] = vae_metrics

    # ── Summary Table ─────────────────────────────
    print("\n" + "=" * 60)
    print("  COMPARISON SUMMARY (test set)")
    print("=" * 60)
    metric_keys = ["rmse_mean", "snr_db", "power_err", "smoothness"]
    header = f"{'Method':<20}" + "".join(f"{k:<18}" for k in metric_keys)
    print(header)
    print("-" * len(header))
    for method, metrics in results.items():
        row = f"{method:<20}"
        for k in metric_keys:
            full_key = f"{method}/{k}"
            val = metrics.get(full_key, float("nan"))
            row += f"{val:<18.5f}"
        print(row)
    print("=" * 60)

    # Save results
    import json
    os.makedirs("experiments", exist_ok=True)
    with open("experiments/eval_results.json", "w") as f:
        # Convert to serialisable
        clean = {m: {k: round(v, 6) for k, v in mv.items()} for m, mv in results.items()}
        json.dump(clean, f, indent=2)
    print("\nResults saved → experiments/eval_results.json")


def _print_metrics(metrics: dict):
    for k, v in metrics.items():
        print(f"  {k}: {v:.5f}")


if __name__ == "__main__":
    main()
