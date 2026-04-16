"""
Gap 5: Component Ablation Study
Trains 6 model variants; records RMSE per variant.
Usage: python evaluation/ablation.py --config configs/config.yaml
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json, argparse
import numpy as np
import torch
from copy import deepcopy

from utils.helpers import set_seed, get_device, load_config
from utils.dataloader import build_dataloaders
from models.speccompress import SpecCompress
from evaluation.metrics import rmse_db, snr_db, smoothness_score, power_conservation_error


def train_quick(model, loader, device, epochs=12):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.to(device).train()
    for _ in range(epochs):
        for batch in loader:
            sp = batch["spectrum"].to(device)
            tp = batch["temperature"].to(device)
            _, _, loss, _ = model(sp, tp)
            opt.zero_grad(); loss.backward(); opt.step()
    return model


@torch.no_grad()
def collect_preds(model, loader, device):
    model.eval()
    hat, tru = [], []
    for b in loader:
        sp = b["spectrum"].to(device)
        tp = b["temperature"].to(device)
        xh, _, _, _ = model(sp, tp)
        hat.append(xh.cpu().numpy())
        tru.append(sp.cpu().numpy())
    return np.concatenate(hat), np.concatenate(tru)


def run_ablation(cfg, device, train_loader, test_loader):
    variants = cfg["ablation"]["variants"]

    print("\n" + "="*72)
    print("  GAP 5: Component Ablation Study")
    print("="*72)
    hdr = f"{'Variant':<22} {'RMSE':>8} {'SNR(dB)':>9} {'PwrErr':>9} {'Smooth':>10}"
    print(hdr); print("-"*72)

    results = []
    for v in variants:
        set_seed(cfg["experiment"]["seed"])
        model = SpecCompress(cfg, ablation_variant=v)
        model = train_quick(model, train_loader, device, epochs=12)
        xh, xt = collect_preds(model, test_loader, device)

        row = {
            "variant"    : v["name"],
            "rmse"       : rmse_db(xh, xt),
            "snr_db"     : snr_db(xh, xt),
            "power_err"  : power_conservation_error(xh, xt),
            "smoothness" : smoothness_score(xh),
        }
        results.append(row)
        print(f"{v['name']:<22} {row['rmse']:>8.5f} {row['snr_db']:>9.3f} "
              f"{row['power_err']:>9.5f} {row['smoothness']:>10.6f}")

    # Delta from full model
    full = next(r for r in results if r["variant"] == "full_model")
    print("\n  Delta RMSE vs full_model (positive = worse):")
    for r in results:
        if r["variant"] != "full_model":
            delta = r["rmse"] - full["rmse"]
            sign  = "+" if delta >= 0 else ""
            print(f"    {r['variant']:<22} Δ RMSE = {sign}{delta:.5f}")

    os.makedirs("experiments", exist_ok=True)
    with open("experiments/ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved → experiments/ablation_results.json")
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
    run_ablation(cfg, device, train_loader, test_loader)
