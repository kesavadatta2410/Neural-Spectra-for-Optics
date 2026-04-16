"""
SpecCompress-India: Full Evaluation Suite  (v2)
Covers all 6 gaps:
  Gap 1: Compression ratio sweep
  Gap 2: LLM benchmark
  Gap 3: Duke zero-shot + few-shot
  Gap 4: SOTA baselines (DeepCompress, VQ-VAE, RD-AE, PCA)
  Gap 5: Component ablation
  Gap 6: Temperature robustness
Usage:
    python evaluation/eval.py --config configs/config.yaml [--ckpt path] [--api_key KEY]
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json, argparse, warnings
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

from utils.helpers import set_seed, get_device, load_config, load_checkpoint
from utils.dataloader import build_dataloaders, SpectraDataset
from models.speccompress import SpecCompress
from models.baselines import DeepCompress, VQVAE, RateDistortionAE
from evaluation.metrics import compute_all_metrics, rmse_db, snr_db
from evaluation.compression_ablation import run_compression_ablation
from evaluation.ablation import run_ablation
from evaluation.temp_robustness import run_temp_robustness
from utils.llm_integration import run_llm_benchmark, print_llm_results
from data.duke_loader import load_duke_dataset, make_synthetic_duke_proxy, run_zero_shot_eval, run_few_shot_finetune


def train_baseline(model, train_loader, device, epochs=15, lr=1e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.to(device).train()
    for _ in range(epochs):
        for b in train_loader:
            sp = b["spectrum"].to(device)
            tp = b["temperature"].to(device)
            _, _, loss, _ = model(sp, tp)
            opt.zero_grad(); loss.backward(); opt.step()
    return model


@torch.no_grad()
def collect(model, loader, device, use_temp=True):
    model.eval()
    hat, tru = [], []
    for b in loader:
        sp = b["spectrum"].to(device)
        tp = b["temperature"].to(device)
        out = model(sp, tp) if use_temp else model(sp)
        xh  = out[0]
        hat.append(xh.cpu().numpy())
        tru.append(sp.cpu().numpy())
    return np.concatenate(hat), np.concatenate(tru)


def section(title):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="configs/config.yaml")
    parser.add_argument("--ckpt",    default=None)
    parser.add_argument("--api_key", default=None,
                        help="Anthropic API key for Gap 2 LLM benchmark")
    parser.add_argument("--skip_gaps", nargs="*", default=[],
                        help="Skip specific gaps, e.g. --skip_gaps 1 5")
    args = parser.parse_args()

    api_key  = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    skip     = set(args.skip_gaps)

    cfg = load_config(args.config)
    cfg["data"]["num_workers"] = 0
    cfg["data"]["real_data"]["enabled"] = False
    set_seed(cfg["experiment"]["seed"])
    device   = get_device(cfg["experiment"]["device"])
    ckpt_dir = cfg["training"]["checkpoint_dir"]
    ckpt_path = args.ckpt or f"{ckpt_dir}/best.pt"

    train_loader, val_loader, test_loader, stats = build_dataloaders(cfg)
    n_pts   = cfg["model"]["input_dim"]
    lat_dim = cfg["model"]["latent_dim"]

    all_results = {}

    # ── GAP 4: SOTA Baseline Comparison ─────────────────────────
    if "4" not in skip:
        section("GAP 4: SOTA Baseline Comparison")

        # Load SpecCompress
        sc = SpecCompress(cfg).to(device)
        ckpt_ok = False
        if os.path.exists(ckpt_path):
            load_checkpoint(ckpt_path, sc)
            ckpt_ok = True
        else:
            warnings.warn(f"Checkpoint not found at {ckpt_path}. Training quick SC model.")
            sc = train_baseline(sc, train_loader, device, epochs=15)
        xh_sc, xt = collect(sc, test_loader, device)
        sc_m = compute_all_metrics(xh_sc, xt, "SpecCompress")

        # PCA
        X_tr = np.concatenate([b["spectrum"].numpy() for b in train_loader])
        pca  = PCA(n_components=lat_dim).fit(X_tr)
        X_te = np.concatenate([b["spectrum"].numpy() for b in test_loader])
        xh_pca = pca.inverse_transform(pca.transform(X_te))
        pca_m  = compute_all_metrics(xh_pca, X_te, "PCA")

        # DeepCompress
        dc = DeepCompress(n_pts, lat_dim)
        dc = train_baseline(dc, train_loader, device, epochs=15)
        xh_dc, _ = collect(dc, test_loader, device)
        dc_m = compute_all_metrics(xh_dc, xt, "DeepCompress")

        # VQ-VAE
        vq = VQVAE(n_pts, lat_dim)
        vq = train_baseline(vq, train_loader, device, epochs=15)
        xh_vq, _ = collect(vq, test_loader, device)
        vq_m = compute_all_metrics(xh_vq, xt, "VQ-VAE")

        # Rate-Distortion AE
        rd = RateDistortionAE(n_pts, lat_dim)
        rd = train_baseline(rd, train_loader, device, epochs=15)
        xh_rd, _ = collect(rd, test_loader, device)
        rd_m = compute_all_metrics(xh_rd, xt, "RD-AE")

        baselines = {
            "SpecCompress": sc_m, "PCA": pca_m,
            "DeepCompress": dc_m, "VQ-VAE": vq_m, "RD-AE": rd_m,
        }
        keys = ["rmse_mean", "snr_db", "power_err", "smoothness"]
        hdr  = f"{'Method':<16}" + "".join(f"{k:<14}" for k in keys)
        print(hdr); print("-"*74)
        for name, m in baselines.items():
            row = f"{name:<16}"
            for k in keys:
                full_k = f"{name}/{k}"
                row += f"{m.get(full_k, float('nan')):<14.5f}"
            print(row)

        all_results["gap4_baselines"] = {
            n: {k.split("/")[1]: v for k,v in m.items()} for n,m in baselines.items()
        }

    # ── GAP 1: Compression Ratio Ablation ───────────────────────
    if "1" not in skip:
        section("GAP 1: Compression Ratio Ablation")
        cr_results = run_compression_ablation(cfg, device, train_loader, test_loader)
        all_results["gap1_compression"] = cr_results

    # ── GAP 5: Component Ablation ────────────────────────────────
    if "5" not in skip:
        section("GAP 5: Component Ablation")
        abl_results = run_ablation(cfg, device, train_loader, test_loader)
        all_results["gap5_ablation"] = abl_results

    # ── GAP 6: Temperature Robustness ────────────────────────────
    if "6" not in skip:
        section("GAP 6: Temperature Robustness")
        if os.path.exists(ckpt_path):
            tr_results = run_temp_robustness(cfg, device, ckpt_path, stats)
            all_results["gap6_temp"] = tr_results
        else:
            print("  Skipping: checkpoint required. Run training first.")

    # ── GAP 3: Duke Real Data ────────────────────────────────────
    if "3" not in skip:
        section("GAP 3: Real Data — Duke EDFA (Zero-shot + Few-shot)")
        duke = load_duke_dataset(cfg["data"].get("duke_path", "data/real/duke/duke_edfa.h5"),
                                  n_points=n_pts)
        if duke is None:
            print("  Real Duke data unavailable → using synthetic proxy")
            duke = make_synthetic_duke_proxy(n_samples=1000, n_points=n_pts)

        sc_eval = SpecCompress(cfg).to(device)
        if os.path.exists(ckpt_path):
            load_checkpoint(ckpt_path, sc_eval)
        else:
            sc_eval = train_baseline(sc_eval, train_loader, device, epochs=10)

        zs = run_zero_shot_eval(sc_eval, duke, stats, device)
        fs = run_few_shot_finetune(sc_eval, duke, stats, device, n_finetune=100)

        print(f"\n  Zero-shot RMSE  : {zs['rmse_zero_shot']:.5f}  (no Duke fine-tuning)")
        print(f"  Few-shot RMSE   : {fs['rmse_few_shot']:.5f}  (100-sample fine-tune)")
        print(f"  Domain source   : {duke['source']}")
        print(f"  Gap analysis    : fine-tuning on 100 real samples improves RMSE by "
              f"{(zs['rmse_zero_shot']-fs['rmse_few_shot'])/zs['rmse_zero_shot']*100:.1f}%")
        all_results["gap3_duke"] = {**zs, **fs}

    # ── GAP 2: LLM Integration ───────────────────────────────────
    if "2" not in skip:
        section("GAP 2: LLM Task Benchmark")
        sc_llm = SpecCompress(cfg).to(device)
        if os.path.exists(ckpt_path):
            load_checkpoint(ckpt_path, sc_llm)
        else:
            sc_llm = train_baseline(sc_llm, train_loader, device, epochs=10)

        # Gather test data
        sp_all  = np.concatenate([b["spectrum"].numpy()    for b in test_loader])[:40]
        tp_all  = np.concatenate([b["temperature"].numpy() for b in test_loader])[:40]
        wl_arr  = np.linspace(1530., 1565., n_pts)

        llm_res = run_llm_benchmark(
            model_sc=sc_llm,
            spectra_norm=sp_all, temperatures=tp_all,
            wavelengths=wl_arr, stats=stats,
            pca=pca if "4" not in skip else None,
            api_key=api_key, n_samples=20, cfg=cfg,
        )
        print_llm_results(llm_res)
        all_results["gap2_llm"] = llm_res

    # ── Save all results ─────────────────────────────────────────
    os.makedirs("experiments", exist_ok=True)
    out = "experiments/full_eval_results.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n{'='*65}")
    print(f"  All results saved → {out}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
