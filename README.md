# SpecCompress-India
## Physics-Informed Neural Spectral Compression for Climate-Aware Optical Networks

> Compresses 1000-dim EDFA gain spectra → 32-dim latent vectors.  
> Temperature-conditioned (25°C–45°C India ambient range).  
> Physics priors: smoothness + power conservation + OSNR penalty.

---

## Architecture Design

```
                    SYNTHETIC DATA GENERATOR
                    ─────────────────────────
  EDFA Physics ──► Gain Curves (Gaussian lobes)
  Temperature  ──► Gain Shift + Spectral Tilt         [25°C–45°C]
  OSNR + Noise ──► Gaussian + Shot noise floor
  Perturbations──► Pump drift, splice loss, notches
                          │
                          ▼
              [B, 1000] spectrum + [B] temperature
                          │
               ┌──────────┴───────────┐
               │         ENCODER       │
               │  1D CNN × 3 blocks   │  ← Learns shape-agnostic features
               │  (stride=2 each)     │    from raw spectrum
               │  GlobalPool + Flat   │
               │         +            │
               │  TempEmbed(16-dim)   │  ← Temperature injected HERE
               │  (normalised °C)     │    AFTER CNN, BEFORE MLP
               │  MLP [512→256→32]    │
               └──────────┬───────────┘
                          │
                    z [B, 32]  ← Compressed latent
                          │
               ┌──────────┴───────────┐
               │         DECODER       │
               │  TempEmbed(16-dim)   │  ← Temperature re-injected
               │  concat(z, temp_emb) │    allows temp-conditional decode
               │  MLP [256→512→1024   │
               │      →1000]          │
               └──────────┬───────────┘
                          │
              [B, 1000] reconstructed spectrum
                          │
               ┌──────────┴───────────┐
               │    PHYSICS LOSS       │
               │  L_recon  (MSE)  ×1.0│  Spectral fidelity
               │  L_smooth (TV)   ×0.1│  No discontinuities
               │  L_power         ×0.05│ Integral conservation
               │  L_osnr          ×0.02│ Low-SNR region penalty
               └──────────────────────┘
```

**Temperature injection points:**
1. **Encoder**: after CNN extraction, before MLP → conditions the compression
2. **Decoder**: at entry → conditions the reconstruction

This dual-injection allows the model to learn temperature-invariant spectral structure in latent space while still recovering temperature-dependent features at decode time.

**Physics constraints influence learning:**
- `L_smooth` prevents the decoder from producing non-physical high-freq artefacts
- `L_power` ensures total optical power is conserved through encode-decode
- `L_osnr` up-weights errors in low-power (high-noise) spectral regions, critical for OSNR budget

---

## Installation

```bash
# 1. Clone / unzip project
cd SpecCompress-India

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify GPU (optional)
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Dataset Generation

```bash
# Generate 50k train + 5k val + 5k test (default)
python main.py generate

# Custom sizes
python main.py generate --n_train 100000 --n_val 10000 --n_test 10000

# Custom resolution
python main.py generate --n_points 1000 --out_dir data/synthetic
```

Output: `data/synthetic/{train,val,test}.h5`

---

## Training

```bash
# Default config
python main.py train

# Override epochs and batch size
python main.py train --epochs 50 --batch_size 512

# Custom config
python main.py train --config configs/config.yaml
```

Checkpoints → `experiments/checkpoints/best.pt`  
Logs → `experiments/logs/`

---

## Evaluation

```bash
# Full eval + baseline comparison
python main.py eval

# Custom checkpoint
python main.py eval --ckpt experiments/checkpoints/best.pt
```

Outputs:
- Console comparison table: SpecCompress vs PCA vs VanillaAE
- `experiments/eval_results.json`

---

## Project Structure

```
SpecCompress-India/
├── data/
│   ├── synthetic/              ← HDF5 datasets (generated)
│   ├── real/                   ← Real data cache (optional)
│   ├── data_generation.py      ← PRIMARY synthetic generator
│   └── real_data_loader.py     ← SECONDARY real data (COSMOS, GNPy)
├── configs/
│   └── config.yaml             ← All hyperparameters
├── models/
│   ├── speccompress.py         ← Full model
│   ├── encoder.py              ← 1D CNN + MLP encoder
│   ├── decoder.py              ← MLP decoder
│   └── physics_loss.py         ← Physics-informed loss
├── training/
│   ├── train.py                ← Training entry point
│   └── trainer.py              ← Training loop + scheduling
├── evaluation/
│   ├── eval.py                 ← Eval + baseline comparison
│   └── metrics.py              ← RMSE, SNR, power error
├── utils/
│   ├── dataloader.py           ← HDF5 DataLoader
│   ├── logger.py               ← CSV + console logger
│   └── helpers.py              ← Seed, device, checkpoint utils
├── experiments/                ← Auto-created during runs
├── main.py                     ← CLI router
├── requirements.txt
└── README.md
```

---

## Reproducibility

- All seeds fixed via `utils/helpers.set_seed(42)`
- `torch.backends.cudnn.deterministic = True`
- Config saved alongside every checkpoint
- HDF5 seed stored as dataset attribute

---

## Real Data Sources (Optional)

| Source | Used For | Auto-download |
|--------|----------|--------------|
| COSMOS EDFA Dataset | Val augmentation | ✓ |
| GNPy amp tables | Val augmentation | ✓ |

Set `data.real_data.enabled: false` in config to disable.

---

## Expected Results (synthetic test set, 50k train)

| Method       | RMSE ↓   | SNR (dB) ↑ | Power Err ↓ |
|-------------|----------|------------|-------------|
| SpecCompress | ~0.04    | ~28        | ~0.01       |
| PCA (32-d)   | ~0.12    | ~18        | ~0.03       |
| VanillaAE    | ~0.07    | ~23        | ~0.02       |

*(Values in normalised spectrum units; exact numbers depend on hardware + epochs)*
