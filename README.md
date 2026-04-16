# SpecCompress-India

> **Physics-Informed Neural Spectral Compression for Climate-Aware Optical Networks**

[![CI](https://github.com/kesavadatta2410/Neural-Spectra-for-Optics/actions/workflows/ci.yml/badge.svg)](https://github.com/kesavadatta2410/Neural-Spectra-for-Optics/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C.svg)](https://pytorch.org/)

---

Compresses 1 000-point EDFA gain spectra → 10-dimensional latent vectors (**100× compression**) while enforcing physical consistency via four complementary loss terms. Temperature-conditioned for India's 25 °C – 45 °C tropical ambient range.

---

## Architecture

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
               │   ENCODER             │
               │  1D CNN × 3 blocks   │  ← shape-agnostic spectral features
               │  GlobalPool + Flat   │
               │  TempEmbed (16-dim)  │  ← temperature injected after CNN
               │  MLP [512→256→10]    │
               └──────────┬───────────┘
                          │  z ∈ ℝ¹⁰   (100× compression)
               ┌──────────┴───────────┐
               │   DECODER             │
               │  TempEmbed (16-dim)  │  ← temperature re-injected
               │  concat(z, t_emb)    │
               │  MLP [256→512→1000]  │
               └──────────┬───────────┘
                          │
              [B, 1000] reconstructed spectrum
                          │
               ┌──────────┴───────────┐
               │  PHYSICS LOSS         │
               │  L_recon  ×1.00      │  spectral fidelity (MSE)
               │  L_smooth ×0.10      │  no high-freq artefacts (TV)
               │  L_power  ×0.05      │  integral power conservation
               │  L_osnr   ×0.02      │  low-SNR region up-weighting
               └──────────────────────┘
```

Temperature is injected at **two** points: after CNN extraction (controls compression), and at decoder entry (controls reconstruction). This dual-injection lets the model learn temperature-invariant spectral structure while recovering temperature-dependent gain shifts at decode time.

---

## Installation

```bash
# Clone
git clone https://github.com/kesavadatta2410/Neural-Spectra-for-Optics.git
cd Neural-Spectra-for-Optics

# Install (editable, with dev extras)
pip install -e ".[dev]"

# Verify GPU (optional)
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Quick Start

```bash
# 1. Generate 50k train / 5k val / 5k test spectra
python main.py generate

# 2. Train (100 epochs, cosine LR, early stopping)
python main.py train

# 3. Full evaluation (all 6 research gaps)
python main.py eval
```

---

## CLI Reference

| Command | Description |
|---|---|
| `python main.py generate` | Synthesise HDF5 datasets |
| `python main.py train [--epochs N] [--batch_size B]` | Train SpecCompress |
| `python main.py eval [--ckpt PATH] [--api_key KEY]` | Full eval suite |
| `python main.py ablation` | Component ablation (Gap 5) |
| `python main.py compression` | Compression ratio sweep (Gap 1) |
| `python main.py temp` | Temperature robustness (Gap 6) |

---

## Project Structure

```
Neural-Spectra-for-Optics/
├── data/
│   ├── data_generation.py      ← synthetic EDFA generator
│   ├── real_data_loader.py     ← COSMOS / GNPy real data
│   └── duke_loader.py          ← Duke EDFA dataset (202k samples)
├── models/
│   ├── speccompress.py         ← full autoencoder (factory + ablation)
│   ├── encoder.py              ← 1D CNN + temperature MLP encoder
│   ├── decoder.py              ← temperature-conditioned MLP decoder
│   ├── physics_loss.py         ← physics-informed composite loss
│   └── baselines.py            ← PCA, DeepCompress, VQ-VAE, RD-AE
├── training/
│   ├── train.py                ← CLI entry-point
│   └── trainer.py              ← training loop, LR scheduling, checkpointing
├── evaluation/
│   ├── eval.py                 ← full evaluation suite (Gaps 1–6)
│   ├── metrics.py              ← RMSE, SNR, power error, smoothness
│   ├── ablation.py             ← component ablation (Gap 5)
│   ├── compression_ablation.py ← latent-dim sweep (Gap 1)
│   └── temp_robustness.py      ← OOD temperature analysis (Gap 6)
├── utils/
│   ├── dataloader.py           ← HDF5 DataLoader + real-data merge
│   ├── logger.py               ← CSV + console logger
│   ├── helpers.py              ← seed, device, checkpoint utilities
│   └── llm_integration.py      ← LLM benchmark (Gap 2)
├── configs/
│   └── config.yaml             ← all hyperparameters (no hardcoding)
├── tests/                      ← CPU smoke-tests (no data required)
├── .github/workflows/ci.yml    ← GitHub Actions CI
├── pyproject.toml              ← PEP 517/518 build config
├── CITATION.cff                ← machine-readable citation
├── CHANGELOG.md
├── CONTRIBUTING.md
└── README.md
```

---

## Reproducibility

- Seeds fixed globally via `utils.helpers.set_seed(42)`
- `torch.backends.cudnn.deterministic = True`
- Config YAML saved alongside every checkpoint
- HDF5 seed stored as dataset attribute

---

## Expected Results (synthetic test set, 50k train, 100 epochs)

| Method | RMSE ↓ | SNR (dB) ↑ | Power Err ↓ |
|---|---|---|---|
| **SpecCompress** | ~0.04 | ~28 | ~0.01 |
| PCA (10-d) | ~0.14 | ~17 | ~0.04 |
| DeepCompress | ~0.08 | ~22 | ~0.02 |
| VQ-VAE | ~0.07 | ~23 | ~0.02 |
| RD-AE | ~0.06 | ~25 | ~0.02 |

*(Normalised spectrum units; exact values depend on hardware and training duration.)*

---

## Real Data (Optional)

| Source | Use | Auto-download |
|---|---|---|
| COSMOS EDFA Dataset | Val augmentation | ✓ |
| GNPy amp tables | Val augmentation | ✓ |
| Duke EDFA (202k) | Zero/few-shot eval (Gap 3) | ✓ |

Set `data.real_data.enabled: false` in `config.yaml` to disable.

---

## Running Tests

```bash
pytest tests/ -v    # CPU-only, no data required, ~10 s
```

---

## Citing This Work

If you use SpecCompress-India in your research, please cite it using the metadata in [`CITATION.cff`](CITATION.cff):

```bibtex
@software{speccompress_india_2026,
  author  = {Malli, Kesavadatta},
  title   = {{SpecCompress-India}: Physics-Informed Neural Spectral Compression
             for Climate-Aware Optical Networks},
  year    = {2026},
  version = {2.0.0},
  url     = {https://github.com/kesavadatta2410/Neural-Spectra-for-Optics},
  license = {MIT}
}
```

---

## License

[MIT](LICENSE) © 2026 Kesavadatta Malli
