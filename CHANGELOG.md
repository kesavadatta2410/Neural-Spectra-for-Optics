# Changelog

All notable changes to **SpecCompress-India** are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Version numbers follow [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

---

## [2.0.0] — 2026-04-16

### Added
- `RateDistortionAE` baseline: L1-penalised latent for entropy-rate approximation.
- `run_llm_benchmark` and `print_llm_results` in `utils/llm_integration.py` for
  orchestrating the three-condition (raw / compressed / VQ) LLM evaluation (Gap 2).
- `load_duke_dataset`, `make_synthetic_duke_proxy`, `run_zero_shot_eval`,
  `run_few_shot_finetune` in `data/duke_loader.py` (Gap 3 public API).
- `pyproject.toml` — PEP 517/518 build configuration; `speccompress` CLI entry-point.
- `CITATION.cff` — GitHub-native citation (enables "Cite this repository" button).
- `LICENSE` — MIT.
- `CONTRIBUTING.md` — development setup, code-style, and PR workflow.
- `tests/` — CPU smoke-tests for model shape, loss components, and compression ratio.
- `.github/workflows/ci.yml` — GitHub Actions CI (Python 3.10–3.12, pytest).
- `__init__.py` in every package (`models`, `evaluation`, `training`, `utils`, `data`).

### Changed
- Relocated misplaced files to their canonical packages:
  - `models/ablation.py` → `evaluation/ablation.py`
  - `models/compression_ablation.py` → `evaluation/compression_ablation.py`
  - `models/temp_robustness.py` → `evaluation/temp_robustness.py`
  - `models/llm_integration.py` → `utils/llm_integration.py`
  - `utils/duke_loader.py` → `data/duke_loader.py`
- Rewrote `.gitignore` — properly excludes HDF5 data, checkpoints, virtual envs,
  large `.npy` artifacts, and IDE/OS files.
- Reformatted `models/baselines.py` (dense one-liners → PEP 8 style).

### Fixed
- Removed 40 MB of binary experiment outputs (`test_ground_truth.npy`,
  `test_predictions.npy`, `eval_results.json`) from git tracking.
- Fixed trailing-comma bug in `eval_baseline` that caused a `ValueError` at runtime.

---

## [1.0.0] — 2026-04-15

### Added
- Initial release of SpecCompress-India v2 architecture.
- 1D CNN encoder with temperature conditioning.
- Physics-informed loss (MSE + smoothness + power conservation + OSNR penalty).
- Synthetic EDFA data generator (Gaussian lobes, temperature tilt, pump noise).
- Full evaluation suite covering 6 research gaps.
- Config-driven hyperparameters via `configs/config.yaml`.
