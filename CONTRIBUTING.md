# Contributing to SpecCompress-India

Thank you for considering a contribution! This document covers the workflow for
submitting bug-fixes, new features, and documentation improvements.

---

## Development Setup

```bash
# 1. Fork & clone
git clone https://github.com/<your-username>/Neural-Spectra-for-Optics.git
cd Neural-Spectra-for-Optics

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install in editable mode with dev extras
pip install -e ".[dev]"

# 4. (Optional) Install pre-commit hooks
pip install pre-commit
pre-commit install
```

---

## Running Tests

```bash
pytest tests/ -v
```

All tests run on CPU without requiring pre-generated data — no HDF5 files are needed.

---

## Code Style

We use **ruff** for linting and formatting:

```bash
ruff check .          # lint
ruff format .         # auto-format (line length 100)
```

The CI pipeline will fail on lint errors, so please run this locally before pushing.

---

## Submitting Changes

1. **Open an issue first** for non-trivial changes so we can discuss the approach.
2. **Branch off `main`**: `git checkout -b feat/my-feature`
3. **Write tests** for any new public functions.
4. **Update `CHANGELOG.md`** under the `[Unreleased]` section.
5. **Open a Pull Request** with a clear description of what changed and why.

---

## Repository Layout

```
speccompress-india/
├── data/           ← data generation + real data loaders
├── models/         ← SpecCompress, baselines, loss functions
├── evaluation/     ← evaluation scripts (ablation, temp robustness, …)
├── training/       ← Trainer class + train.py entry-point
├── utils/          ← helpers, dataloader, logger, LLM integration
├── configs/        ← YAML hyperparameter files
├── tests/          ← pytest smoke-tests (CPU, no data required)
└── main.py         ← CLI router
```

---

## Reporting Bugs

Please open a GitHub Issue with:
- OS / Python / CUDA version
- Minimal reproducible example
- Full traceback
