"""
SpecCompress-India: Helpers
Shared utility functions.
"""

import os
import random
import numpy as np
import torch
import yaml
from pathlib import Path
from typing import Dict, Any


def set_seed(seed: int = 42) -> None:
    """Deterministic training: fix all RNG sources."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def get_device(preference: str = "auto") -> torch.device:
    """Resolve device. 'auto' selects CUDA > MPS > CPU."""
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preference)


def load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def save_config(cfg: Dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def denormalize_spectrum(
    spec_norm: np.ndarray,
    spec_mean: float,
    spec_std:  float
) -> np.ndarray:
    return spec_norm * spec_std + spec_mean


def denormalize_temperature(
    temp_norm: np.ndarray,
    temp_mean: float,
    temp_std:  float
) -> np.ndarray:
    return temp_norm * temp_std + temp_mean


def save_checkpoint(
    model:     torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch:     int,
    val_loss:  float,
    path:      str
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch"     : epoch,
        "val_loss"  : val_loss,
        "model"     : model.state_dict(),
        "optimizer" : optimizer.state_dict(),
    }, path)


def load_checkpoint(
    path:      str,
    model:     torch.nn.Module,
    optimizer: torch.optim.Optimizer = None
) -> Dict:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt
