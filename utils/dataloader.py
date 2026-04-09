"""
SpecCompress-India: DataLoader
Wraps HDF5 synthetic + optional real data.
"""

import os
import warnings
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Dict, Tuple
import h5py


class SpectraDataset(Dataset):
    """
    HDF5-backed spectrum dataset.
    Supports Z-score normalisation (fit on train, apply everywhere).
    """

    def __init__(
        self,
        hdf5_path:     str,
        normalize:     bool = True,
        stats:         Optional[Dict] = None,  # pre-computed mean/std
        temp_normalize: bool = True,
    ):
        self.path  = hdf5_path
        self.norm  = normalize
        self.t_norm = temp_normalize

        with h5py.File(hdf5_path, "r") as f:
            self.n_samples  = f["spectra"].shape[0]
            self.n_points   = f["spectra"].shape[1]
            self.wavelengths = f["wavelengths"][:]

        # Pre-compute stats if not provided
        if normalize and stats is None:
            with h5py.File(hdf5_path, "r") as f:
                sp = f["spectra"][:]
                self.spec_mean = float(sp.mean())
                self.spec_std  = float(sp.std()) + 1e-8
        elif normalize and stats is not None:
            self.spec_mean = stats["spec_mean"]
            self.spec_std  = stats["spec_std"]
        else:
            self.spec_mean = 0.0
            self.spec_std  = 1.0

        if temp_normalize and stats is None:
            with h5py.File(hdf5_path, "r") as f:
                t = f["temperature"][:]
                self.temp_mean = float(t.mean())
                self.temp_std  = float(t.std()) + 1e-8
        elif temp_normalize and stats is not None:
            self.temp_mean = stats.get("temp_mean", 35.0)
            self.temp_std  = stats.get("temp_std", 5.77)
        else:
            self.temp_mean = 0.0
            self.temp_std  = 1.0

    def get_stats(self) -> Dict:
        return {
            "spec_mean": self.spec_mean,
            "spec_std" : self.spec_std,
            "temp_mean": self.temp_mean,
            "temp_std" : self.temp_std,
        }

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        with h5py.File(self.path, "r") as f:
            spec = f["spectra"][idx].astype(np.float32)
            temp = float(f["temperature"][idx])

        if self.norm:
            spec = (spec - self.spec_mean) / self.spec_std
        if self.t_norm:
            temp = (temp - self.temp_mean) / self.temp_std

        return {
            "spectrum"   : torch.from_numpy(spec),
            "temperature": torch.tensor(temp, dtype=torch.float32),
        }


class RealDataDataset(Dataset):
    """Wraps pre-loaded real data dict."""
    def __init__(self, data: Dict, stats: Optional[Dict] = None):
        self.spectra     = data["spectra"].astype(np.float32)
        self.temperature = data["temperature"].astype(np.float32)
        if stats:
            self.spectra     = (self.spectra - stats["spec_mean"]) / stats["spec_std"]
            self.temperature = (self.temperature - stats["temp_mean"]) / stats["temp_std"]

    def __len__(self) -> int:
        return len(self.spectra)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "spectrum"   : torch.from_numpy(self.spectra[idx]),
            "temperature": torch.tensor(self.temperature[idx], dtype=torch.float32),
        }


def build_dataloaders(cfg: dict) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Build train / val / test DataLoaders from config.
    Returns loaders + stats dict (for de-normalisation).
    """
    d = cfg["data"]

    train_ds = SpectraDataset(
        d["train_path"],
        normalize=d["normalize"],
        temp_normalize=d["temperature_normalize"],
    )
    stats = train_ds.get_stats()

    val_ds = SpectraDataset(
        d["val_path"],
        normalize=d["normalize"],
        stats=stats,
        temp_normalize=d["temperature_normalize"],
    )
    test_ds = SpectraDataset(
        d["test_path"],
        normalize=d["normalize"],
        stats=stats,
        temp_normalize=d["temperature_normalize"],
    )

    # Optional: augment val with real data
    if d["real_data"]["enabled"]:
        try:
            import sys
            sys.path.insert(0, ".")
            from data.real_data_loader import load_real_data
            real = load_real_data(
                sources=d["real_data"]["sources"],
                download=d["real_data"]["download"],
                n_points=d["n_points"]
            )
            if real is not None:
                real_ds  = RealDataDataset(real, stats=stats)
                from torch.utils.data import ConcatDataset
                val_ds   = ConcatDataset([val_ds, real_ds])
                print(f"[DataLoader] Added {len(real_ds)} real samples to val set.")
        except Exception as e:
            warnings.warn(f"[DataLoader] Real data merge failed: {e}")

    workers = min(d["num_workers"], os.cpu_count() or 1)

    train_loader = DataLoader(
        train_ds, batch_size=d["batch_size"],
        shuffle=True,  num_workers=workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,   batch_size=d["batch_size"],
        shuffle=False, num_workers=workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds,  batch_size=d["batch_size"],
        shuffle=False, num_workers=workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader, stats
