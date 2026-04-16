"""
Real Data Loader — SECONDARY pipeline.
Sources:
  1. COSMOS EDFA Dataset (GitHub: geeanloong/COSMOS-EDFA-dataset)
  2. GNPy optical network simulator gain tables
Both are OPTIONAL. Falls back to synthetic if unavailable.
"""

import os
import warnings
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict

try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


# ──────────────────────────────────────────────
# COSMOS EDFA Loader
# ──────────────────────────────────────────────
COSMOS_GITHUB_URL = (
    "https://raw.githubusercontent.com/geeanloong/"
    "COSMOS-EDFA-dataset/main/data/edfa_data.csv"
)

def load_cosmos_edfa(
    local_path: Optional[str] = None,
    download: bool = True,
    cache_dir: str = "data/real/cosmos"
) -> Optional[Dict[str, np.ndarray]]:
    """
    Load COSMOS EDFA dataset.
    Args:
        local_path : path to pre-downloaded CSV
        download   : auto-download if not found
        cache_dir  : where to cache
    Returns:
        dict with keys: spectra, wavelengths, metadata
        OR None if unavailable
    """
    import csv

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_file = cache_dir / "edfa_data.csv"

    # Prefer explicit local path
    target = Path(local_path) if local_path else cached_file

    if not target.exists():
        if download and REQUESTS_AVAILABLE:
            print(f"[RealData] Downloading COSMOS EDFA → {cached_file}")
            try:
                r = requests.get(COSMOS_GITHUB_URL, timeout=30)
                r.raise_for_status()
                cached_file.write_bytes(r.content)
                target = cached_file
                print("[RealData] COSMOS download successful.")
            except Exception as e:
                warnings.warn(f"[RealData] COSMOS download failed: {e}")
                return None
        else:
            warnings.warn(f"[RealData] COSMOS file not found at {target}. Skipping.")
            return None

    # Parse CSV
    try:
        rows = []
        with open(target, newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                rows.append([float(x) for x in row])
        data = np.array(rows, dtype=np.float32)
        # Assume last column = temperature if present
        if data.shape[1] > 1:
            spectra     = data[:, :-1]
            temperature = data[:, -1]
        else:
            spectra     = data
            temperature = np.full(len(data), 25.0, dtype=np.float32)

        n_pts = spectra.shape[1]
        wavelengths = np.linspace(1530.0, 1565.0, n_pts)
        print(f"[RealData] COSMOS loaded: {spectra.shape[0]} samples, {n_pts} points each.")
        return {
            "spectra"    : spectra,
            "temperature": temperature,
            "wavelengths": wavelengths,
            "source"     : "COSMOS-EDFA"
        }
    except Exception as e:
        warnings.warn(f"[RealData] COSMOS parse error: {e}")
        return None


# ──────────────────────────────────────────────
# GNPy Gain Table Loader
# ──────────────────────────────────────────────
GNPY_GITHUB_URL = (
    "https://raw.githubusercontent.com/Telecominfraproject/"
    "oopt-gnpy/master/gnpy/network_elements/amps.json"
)

def load_gnpy_gain_tables(
    local_path: Optional[str] = None,
    download: bool = True,
    cache_dir: str = "data/real/gnpy",
    n_points: int = 1000
) -> Optional[Dict[str, np.ndarray]]:
    """
    Load GNPy EDFA gain profiles from JSON amp definitions.
    Synthesises spectra from gain/NF/frequency tables.
    Returns dict with keys: spectra, wavelengths, metadata
    OR None if unavailable.
    """
    import json

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_file = cache_dir / "amps.json"

    target = Path(local_path) if local_path else cached_file
    if not target.exists():
        if download and REQUESTS_AVAILABLE:
            print(f"[RealData] Downloading GNPy amp tables → {cached_file}")
            try:
                r = requests.get(GNPY_GITHUB_URL, timeout=30)
                r.raise_for_status()
                cached_file.write_bytes(r.content)
                target = cached_file
                print("[RealData] GNPy download successful.")
            except Exception as e:
                warnings.warn(f"[RealData] GNPy download failed: {e}")
                return None
        else:
            warnings.warn(f"[RealData] GNPy file not found at {target}. Skipping.")
            return None

    try:
        with open(target) as f:
            amps = json.load(f)

        wavelengths = np.linspace(1530.0, 1565.0, n_points)
        spectra     = []
        temperatures = []

        for amp_name, amp_data in amps.items():
            # GNPy stores gain_flatmax and nf in some versions
            if isinstance(amp_data, dict):
                gain_max = amp_data.get("gain_flatmax", 20.0)
                # Build Gaussian approximation from single gain value
                spec = gain_max * np.exp(
                    -0.5 * ((wavelengths - 1550.0) / 12.0) ** 2
                ).astype(np.float32)
                spectra.append(spec)
                temperatures.append(25.0)

        if not spectra:
            warnings.warn("[RealData] No valid GNPy amp entries found.")
            return None

        spectra_arr = np.stack(spectra).astype(np.float32)
        temp_arr    = np.array(temperatures, dtype=np.float32)
        print(f"[RealData] GNPy loaded: {spectra_arr.shape[0]} amps.")
        return {
            "spectra"    : spectra_arr,
            "temperature": temp_arr,
            "wavelengths": wavelengths,
            "source"     : "GNPy"
        }
    except Exception as e:
        warnings.warn(f"[RealData] GNPy parse error: {e}")
        return None


# ──────────────────────────────────────────────
# Unified Loader
# ──────────────────────────────────────────────
def load_real_data(
    sources: list = ["cosmos", "gnpy"],
    cosmos_path: Optional[str] = None,
    gnpy_path: Optional[str] = None,
    download: bool = True,
    n_points: int = 1000
) -> Optional[Dict[str, np.ndarray]]:
    """
    Try each source; merge and return combined dict.
    Returns None if all sources fail.
    """
    all_spectra     = []
    all_temperature = []
    all_sources     = []
    ref_wavelengths = None

    if "cosmos" in sources:
        d = load_cosmos_edfa(local_path=cosmos_path, download=download)
        if d is not None:
            all_spectra.append(d["spectra"])
            all_temperature.append(d["temperature"])
            all_sources.append(d["source"])
            ref_wavelengths = d["wavelengths"]

    if "gnpy" in sources:
        d = load_gnpy_gain_tables(local_path=gnpy_path, download=download, n_points=n_points)
        if d is not None:
            all_spectra.append(d["spectra"])
            all_temperature.append(d["temperature"])
            all_sources.append(d["source"])
            if ref_wavelengths is None:
                ref_wavelengths = d["wavelengths"]

    if not all_spectra:
        warnings.warn("[RealData] No real data sources available. Using synthetic only.")
        return None

    # Pad/crop all spectra to n_points
    merged = []
    for s in all_spectra:
        if s.shape[1] != n_points:
            # Interpolate each spectrum
            xs = np.linspace(0, 1, s.shape[1])
            xt = np.linspace(0, 1, n_points)
            s_interp = np.stack([np.interp(xt, xs, row) for row in s]).astype(np.float32)
            merged.append(s_interp)
        else:
            merged.append(s)

    combined_spectra = np.concatenate(merged, axis=0)
    combined_temp    = np.concatenate(all_temperature, axis=0)
    wavelengths      = np.linspace(1530.0, 1565.0, n_points)

    print(f"[RealData] Merged {combined_spectra.shape[0]} samples from: {all_sources}")
    return {
        "spectra"    : combined_spectra,
        "temperature": combined_temp,
        "wavelengths": wavelengths,
        "sources"    : all_sources
    }
