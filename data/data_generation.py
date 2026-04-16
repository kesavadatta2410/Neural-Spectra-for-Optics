"""
SpecCompress-India: Synthetic Spectral Data Generator
Simulates EDFA gain spectra with temperature variation for Indian optical networks.
"""

import numpy as np
import h5py
import os
import argparse
from pathlib import Path
from typing import Tuple, Dict, Optional

# ──────────────────────────────────────────────
# EDFA Physics Parameters
# ──────────────────────────────────────────────
C_BAND_START_NM = 1530.0
C_BAND_END_NM   = 1565.0
L_BAND_START_NM = 1565.0
L_BAND_END_NM   = 1610.0

TEMP_MIN_C = 25.0   # °C  (India ambient floor)
TEMP_MAX_C = 45.0   # °C  (India summer ceiling)
TEMP_REF_C = 25.0   # Reference temperature

GAIN_COEFF_PER_DEG = -0.015   # dB/°C gain shift per degree
TILT_COEFF_PER_DEG =  0.003   # dB/nm/°C spectral tilt change


def generate_wavelength_grid(
    n_points: int = 1000,
    wl_start: float = C_BAND_START_NM,
    wl_end: float   = C_BAND_END_NM
) -> np.ndarray:
    """Return uniform wavelength grid (nm)."""
    return np.linspace(wl_start, wl_end, n_points)


# ──────────────────────────────────────────────
# EDFA Gain Curve Simulation
# ──────────────────────────────────────────────
def edfa_base_gain(
    wavelengths: np.ndarray,
    peak_gain_db: float = 20.0,
    peak_wl_nm: float   = 1550.0,
    bandwidth_nm: float = 10.0,
    secondary_peak: bool = True
) -> np.ndarray:
    """
    Erbium gain spectrum: primary + secondary emission peaks.
    Uses Gaussian lobes for smooth physics-realistic profiles.
    """
    # Primary emission peak
    gain = peak_gain_db * np.exp(
        -0.5 * ((wavelengths - peak_wl_nm) / bandwidth_nm) ** 2
    )
    if secondary_peak:
        # Secondary lobe ~1532 nm (Er3+ stark splitting)
        sec_amplitude = 0.65 * peak_gain_db
        sec_wl        = peak_wl_nm - 18.0
        sec_bw        = 6.0
        gain += sec_amplitude * np.exp(
            -0.5 * ((wavelengths - sec_wl) / sec_bw) ** 2
        )
    # Add sinusoidal ripple (PDL + gain ripple ~0.3 dB)
    ripple_period = 5.0   # nm
    ripple_amp    = 0.15
    phase         = np.random.uniform(0, 2 * np.pi)
    gain += ripple_amp * np.sin(2 * np.pi * (wavelengths - wavelengths[0]) / ripple_period + phase)
    return gain


def apply_temperature_effects(
    gain: np.ndarray,
    wavelengths: np.ndarray,
    temperature_c: float
) -> np.ndarray:
    """
    Inject temperature physics:
      - Flat gain shift (GAIN_COEFF_PER_DEG)
      - Linear spectral tilt increasing with ΔT
    """
    delta_t = temperature_c - TEMP_REF_C
    gain_shift = GAIN_COEFF_PER_DEG * delta_t
    tilt = TILT_COEFF_PER_DEG * delta_t * (wavelengths - wavelengths[len(wavelengths) // 2])
    return gain + gain_shift + tilt


def apply_noise(
    gain: np.ndarray,
    osnr_db: float,
    gaussian_std: float = 0.05
) -> np.ndarray:
    """
    Composite noise model:
      - Gaussian thermal noise
      - OSNR-based shot noise floor
    """
    thermal = np.random.normal(0, gaussian_std, gain.shape)
    osnr_linear = 10 ** (osnr_db / 10.0)
    osnr_noise_std = 1.0 / np.sqrt(osnr_linear + 1e-8)
    osnr_noise = np.random.normal(0, osnr_noise_std * 0.1, gain.shape)
    return gain + thermal + osnr_noise


def apply_random_perturbations(gain: np.ndarray, wavelengths: np.ndarray) -> np.ndarray:
    """Random per-sample physics perturbations (pump power drift, splice loss)."""
    # Random gain pedestal shift
    pedestal = np.random.uniform(-0.5, 0.5)
    # Random exponential tilt (fiber birefringence)
    exp_tilt = np.random.uniform(-0.002, 0.002) * (wavelengths - wavelengths[0])
    # Random notch (WDM channel drop artefact)
    if np.random.rand() < 0.1:
        notch_wl  = np.random.choice(wavelengths)
        notch_bw  = np.random.uniform(0.5, 2.0)
        notch_amp = np.random.uniform(0.5, 2.0)
        gain     -= notch_amp * np.exp(-0.5 * ((wavelengths - notch_wl) / notch_bw) ** 2)
    return gain + pedestal + exp_tilt


# ──────────────────────────────────────────────
# Dataset Generator
# ──────────────────────────────────────────────
def generate_sample(
    wavelengths: np.ndarray,
    config: Dict
) -> Tuple[np.ndarray, float, float, float]:
    """
    Generate one (spectrum, temperature, peak_gain, osnr) sample.
    Returns:
        spectrum     : ndarray [n_points]
        temperature  : float (°C)
        peak_gain_db : float (dB)
        osnr_db      : float (dB)
    """
    temperature  = np.random.uniform(config["temp_min"], config["temp_max"])
    peak_gain_db = np.random.uniform(config["gain_min"], config["gain_max"])
    peak_wl      = np.random.uniform(config["peak_wl_min"], config["peak_wl_max"])
    bw           = np.random.uniform(config["bw_min"], config["bw_max"])
    osnr_db      = np.random.uniform(config["osnr_min"], config["osnr_max"])

    gain = edfa_base_gain(
        wavelengths,
        peak_gain_db=peak_gain_db,
        peak_wl_nm=peak_wl,
        bandwidth_nm=bw,
        secondary_peak=config["secondary_peak"]
    )
    gain = apply_temperature_effects(gain, wavelengths, temperature)
    gain = apply_noise(gain, osnr_db, gaussian_std=config["gaussian_std"])
    gain = apply_random_perturbations(gain, wavelengths)

    return gain.astype(np.float32), float(temperature), float(peak_gain_db), float(osnr_db)


def generate_dataset(
    n_samples: int,
    config: Dict,
    output_path: str,
    seed: int = 42,
    verbose: bool = True
) -> None:
    """
    Generate full synthetic dataset; save to HDF5.
    HDF5 structure:
        /spectra       [n_samples, n_points]  float32
        /temperature   [n_samples]            float32
        /peak_gain_db  [n_samples]            float32
        /osnr_db       [n_samples]            float32
        /wavelengths   [n_points]             float64
    """
    np.random.seed(seed)
    wavelengths = generate_wavelength_grid(
        n_points=config["n_points"],
        wl_start=config["wl_start"],
        wl_end=config["wl_end"]
    )

    n_points = len(wavelengths)
    spectra      = np.zeros((n_samples, n_points), dtype=np.float32)
    temperatures = np.zeros(n_samples, dtype=np.float32)
    peak_gains   = np.zeros(n_samples, dtype=np.float32)
    osnr_vals    = np.zeros(n_samples, dtype=np.float32)

    log_every = max(1, n_samples // 20)
    for i in range(n_samples):
        spec, temp, pgain, osnr = generate_sample(wavelengths, config)
        spectra[i]      = spec
        temperatures[i] = temp
        peak_gains[i]   = pgain
        osnr_vals[i]    = osnr
        if verbose and (i + 1) % log_every == 0:
            print(f"  Generated {i + 1:>6}/{n_samples} samples", flush=True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        f.create_dataset("spectra",      data=spectra,      compression="gzip", compression_opts=4)
        f.create_dataset("temperature",  data=temperatures, compression="gzip")
        f.create_dataset("peak_gain_db", data=peak_gains,   compression="gzip")
        f.create_dataset("osnr_db",      data=osnr_vals,    compression="gzip")
        f.create_dataset("wavelengths",  data=wavelengths)
        # Save config as attributes
        for k, v in config.items():
            if isinstance(v, (int, float, bool, str)):
                f.attrs[k] = v
        f.attrs["n_samples"] = n_samples
        f.attrs["seed"]      = seed

    size_mb = os.path.getsize(output_path) / 1e6
    if verbose:
        print(f"\n[DataGen] Saved {n_samples} samples → {output_path} ({size_mb:.1f} MB)")


# ──────────────────────────────────────────────
# Default Config
# ──────────────────────────────────────────────
DEFAULT_CONFIG = {
    # Spectral grid
    "n_points"      : 1000,
    "wl_start"      : C_BAND_START_NM,
    "wl_end"        : C_BAND_END_NM,
    # Temperature
    "temp_min"      : TEMP_MIN_C,
    "temp_max"      : TEMP_MAX_C,
    # Gain
    "gain_min"      : 10.0,
    "gain_max"      : 30.0,
    "peak_wl_min"   : 1540.0,
    "peak_wl_max"   : 1560.0,
    "bw_min"        : 8.0,
    "bw_max"        : 15.0,
    # OSNR
    "osnr_min"      : 15.0,
    "osnr_max"      : 35.0,
    # Noise
    "gaussian_std"  : 0.05,
    "secondary_peak": True,
}


# ──────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SpecCompress-India: Synthetic Data Generator")
    parser.add_argument("--n_train",  type=int, default=50000, help="Training samples")
    parser.add_argument("--n_val",    type=int, default=5000,  help="Validation samples")
    parser.add_argument("--n_test",   type=int, default=5000,  help="Test samples")
    parser.add_argument("--n_points", type=int, default=1000,  help="Spectrum resolution")
    parser.add_argument("--out_dir",  type=str, default="data/synthetic")
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()

    cfg = DEFAULT_CONFIG.copy()
    cfg["n_points"] = args.n_points

    splits = {
        "train": (args.n_train, args.seed),
        "val"  : (args.n_val,   args.seed + 1),
        "test" : (args.n_test,  args.seed + 2),
    }

    print("=" * 55)
    print("  SpecCompress-India | Synthetic Data Generator")
    print("=" * 55)
    for split, (n, seed) in splits.items():
        out_path = os.path.join(args.out_dir, f"{split}.h5")
        print(f"\n[{split.upper()}] Generating {n} samples...")
        generate_dataset(n, cfg, out_path, seed=seed)
    print("\n[Done] All splits generated.")
