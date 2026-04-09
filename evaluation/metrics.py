"""
SpecCompress-India: Evaluation Metrics
RMSE, reconstruction error, SNR — all in dB-space.
"""

import numpy as np
import torch
from typing import Dict, Optional


def rmse_db(
    x_hat: np.ndarray,
    x:     np.ndarray
) -> float:
    """
    RMSE in the original (dB) domain after de-normalisation.
    Args:
        x_hat, x : [N, L] arrays (normalised or raw)
    Returns:
        scalar RMSE (same units as input)
    """
    return float(np.sqrt(np.mean((x_hat - x) ** 2)))


def per_sample_rmse(x_hat: np.ndarray, x: np.ndarray) -> np.ndarray:
    """RMSE per sample → [N]."""
    return np.sqrt(np.mean((x_hat - x) ** 2, axis=-1))


def reconstruction_error_stats(
    x_hat: np.ndarray,
    x:     np.ndarray
) -> Dict[str, float]:
    """Mean/median/std/max of per-sample RMSE."""
    per_s = per_sample_rmse(x_hat, x)
    return {
        "rmse_mean"  : float(per_s.mean()),
        "rmse_median": float(np.median(per_s)),
        "rmse_std"   : float(per_s.std()),
        "rmse_max"   : float(per_s.max()),
        "rmse_p95"   : float(np.percentile(per_s, 95)),
    }


def snr_db(x_hat: np.ndarray, x: np.ndarray) -> float:
    """
    Signal-to-Noise Ratio (dB) of reconstruction.
    SNR = 10 * log10( var(x) / var(x - x_hat) )
    """
    signal_power = np.var(x)
    noise_power  = np.var(x - x_hat) + 1e-12
    return float(10.0 * np.log10(signal_power / noise_power))


def power_conservation_error(x_hat: np.ndarray, x: np.ndarray) -> float:
    """
    Mean absolute error in spectral power (trapezoidal sum).
    """
    power_true = x.mean(axis=-1)
    power_hat  = x_hat.mean(axis=-1)
    return float(np.mean(np.abs(power_hat - power_true)))


def smoothness_score(x_hat: np.ndarray) -> float:
    """Mean L2 of first differences — lower = smoother."""
    diff = np.diff(x_hat, axis=-1)
    return float(np.mean(diff ** 2))


def compute_all_metrics(
    x_hat:    np.ndarray,
    x:        np.ndarray,
    label:    str = "model"
) -> Dict[str, float]:
    """Compute full metric suite. Returns flat dict."""
    stats = reconstruction_error_stats(x_hat, x)
    return {
        f"{label}/rmse_mean"   : stats["rmse_mean"],
        f"{label}/rmse_median" : stats["rmse_median"],
        f"{label}/rmse_p95"    : stats["rmse_p95"],
        f"{label}/snr_db"      : snr_db(x_hat, x),
        f"{label}/power_err"   : power_conservation_error(x_hat, x),
        f"{label}/smoothness"  : smoothness_score(x_hat),
    }
