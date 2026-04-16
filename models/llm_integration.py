"""
SpecCompress-India v2: LLM Integration (Gap 2)

Conditions benchmarked:
  A. Raw spectrum (1000 floats, truncated to 50 pts for prompt) ~150 tokens
  B. Compressed description from latent vector           ~50 tokens
  C. VQ-VAE integer token indices                        ~latent_dim tokens

Tasks:
  1. channel_failure_detection
  2. few_shot_failure_classification
"""

import json, time, os
import numpy as np
import requests
from typing import Dict, List, Tuple, Optional


# ──────────────────────────────────────────────
# Serialisers
# ──────────────────────────────────────────────
def spectrum_to_raw_text(spectrum: np.ndarray, wavelengths: np.ndarray, decimals: int = 2) -> str:
    vals = ", ".join(f"{v:.{decimals}f}" for v in spectrum)
    return f"Spectrum (gain dB, {wavelengths[0]:.0f}-{wavelengths[-1]:.0f}nm):\n[{vals}]"


def latent_to_compressed_description(
    z: np.ndarray, temperature: float, wavelengths: np.ndarray,
    spectrum: np.ndarray, threshold_db: float = 3.0
) -> str:
    peak_gain   = float(spectrum.max())
    peak_wl     = float(wavelengths[spectrum.argmax()])
    mean_gain   = float(spectrum.mean())
    tilt        = float(spectrum[-50:].mean() - spectrum[:50].mean())
    ripple_std  = float(np.diff(spectrum).std())
    smooth      = np.convolve(spectrum, np.ones(20)/20, mode="same")
    anomaly_sc  = float(np.abs(spectrum - smooth).max())
    has_anomaly = anomaly_sc > threshold_db
    z_l2        = float(np.linalg.norm(z))
    z_top3      = ", ".join(f"{v:.3f}" for v in z[:3])
    return (
        f"EDFA Spectrum Summary:\n"
        f"  Temperature: {temperature:.1f}°C\n"
        f"  Peak gain: {peak_gain:.1f} dB @ {peak_wl:.1f}nm\n"
        f"  Mean gain: {mean_gain:.1f} dB\n"
        f"  Spectral tilt: {tilt:+.2f} dB\n"
        f"  Ripple std: {ripple_std:.3f} dB\n"
        f"  Anomaly: {'YES (score=' + f'{anomaly_sc:.2f}' + ')' if has_anomaly else 'NO'}\n"
        f"  Latent [{z_top3}...] L2={z_l2:.2f}\n"
        f"  Compression: 1000→{len(z)} dims ({1000/len(z):.0f}×)"
    )


def vqvae_indices_to_text(indices: np.ndarray, temperature: float) -> str:
    return (
        f"Spectrum VQ tokens (T={temperature:.1f}°C):\n"
        f"[{' '.join(str(int(i)) for i in indices)}]"
    )


def token_budget_table(n_points: int = 1000, latent_dim: int = 10) -> Dict:
    """Estimate token counts for each representation (no API needed)."""
    wl   = np.linspace(1530, 1565, n_points)
    spec = np.random.randn(n_points) * 2 + 20
    z    = np.random.randn(latent_dim)
    idx  = np.random.randint(0, 512, latent_dim)
    # approx: 1 token ≈ 4 chars
    def approx_tokens(s): return max(1, len(s) // 4)
    raw_t  = approx_tokens(spectrum_to_raw_text(spec, wl))
    comp_t = approx_tokens(latent_to_compressed_description(z, 35.0, wl, spec))
    vq_t   = approx_tokens(vqvae_indices_to_text(idx, 35.0))
    return {
        "raw_spectrum_tokens"       : raw_t,
        "compressed_desc_tokens"    : comp_t,
        "vqvae_tokens"             : vq_t,
        "compression_token_ratio"  : round(raw_t / max(comp_t, 1), 1),
    }


# ──────────────────────────────────────────────
# Anomaly injection
# ──────────────────────────────────────────────
def inject_channel_failure(
    spectrum: np.ndarray, wavelengths: np.ndarray,
    failure_wl: float = 1550.0, drop_db: float = 8.0, width_nm: float = 1.5
) -> np.ndarray:
    notch = drop_db * np.exp(-0.5*((wavelengths-failure_wl)/(width_nm/2.355))**2)
    return spectrum - notch


# ──────────────────────────────────────────────
# API caller
# ──────────────────────────────────────────────
def call_claude(prompt: str, model: str = "claude-haiku-4-5-20251001",
                max_tokens: int = 256, system: str = "Answer only with JSON.") -> Tuple[str,int,float]:
    api_key = os.environ.get("ANTHROPIC_API_KEY","")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set")
    t0   = time.time()
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": api_key, "anthropic-version": "2023-06-01",
                 "content-type": "application/json"},
        json={"model": model, "max_tokens": max_tokens, "system": system,
              "messages": [{"role": "user", "content": prompt}]},
        timeout=30
    )
    resp.raise_for_status()
    data    = resp.json()
    latency = (time.time()-t0)*1000
    text    = data["content"][0]["text"]
    tokens  = data["usage"]["input_tokens"]+data["usage"]["output_tokens"]
    return text, tokens, latency


# ──────────────────────────────────────────────
# Task: channel failure detection benchmark
# ──────────────────────────────────────────────
FAILURE_MODES = {
    "channel_drop"       : dict(failure_wl=1550.0, drop_db=8.0, width_nm=1.5),
    "gain_tilt"          : dict(failure_wl=1540.0, drop_db=3.0, width_nm=8.0),
    "wideband_degradation": dict(failure_wl=1548.0, drop_db=5.0, width_nm=6.0),
}


def run_failure_detection_benchmark(
    spectra: np.ndarray, wavelengths: np.ndarray,
    latents: np.ndarray, temperatures: np.ndarray,
    n_samples: int = 20, model_name: str = "claude-haiku-4-5-20251001",
    vq_indices: Optional[np.ndarray] = None,
) -> Dict:
    np.random.seed(42)
    N        = min(n_samples, len(spectra))
    labels   = np.zeros(N, dtype=bool)
    test_sp  = spectra[:N].copy()
    for i in range(N//2):
        test_sp[i] = inject_channel_failure(spectra[i], wavelengths)
        labels[i]  = True

    conditions = {
        "A_raw"       : lambda i: spectrum_to_raw_text(test_sp[i], wavelengths),
        "B_compressed": lambda i: latent_to_compressed_description(latents[i], temperatures[i], wavelengths, test_sp[i]),
    }
    if vq_indices is not None:
        conditions["C_vqvae"] = lambda i: vqvae_indices_to_text(vq_indices[i], temperatures[i])

    results = {}
    for cond, repr_fn in conditions.items():
        correct=0; tok_tot=0; lat_tot=0.0; errs=0
        for i in range(N):
            try:
                prompt = (f"{repr_fn(i)}\n\nIs there a channel failure (>3dB drop) near 1550nm?\n"
                          f"JSON only: {{\"failure_detected\": true/false, \"confidence\": 0-1}}")
                text, tok, lat = call_claude(prompt, model=model_name)
                pred  = json.loads(text.strip()).get("failure_detected", False)
                if pred == labels[i]: correct += 1
                tok_tot += tok; lat_tot += lat
            except Exception as e:
                errs += 1; print(f"  [{cond}] err: {e}")
        n_valid = N-errs
        results[cond] = {
            "accuracy"      : round(correct/max(n_valid,1), 3),
            "avg_tokens"    : round(tok_tot/max(n_valid,1), 1),
            "avg_latency_ms": round(lat_tot/max(n_valid,1), 1),
            "errors"        : errs,
        }
        print(f"  [{cond}] acc={results[cond]['accuracy']:.2f} "
              f"tok={results[cond]['avg_tokens']:.0f} "
              f"lat={results[cond]['avg_latency_ms']:.0f}ms")
    return results
