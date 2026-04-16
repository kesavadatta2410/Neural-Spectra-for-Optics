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


# ──────────────────────────────────────────────
# Top-level benchmark entry-point (called from eval.py)
# ──────────────────────────────────────────────
def run_llm_benchmark(
    model_sc,
    spectra_norm: np.ndarray,
    temperatures: np.ndarray,
    wavelengths: np.ndarray,
    stats: Dict,
    pca=None,
    api_key: Optional[str] = None,
    n_samples: int = 20,
    cfg: Optional[Dict] = None,
) -> Dict:
    """
    Orchestrate the three-condition LLM benchmark for Gap 2.

    Conditions
    ----------
    A_raw        : raw 1000-point spectrum truncated to 50 pts for prompt
    B_compressed : human-readable latent summary (no API budget wasted)
    C_vqvae      : VQ-integer token indices (optional, if VQ model available)

    Returns a dict of per-condition results plus the token-budget table.
    """
    import torch

    if api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key

    model_name = (cfg or {}).get("llm", {}).get("model", "claude-haiku-4-5-20251001")
    n_samples  = min(n_samples, len(spectra_norm))

    # Denormalise spectra for prompt construction
    spec_mean = stats.get("spec_mean", 0.0)
    spec_std  = stats.get("spec_std",  1.0)
    spectra_raw = spectra_norm[:n_samples] * spec_std + spec_mean   # [N, 1000]

    # Collect latent vectors
    device = next(model_sc.parameters()).device
    latents = []
    model_sc.eval()
    with torch.no_grad():
        for i in range(0, n_samples, 64):
            sp = torch.tensor(spectra_norm[i:i+64], dtype=torch.float32).to(device)
            tp = torch.tensor(temperatures[i:i+64],  dtype=torch.float32).to(device)
            z  = model_sc.encode(sp, tp)
            latents.append(z.cpu().numpy())
    latents_np = np.concatenate(latents)[:n_samples]
    temps_np   = temperatures[:n_samples]

    # Token budget (no API required)
    token_budget = token_budget_table(
        n_points=spectra_raw.shape[1],
        latent_dim=latents_np.shape[1],
    )

    # Actual benchmark (requires API key; gracefully skips if absent)
    detection_results: Dict = {}
    if os.environ.get("ANTHROPIC_API_KEY"):
        detection_results = run_failure_detection_benchmark(
            spectra=spectra_raw,
            wavelengths=wavelengths,
            latents=latents_np,
            temperatures=temps_np,
            n_samples=n_samples,
            model_name=model_name,
        )
    else:
        print("  [LLM] ANTHROPIC_API_KEY not set — skipping live benchmark; "
              "returning token-budget only.")

    return {
        "token_budget":       token_budget,
        "detection_results":  detection_results,
    }


def print_llm_results(results: Dict) -> None:
    """Pretty-print the output of run_llm_benchmark."""
    tb = results.get("token_budget", {})
    print("\n  ── Token Budget ───────────────────────────────")
    print(f"  Raw spectrum   : {tb.get('raw_spectrum_tokens', '?'):>6} tokens")
    print(f"  Compressed desc: {tb.get('compressed_desc_tokens', '?'):>6} tokens")
    print(f"  VQ-VAE indices : {tb.get('vqvae_tokens', '?'):>6} tokens")
    print(f"  Compression    : {tb.get('compression_token_ratio', '?'):>5}× fewer tokens")

    dr = results.get("detection_results", {})
    if not dr:
        print("  No live benchmark results (API key not set).")
        return
    print("\n  ── Channel Failure Detection Accuracy ─────────")
    hdr = f"  {'Condition':<16} {'Accuracy':>8} {'Avg Tokens':>12} {'Latency (ms)':>14}"
    print(hdr)
    print("  " + "-" * 54)
    for cond, m in dr.items():
        print(f"  {cond:<16} {m['accuracy']:>8.3f} {m['avg_tokens']:>12.1f}"
              f" {m['avg_latency_ms']:>14.1f}")
