"""
SpecCompress-India: CPU Smoke-Tests
No pre-generated data or GPU required.
Run with: pytest tests/ -v
"""
import pytest
import torch
import yaml
from pathlib import Path

# ── Helpers ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent

def _minimal_cfg(latent_dim: int = 10) -> dict:
    """Return a minimal config dict that mirrors configs/config.yaml."""
    return {
        "model": {
            "input_dim":       1000,
            "latent_dim":      latent_dim,
            "temp_embed_dim":  16,
            "temp_min":        25.0,
            "temp_max":        45.0,
            "encoder": {
                "channels":     [1, 32, 64, 128],
                "kernel_sizes": [7, 5, 3],
                "strides":      [2, 2, 2],
                "dropout":      0.1,
            },
            "bottleneck": {"hidden_dims": [512, 256], "dropout": 0.1},
            "decoder":    {"hidden_dims": [256, 512, 1024], "dropout": 0.1},
        },
        "loss": {
            "reconstruction_weight":    1.0,
            "smoothness_weight":        0.1,
            "power_conservation_weight": 0.05,
            "osnr_penalty_weight":      0.02,
        },
    }


B, N = 4, 1000   # batch size, spectrum length


# ── Encoder ────────────────────────────────────────────────────────────────────
def test_encoder_output_shape():
    from models.encoder import Encoder
    enc = Encoder(n_points=N, latent_dim=10, temp_embed_dim=16)
    x   = torch.randn(B, N)
    t   = torch.randint(25, 46, (B,)).float()
    z   = enc(x, t)
    assert z.shape == (B, 10), f"Expected ({B}, 10), got {z.shape}"


def test_encoder_no_cnn_no_temp():
    from models.encoder import Encoder
    enc = Encoder(n_points=N, latent_dim=8, temp_embed_dim=16,
                  use_cnn=False, use_temp=False)
    x = torch.randn(B, N)
    t = torch.zeros(B)
    z = enc(x, t)
    assert z.shape == (B, 8)


# ── Decoder ────────────────────────────────────────────────────────────────────
def test_decoder_output_shape():
    from models.decoder import Decoder
    dec = Decoder(n_points=N, latent_dim=10, temp_embed_dim=16,
                  mlp_hidden=[256, 512, 1024])
    z = torch.randn(B, 10)
    t = torch.randint(25, 46, (B,)).float()
    x_hat = dec(z, t)
    assert x_hat.shape == (B, N), f"Expected ({B}, {N}), got {x_hat.shape}"


# ── PhysicsInformedLoss ────────────────────────────────────────────────────────
def test_physics_loss_all_components():
    from models.physics_loss import PhysicsInformedLoss
    loss_fn = PhysicsInformedLoss()
    x     = torch.randn(B, N)
    x_hat = torch.randn(B, N)
    total, comps = loss_fn(x_hat, x)
    assert total.item() >= 0, "Total loss must be non-negative"
    for key in ("loss_recon", "loss_smooth", "loss_power", "loss_osnr"):
        assert key in comps, f"Missing loss component: {key}"
        assert comps[key] >= 0, f"{key} must be non-negative"


def test_physics_loss_disabled_components():
    from models.physics_loss import PhysicsInformedLoss
    loss_fn = PhysicsInformedLoss(use_smoothness=False, use_power=False, use_osnr=False)
    x     = torch.randn(B, N)
    x_hat = torch.randn(B, N)
    total, comps = loss_fn(x_hat, x)
    assert comps["loss_smooth"] == 0.0
    assert comps["loss_power"]  == 0.0
    assert comps["loss_osnr"]   == 0.0


# ── SpecCompress (full model) ──────────────────────────────────────────────────
def test_speccompress_forward():
    from models.speccompress import SpecCompress
    cfg   = _minimal_cfg(latent_dim=10)
    model = SpecCompress(cfg)
    x     = torch.randn(B, N)
    t     = torch.randint(25, 46, (B,)).float()
    x_hat, z, loss, comps = model(x, t)
    assert x_hat.shape == (B, N)
    assert z.shape     == (B, 10)
    assert loss.item() >= 0


def test_speccompress_compression_ratio():
    from models.speccompress import SpecCompress
    cfg   = _minimal_cfg(latent_dim=10)
    model = SpecCompress(cfg)
    assert model.compression_ratio == pytest.approx(100.0)


def test_speccompress_count_parameters():
    from models.speccompress import SpecCompress
    cfg   = _minimal_cfg()
    model = SpecCompress(cfg)
    assert model.count_parameters() > 0


# ── Baselines ─────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("cls_name", [
    "VanillaAutoencoder", "DeepCompress", "VQVAE", "RateDistortionAE"
])
def test_baseline_forward(cls_name):
    import models.baselines as bm
    cls   = getattr(bm, cls_name)
    model = cls(n_points=N, latent_dim=10)
    x     = torch.randn(B, N)
    out   = model(x)
    assert out[0].shape == (B, N), \
        f"{cls_name}: expected output ({B},{N}), got {out[0].shape}"


# ── Config round-trip ─────────────────────────────────────────────────────────
def test_config_loads():
    cfg_path = ROOT / "configs" / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    assert "model"    in cfg
    assert "training" in cfg
    assert "loss"     in cfg
