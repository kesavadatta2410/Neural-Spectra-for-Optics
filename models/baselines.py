"""
SpecCompress-India v2: SOTA Baseline Models (Gap 4)

Baselines:
  - VanillaAutoencoder : plain MLP encoder-decoder
  - DeepCompress       : CNN + self-attention encoder-decoder
  - VQVAE              : vector-quantised VAE
  - RateDistortionAE   : AE with L1 entropy-rate penalty on latent
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict


class VanillaAutoencoder(nn.Module):
    def __init__(self, n_points=1000, latent_dim=10):
        super().__init__()
        self.n_points, self.latent_dim = n_points, latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(n_points, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, latent_dim))
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, n_points))

    def forward(self, x, temp=None):
        return self.decoder(self.encoder(x)), self.encoder(x)

    @property
    def compression_ratio(self):
        return self.n_points / self.latent_dim


class SpectralAttention(nn.Module):
    def __init__(self, channels, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        xt = x.permute(0, 2, 1)
        out, _ = self.attn(xt, xt, xt)
        return self.norm(out + xt).permute(0, 2, 1)


class DeepCompress(nn.Module):
    def __init__(self, n_points=1000, latent_dim=10, dropout=0.1):
        super().__init__()
        self.n_points, self.latent_dim = n_points, latent_dim
        self.enc_cnn = nn.Sequential(
            nn.Conv1d(1, 32, 7, stride=2, padding=3), nn.GELU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(64, 128, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv1d(128, 64, 3, stride=2, padding=1), nn.GELU())
        self.enc_attn = SpectralAttention(64, heads=4)
        with torch.no_grad():
            d = torch.zeros(1, 1, n_points)
            co = self.enc_cnn(d)
            self._L = co.shape[2]
            fd = 64 * self._L
        self.enc_proj = nn.Linear(fd, latent_dim)
        self.dec_proj = nn.Linear(latent_dim, fd)
        self.dec_cnn = nn.Sequential(
            nn.ConvTranspose1d(64, 128, 4, stride=2, padding=1), nn.GELU(),
            nn.ConvTranspose1d(128, 64, 4, stride=2, padding=1), nn.GELU(),
            nn.ConvTranspose1d(64, 32, 4, stride=2, padding=1), nn.GELU(),
            nn.ConvTranspose1d(32, 1, 4, stride=2, padding=1))

    def encode(self, x):
        h = self.enc_cnn(x.unsqueeze(1))
        h = self.enc_attn(h)
        return self.enc_proj(h.flatten(1))

    def decode(self, z):
        h = self.dec_proj(z).view(z.shape[0], 64, self._L)
        h = self.dec_cnn(h).squeeze(1)
        if h.shape[1] > self.n_points:
            h = h[:, :self.n_points]
        elif h.shape[1] < self.n_points:
            h = F.pad(h, (0, self.n_points - h.shape[1]))
        return h

    def forward(self, x, temp=None):
        return self.decode(self.encode(x)), self.encode(x)

    @property
    def compression_ratio(self):
        return self.n_points / self.latent_dim


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=10, commitment_cost=0.25):
        super().__init__()
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, z):
        d = (torch.sum(z ** 2, dim=1, keepdim=True)
             - 2 * z @ self.embedding.weight.T
             + torch.sum(self.embedding.weight ** 2, dim=1))
        idx = torch.argmin(d, dim=1)
        zq = self.embedding(idx)
        loss = (F.mse_loss(zq.detach(), z)
                + self.commitment_cost * F.mse_loss(zq, z.detach()))
        zq = z + (zq - z).detach()
        return zq, idx, loss


class VQVAE(nn.Module):
    def __init__(self, n_points=1000, latent_dim=10, num_embeddings=512):
        super().__init__()
        self.n_points, self.latent_dim = n_points, latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(n_points, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, latent_dim))
        self.vq = VectorQuantizer(num_embeddings, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, n_points))

    def forward(self, x, temp=None):
        z = self.encoder(x)
        zq, idx, vql = self.vq(z)
        xh = self.decoder(zq)
        rl = F.mse_loss(xh, x)
        return xh, idx, {
            "loss_total": (rl + vql).item(),
            "loss_recon": rl.item(),
            "loss_vq":    vql.item(),
        }

    def encode_to_tokens(self, x):
        z = self.encoder(x)
        _, idx, _ = self.vq(z)
        return idx

    @property
    def compression_ratio(self):
        return self.n_points / self.latent_dim


# ── Rate-Distortion Autoencoder ───────────────────────────────────────────────
class RateDistortionAE(nn.Module):
    """
    Autoencoder with an L1 entropy-rate penalty on the latent vector.

    Loss = MSE(x_hat, x)  +  lambda_rate * ||z||_1

    The L1 norm promotes sparsity in the latent space, approximating
    a learned entropy-coding bottleneck at low compute cost.
    """

    def __init__(self, n_points: int = 1000, latent_dim: int = 10,
                 lambda_rate: float = 1e-3):
        super().__init__()
        self.n_points    = n_points
        self.latent_dim  = latent_dim
        self.lambda_rate = lambda_rate

        self.encoder = nn.Sequential(
            nn.Linear(n_points, 512), nn.GELU(),
            nn.Linear(512, 256),      nn.GELU(),
            nn.Linear(256, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.GELU(),
            nn.Linear(256, 512),        nn.GELU(),
            nn.Linear(512, n_points),
        )

    def forward(self, x, temp=None):
        z     = self.encoder(x)
        x_hat = self.decoder(z)
        rec_loss  = F.mse_loss(x_hat, x)
        rate_loss = self.lambda_rate * z.abs().mean()
        total     = rec_loss + rate_loss
        return (
            x_hat, z,
            {
                "loss_total": total.item(),
                "loss_recon": rec_loss.item(),
                "loss_rate":  rate_loss.item(),
            },
        )

    @property
    def compression_ratio(self):
        return self.n_points / self.latent_dim


# ── Shared helpers ─────────────────────────────────────────────────────────────
def train_baseline(model, train_loader, device, epochs=10, lr=1e-3):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    for epoch in range(epochs):
        model.train()
        total, n = 0.0, 0
        for batch in train_loader:
            x = batch["spectrum"].to(device)
            out = model(x)
            loss = F.mse_loss(out[0], x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            n += 1
        if (epoch + 1) % max(1, epochs // 3) == 0:
            print(f"    [{model.__class__.__name__}] epoch {epoch+1}/{epochs}"
                  f" loss={total/n:.4f}")
    return model


def eval_baseline(model, loader, device):
    model.eval()
    all_hat, all_true = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["spectrum"].to(device)
            out = model(x)
            all_hat.append(out[0].cpu().numpy())
            all_true.append(x.cpu().numpy())
    return np.concatenate(all_hat), np.concatenate(all_true)
