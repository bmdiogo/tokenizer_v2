"""Baselines at the same budget as the codec (see also pre-flight test T5).

Each is a function x_n [C, T] -> x̂ [C, T]; its x̂ goes through exactly the same readouts and
metrics as the codec's. Budget: K numbers per F samples (the codec: 32 per 240 = "4 kHz").
Only LinearAE is trained by gradient descent (scripts/train_linear_ae.py).
"""
from math import gcd

import numpy as np
import torch
from scipy.signal import resample_poly
from torch import nn

from utils.config import HZ


class PCACodec:
    """The optimal linear codec under MSE (Eckart-Young): every F-sample frame of every electrode is
    kept as its K coefficients on the top-K principal directions of TRAINING frames."""

    def __init__(self, F=240, K=32):
        self.F, self.K = F, K

    def fit(self, train_sessions, seconds=2.0, max_frames=200_000, seed=0, spike_weight=1.0):
        """spike_weight > 1: frames holding a NIX Good spike (the mask m that L_wave sees) weigh more in the
        mean and covariance, so the linear code gets the codec's labels; applying it needs no labels."""
        n = int(seconds * HZ) // self.F * self.F
        X = np.concatenate([s.x(0, n).reshape(-1, self.F) for s in train_sessions])
        keep = np.random.default_rng(seed).permutation(len(X))[:max_frames]
        X = X[keep]
        if spike_weight == 1.0:
            self.mu = X.mean(0)
            self.U = np.linalg.svd(X - self.mu, full_matrices=False)[2][:self.K]  # [K, F]
            return self
        spk = np.concatenate([s.m(0, n).reshape(-1, self.F).any(1) for s in train_sessions])[keep]
        w = np.where(spk, spike_weight, 1.0).astype(np.float32)[:, None]
        self.mu = (w * X).sum(0) / w.sum()
        self.U = np.linalg.svd(np.sqrt(w) * (X - self.mu), full_matrices=False)[2][:self.K]
        return self

    @classmethod
    def from_stats(cls, stats, K=32, spike_weight=1.0):
        """Every training frame, from the sums of frame_stats: the same solution as fit() on all of them."""
        extra = spike_weight - 1.0
        n = stats["n"] + extra * stats["n_spk"]
        mu = (stats["sum"] + extra * stats["sum_spk"]) / n
        cov = stats["xx"] + extra * stats["xx_spk"] - n * np.outer(mu, mu)
        self = cls(len(mu), K)
        self.mu = mu.astype(np.float32)
        self.U = np.linalg.eigh(cov)[1][:, ::-1][:, :K].T.astype(np.float32)   # [K, F]
        return self

    def __call__(self, x):
        C, T = x.shape
        n = T // self.F * self.F
        frames = x[:, :n].reshape(C, -1, self.F) - self.mu
        out = x.copy()                                  # a trailing partial frame is left as is
        out[:, :n] = ((frames @ self.U.T) @ self.U + self.mu).reshape(C, n)
        return out


def frame_stats(train_sessions, F, read=288_000):
    """One pass over every F-sample frame of every electrode, in `read`-sample pieces: count, sum and
    X'X of all frames and of the frames holding a sample of the mask m. Any spike weight follows from these."""
    read = read // F * F
    z = dict(n=0.0, sum=np.zeros(F), xx=np.zeros((F, F)), n_spk=0.0, sum_spk=np.zeros(F), xx_spk=np.zeros((F, F)))
    for s in train_sessions:
        end = s.T // F * F
        for t0 in range(0, end, read):
            n = min(read, end - t0)
            X = s.x(t0, n).reshape(-1, F).astype(np.float64)
            spk = X[s.m(t0, n).reshape(-1, F).any(1)]
            z["n"], z["sum"], z["xx"] = z["n"] + len(X), z["sum"] + X.sum(0), z["xx"] + X.T @ X
            z["n_spk"] += len(spk)
            z["sum_spk"] = z["sum_spk"] + spk.sum(0)
            z["xx_spk"] = z["xx_spk"] + spk.T @ spk
        print(f"  {s.name}: {z['n']:.0f} frames", flush=True)
    return z


class LinearAE(nn.Module):
    """The trained linear autoencoder: one fully-connected layer each way, no activation. Trained with
    MSE it converges to the PCA subspace (Bourlard & Kamp 1988; Baldi & Hornik 1989). One frame of latency."""

    def __init__(self, F=240, K=32):
        super().__init__()
        self.F, self.K = F, K
        self.enc, self.dec = nn.Linear(F, K), nn.Linear(K, F)

    def forward(self, frames):
        return self.dec(self.enc(frames))

    @torch.no_grad()
    def reconstruct(self, x):
        C, T = x.shape
        n = T // self.F * self.F
        out = x.copy()                                  # a trailing partial frame is left as is
        out[:, :n] = self(torch.from_numpy(x[:, :n].reshape(-1, self.F))).numpy().reshape(C, n)
        return out


def low_rate(x, F=240, K=32):
    """Record at K / F of the rate (4 kHz for 32 / 240), then interpolate back to 30 kHz."""
    rate = K * HZ // F
    g = gcd(rate, HZ)
    y = resample_poly(x, rate // g, HZ // g, axis=-1)
    return resample_poly(y, HZ // g, rate // g, axis=-1)[..., :x.shape[-1]].astype(np.float32)


LOWPASS = {  # orthonormal scaling filters (PyWavelets rec_lo), so no extra dependency in the container
    "haar": [0.7071067811865476, 0.7071067811865476],
    "db4": [0.2303778133088965, 0.7148465705529157, 0.6308807679298589, -0.027983769416859854,
            -0.18703481171909309, 0.030841381835560764, 0.0328830116668852, -0.010597401785069032],
    "sym4": [0.0322231006040427, -0.012603967262037833, -0.09921954357684722, 0.29785779560527736,
             0.8037387518059161, 0.49761866763201545, -0.02963552764599851, -0.07576571478927333],
}


def dwt_matrix(F, wavelet="sym4"):
    """[F, F] orthonormal periodic DWT of one frame, as many levels as halve it while it is even and
    at least one filter long (240 -> 4 levels, 480 -> 5). Rows are the wavelet basis functions."""
    h = np.asarray(LOWPASS[wavelet])
    g = h[::-1] * (-1.0) ** np.arange(len(h))           # quadrature mirror high-pass
    W, n = np.eye(F), F
    while n % 2 == 0 and n >= len(h):
        A = np.zeros((n, n))
        for k in range(n // 2):
            idx = (2 * k + np.arange(len(h))) % n
            np.add.at(A[k], idx, h)
            np.add.at(A[n // 2 + k], idx, g)
        step = np.eye(F)
        step[:n, :n] = A                                 # transform only the approximation part
        W, n = step @ W, n // 2
    return W


class WaveletCodec:
    """Adaptive, label-free baseline: each F-sample frame keeps its K largest wavelet coefficients.
    Which K is chosen per frame, so unlike PCA it follows the spike; the positions are not counted in
    the budget, which favours it. Same basis as PyWavelets 'periodization' up to a circular shift."""

    def __init__(self, F=240, K=32, wavelet="sym4"):
        self.F, self.K, self.W = F, K, dwt_matrix(F, wavelet).astype(np.float32)

    def __call__(self, x):
        C, T = x.shape
        n = T // self.F * self.F
        out = x.copy()                                  # a trailing partial frame is left as is
        for c in range(C):                              # one electrode at a time keeps memory small
            coef = x[c, :n].reshape(-1, self.F) @ self.W.T
            kth = np.partition(np.abs(coef), self.F - self.K, axis=1)[:, self.F - self.K, None]
            out[c, :n] = ((coef * (np.abs(coef) >= kth)) @ self.W).reshape(-1)
        return out
