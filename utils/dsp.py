"""Fixed signal processing (no training): the spike band, the spike mask, the artifact flag,
and the LFP / MUA readouts. The same function is always applied to x and to x̂.

Every function works on the last axis, so [T], [C, T] and [B, C, T] all work.
"""
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal as sps

from utils.config import HZ, env

WF_PRE, WF_POST = 21, 40                  # spike window [t-21, t+40): 61 samples, NIX time = sample 21 (T1)
LFP_BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 12),
             "beta": (13, 30), "gamma": (30, 80), "high_gamma": (80, 200)}


# ------------------------------------------------------------------ spike band

def bp(s, lo=500.0, hi=7500.0, fs=HZ, order=4):
    """ElcheSort's band-pass: Butterworth |H|^2 applied in the FFT domain (zero phase, = filtfilt).

    One spike band everywhere: detection, sorting, ℒ_wave, ℒ_spike. Accepts numpy or torch
    (keeps the device, so it runs on GPU inside the loss and stays differentiable).
    """
    t = torch.as_tensor(s, dtype=torch.float32)
    n = t.shape[-1]
    n_fft = 1 << (n - 1).bit_length()
    f = torch.fft.rfftfreq(n_fft, 1.0 / fs, device=t.device)
    h = 1.0 / (1.0 + (lo / f.clamp_min(1e-10)) ** (2 * order)) / (1.0 + (f / hi) ** (2 * order))
    h[0] = 0.0
    out = torch.fft.irfft(torch.fft.rfft(t, n_fft) * h, n_fft)[..., :n]
    return out if torch.is_tensor(s) else out.numpy()


def mad_sigma(s):
    """Noise level of a zero-median signal: median(|s|) / 0.6745 (Quiroga 2004)."""
    return np.median(np.abs(s), axis=-1) / 0.6745


def robust_stats(x):
    """Per-channel median and MAD * 1.4826 (the normalisation of the model input)."""
    mu = np.median(x, axis=-1)
    return mu, np.maximum(1.4826 * np.median(np.abs(x - mu[..., None]), axis=-1), 1e-6)


# ------------------------------------------------------------------ spike mask m

@lru_cache
def _tukey(alpha):
    return sps.windows.tukey(WF_PRE + WF_POST, alpha).astype(np.float32)


def mask_window(alpha=None):
    """The 61-sample window of m: tukey(61, alpha). 0 = box, 0.25 = default (soft edges: no step
    in y), 1 = Hann. None reads MASK_ALPHA, which train() sets from the run's LossConfig (S7-E)."""
    return _tukey(float(env("MASK_ALPHA", 0.25) if alpha is None else alpha))


TUKEY = mask_window(0.25)                 # the default window (tests, figures)


def spike_mask(rows, times, n_channels, t0, n, alpha=None):
    """m on [t0, t0 + n): a tukey window over [t-21, t+40) of each Good spike, max where they overlap."""
    win = mask_window(alpha)
    m = np.zeros((n_channels, n), np.float32)
    keep = (times + WF_POST > t0) & (times - WF_PRE < t0 + n)
    rows, times = rows[keep], times[keep]
    offs = np.arange(-WF_PRE, WF_POST)
    idx = times[:, None] - t0 + offs[None]                     # [n_spikes, 61]
    ok = (idx >= 0) & (idx < n)
    r = np.broadcast_to(rows[:, None], idx.shape)
    np.maximum.at(m, (r[ok], idx[ok]), np.broadcast_to(win, idx.shape)[ok])
    return m


# ------------------------------------------------------------------ artifacts (flagged, not masked)

RAIL = 32767                      # int16 NS5 counts at the rail
ART_BEFORE, ART_AFTER = 0.001, 0.050   # every flag is padded 1 ms before (onset), 50 ms after (recovery)


def _runs(b, n):
    """True on every sample that belongs to a run of >= n consecutive True (along the last axis)."""
    c = np.cumsum(np.pad(b, [(0, 0)] * (b.ndim - 1) + [(1, 0)]), axis=-1)
    full = (c[..., n:] - c[..., :-n]) == n
    out = np.zeros_like(b)
    for j in range(n):
        out[..., j:j + full.shape[-1]] |= full
    return out


def artifact_flags(counts, x_n, bp_x, sigma_bp, fs=HZ):
    """The three artifact rules, on one chunk. -> {rule: bool [C, T] or [T]}.

    counts: raw int16 · x_n: normalised · bp_x: bp(x_n) · sigma_bp: MAD of bp(x_n) per channel.
    sat = >= 3 samples at the rail (per channel) · cm = > half the channels beyond 10 MAD at once ·
    hse = >= 10 channels cross -5 sigma within +-1 sample (hyper-synchronous, non-neuronal).
    """
    sat = _runs(np.abs(counts.astype(np.int32)) >= RAIL, 3)
    cm = (np.abs(x_n) > 10).mean(0) > 0.5
    below = bp_x < -5 * sigma_bp[:, None]
    hse = np.maximum.reduce([np.roll(below, s, -1) for s in (-1, 0, 1)]).sum(0) >= 10
    return {"sat": sat, "cm": cm, "hse": hse}


def spans(flag):
    """bool [T] -> (starts, ends) of its True runs."""
    d = np.diff(np.r_[0, flag.astype(np.int8), 0])
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


# ------------------------------------------------------------------ LFP and MUA readouts

def down_1k(s, fs=HZ):
    """Anti-aliased decimation to 1 kHz (zero-phase FIR, in stages)."""
    s = np.asarray(s, np.float64)
    for q in (10, 3):                         # 30 kHz / 30
        s = sps.decimate(s, q, ftype="fir", zero_phase=True, axis=-1)
    return s


def lfp_band(s_1k, lo, hi, fs=1000.0):
    """Envelope of one LFP band: |Hilbert(bandpass(s))|, on a 1 kHz signal."""
    sos = sps.butter(4, [lo, hi], btype="band", fs=fs, output="sos")
    return np.abs(sps.hilbert(sps.sosfiltfilt(sos, s_1k, axis=-1), axis=-1))


def mua(s, fs=HZ):
    """MUA: bp 500-5000 Hz -> |.| -> lowpass 200 Hz -> average over 30 samples (1 kHz)."""
    s = sps.sosfiltfilt(sps.butter(4, [500, 5000], btype="band", fs=fs, output="sos"), s, axis=-1)
    s = sps.sosfiltfilt(sps.butter(4, 200, fs=fs, output="sos"), np.abs(s), axis=-1)
    n = s.shape[-1] // 30 * 30
    return s[..., :n].reshape(*s.shape[:-1], -1, 30).mean(-1)


def lowpass_fft(s, hi=300.0, fs=HZ):
    """Everything below `hi` (the LFP part for the SNR per band), brick-wall, zero phase."""
    S = np.fft.rfft(s, axis=-1)
    S[..., np.fft.rfftfreq(s.shape[-1], 1 / fs) >= hi] = 0
    return np.fft.irfft(S, n=s.shape[-1], axis=-1)


def pool_1k(t):
    """Differentiable 30 kHz -> 1 kHz for the 1 kHz STFT branch: mean of every 30 samples."""
    return F.avg_pool1d(t.reshape(-1, 1, t.shape[-1]), 30).reshape(*t.shape[:-1], -1)
