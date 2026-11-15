"""Cached recordings -> model inputs. Split and group come from info/splits.csv, nothing else.

A training item is one window of W = B + L samples on all 96 electrodes:
    x    [96, B + L]  normalised recording (x - mu) / sigma; the model burns in on the first B
                      (+ F future samples for the non-causal ceiling, zeros past the recording's end)
    m    [96, L]      spike mask on the loss span: tukey windows at NIX Good times, 0 on artifacts
    art  [96, L]      artifact flag (reported, never a loss mask)
    good [96, L]      1 at each NIX Good spike time (in-training detection: recall and precision)
    bad  [96, L]      1 at each NIX Bad spike time (precision only: a detection on it is not false)
"""
from itertools import count

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from utils.config import CACHE_DIR, INFO_DIR, env
from utils.dsp import bp, spike_mask
from utils.metrics import detections

CROSS_PAD = 1024                  # context on each side of a window for bp() before the crossings (S7-D)
CROSS_SHIFT = 2                   # crossings come 2 samples before the NIX time (T3)


class Session:
    """One cached recording, memory-mapped."""

    def __init__(self, row, cache_dir=None):
        d = (cache_dir or CACHE_DIR) / row["patient"]
        self.name, self.patient, self.split, self.group = row["name"], row["patient"], row["split"], row["group"]
        self.counts = np.load(d / f"{self.name}.npy", mmap_mode="r")
        z = np.load(d / f"{self.name}.npz")
        self.scale, self.mu, self.sigma, self.sigma_bp = (
            z[k].astype(np.float32) for k in ("scale", "mu", "sigma", "sigma_bp"))
        good = z["spike_good"]                        # Good units are the ground truth; Bad ones are kept apart
        row, t, unit = z["spike_row"].astype(np.int64), z["spike_t"], z["spike_unit"]
        self.spike_row, self.spike_t, self.spike_unit = row[good], t[good], unit[good]
        self.bad_row, self.bad_t, self.bad_unit = row[~good], t[~good], unit[~good]
        self.art_spans = np.stack([z["art_row"], z["art_start"], z["art_end"]], 1)
        self.C, self.T = self.counts.shape

    def x(self, t0, n):
        c = np.asarray(self.counts[:, t0:t0 + n], np.float32)
        return (c * self.scale[:, None] - self.mu[:, None]) / self.sigma[:, None]

    def art(self, t0, n):
        a = np.zeros((self.C, n), np.float32)
        for r, s, e in self.art_spans[(self.art_spans[:, 2] > t0) & (self.art_spans[:, 1] < t0 + n)]:
            a[slice(None) if r < 0 else r, max(s - t0, 0):min(e - t0, n)] = 1
        return a

    def crossing_times(self, t0, n):
        """S7-D third arm: -K·sigma crossings on bp(x), no labels, in place of the NIX Good times."""
        a, e = max(t0 - CROSS_PAD, 0), min(t0 + n + CROSS_PAD, self.T)
        s = bp(self.x(a, e - a))
        times = [detections(s[r], self.sigma_bp[r]) + a + CROSS_SHIFT for r in range(self.C)]
        return np.repeat(np.arange(self.C), [len(t) for t in times]), np.concatenate(times)

    def m(self, t0, n):
        """MASK_SOURCE: nix = NIX Good times (default) · crossings = S7-D; train() sets it from the run's LossConfig."""
        if env("MASK_SOURCE", "nix") == "nix":
            rows, times = self.spike_row, self.spike_t
        else:
            rows, times = self.crossing_times(t0, n)
        return spike_mask(rows, times, self.C, t0, n) * (1 - self.art(t0, n))

    def impulses(self, rows, times, t0, n):
        g = np.zeros((self.C, n), np.float32)
        k = (times >= t0) & (times < t0 + n)
        g[rows[k], times[k] - t0] = 1
        return g

    def item(self, t0, burn_in, window, future=0):
        t, x = t0 + burn_in, self.x(t0, burn_in + window + future)
        x = np.pad(x, ((0, 0), (0, burn_in + window + future - x.shape[1])))
        return {"x": x, "m": self.m(t, window), "art": self.art(t, window),
                "good": self.impulses(self.spike_row, self.spike_t, t, window),
                "bad": self.impulses(self.bad_row, self.bad_t, t, window), "sigma_bp": self.sigma_bp}


def sessions(split=None, groups=None, patients=None, cache_dir=None):
    """The cached recordings of a selection. Raises if one of them is missing from the cache."""
    table = INFO_DIR / "splits.csv"
    if not table.exists():
        raise FileNotFoundError(f"{table} is missing: it names clinical recordings and is not distributed; "
                                "build it from your data with `python -m utils.split`")
    df = pd.read_csv(table, keep_default_na=False)
    df = df[df["cached"] == 1]
    for col, want in (("split", split), ("group", groups), ("patient", patients)):
        if want is not None:
            df = df[df[col].isin([want] if isinstance(want, str) else want)]
    missing = [n for n, p in zip(df["name"], df["patient"]) if not ((cache_dir or CACHE_DIR) / p / f"{n}.npz").exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} recordings not cached, e.g. {missing[:3]}: "
                                "run `python -m utils.cache`")
    return [Session(r, cache_dir) for _, r in df.iterrows()]


class Windows(Dataset):
    """Training windows. Item i is drawn from rng([seed, i]): session in proportion to its length,
    start uniform. The same i always gives the same window, so a resumed run continues the exact
    data stream. With `fixed` [(session, t0)], items are those fixed windows instead (validation)."""

    def __init__(self, sessions, burn_in, window, seed=0, fixed=None, future=0):
        self.s, self.b, self.w, self.seed, self.fixed, self.f = sessions, burn_in, window, seed, fixed, future
        self.p = np.array([s.T for s in sessions], float) / sum(s.T for s in sessions)

    def __len__(self):
        return len(self.fixed) if self.fixed is not None else 2 ** 40

    def __getitem__(self, i):
        if self.fixed is not None:
            s, t0 = self.fixed[i]
        else:
            rng = np.random.default_rng([self.seed, i])
            s = self.s[rng.choice(len(self.s), p=self.p)]
            t0 = int(rng.integers(0, s.T - self.b - self.w))
        return {k: torch.from_numpy(v) for k, v in s.item(t0, self.b, self.w, self.f).items()}   # same t0 for any F


class CountFrom(Sampler):
    """0, 1, 2, ... starting at `start` (a resumed run starts at step x batch)."""

    def __init__(self, start):
        self.start = start

    def __iter__(self):
        return count(self.start)


def train_loader(sessions, burn_in, window, batch_size, seed, start_step=0, num_workers=4, future=0):
    kw = dict(persistent_workers=True, prefetch_factor=4) if num_workers else {}
    return DataLoader(Windows(sessions, burn_in, window, seed, future=future), batch_size,
                      sampler=CountFrom(start_step * batch_size), num_workers=num_workers, **kw)


def fixed_windows(sessions, burn_in, window, n):
    """n windows evenly spread over every session (the same ones at every evaluation)."""
    W = burn_in + window
    all_ = [(s, t0) for s in sessions for t0 in range(0, s.T - W, W)]
    return [all_[i] for i in np.linspace(0, len(all_) - 1, min(n, len(all_))).astype(int)]


def val_loader(sessions, burn_in, window, batch_size, n_windows, future=0):
    return DataLoader(Windows(sessions, burn_in, window, fixed=fixed_windows(sessions, burn_in, window, n_windows),
                              future=future),
                      batch_size, shuffle=False)
