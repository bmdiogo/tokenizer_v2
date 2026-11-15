"""Synthetic recordings in the real cache format, so every test runs on a CPU in seconds."""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["WANDB_MODE"] = "disabled"

from utils.cache import artifact_spans  # noqa: E402
from utils.dsp import bp, mad_sigma, robust_stats  # noqa: E402

TEMPLATE = -np.exp(-0.5 * ((np.arange(61) - 21) / 3.0) ** 2) * 60      # a 61-sample trough, in µV


def fake_recording(rng, C=4, T=60_000, n_spikes=60, rail_at=None):
    """int16 counts at 0.25 µV/count: noise + Good-unit spikes at known times (+ an optional rail run)."""
    x = rng.normal(0, 10, (C, T))
    rows, times = [], []
    for c in range(C):
        t = np.sort(rng.choice(np.arange(200, T - 200, 150), n_spikes, replace=False))
        for ti in t:
            x[c, ti - 21:ti + 40] += TEMPLATE
        rows.append(np.full(n_spikes, c)), times.append(t)
    counts = np.rint(x / 0.25).astype(np.int16)
    if rail_at is not None:
        counts[0, rail_at:rail_at + 5] = 32767
    return counts, np.concatenate(rows), np.concatenate(times)


def write_session(d, name, counts, rows, times):
    gain = np.full(counts.shape[0], 0.25, np.float32)
    mu, sigma = robust_stats(counts * gain[:, None])
    sigma_bp = mad_sigma(bp((counts * gain[:, None] - mu[:, None]) / sigma[:, None]))
    art, pct = artifact_spans(counts, gain, mu, sigma, sigma_bp)
    good = np.arange(len(times)) % 5 > 0                        # one spike in five belongs to a Bad unit
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / f"{name}.npy", counts)
    np.savez(d / f"{name}.npz", scale=gain, mu=mu, sigma=sigma, sigma_bp=sigma_bp, band="raw",
             spike_row=rows.astype(np.int16), spike_t=times, spike_unit=np.where(good, rows, rows + 100).astype(np.int16),
             spike_good=good,
             art_row=art[:, 0], art_start=art[:, 1], art_end=art[:, 2], **pct)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def fake_cache(tmp_path, monkeypatch, rng):
    """Two train and one val recording (P2), a splits.csv pointing at them, paths patched."""
    import utils.dataset
    import utils.train
    split = []
    for i, (s, g) in enumerate([("train", ""), ("train", ""), ("val", "p2_late")]):
        name = f"P2_2022_01_0{i + 1}_fake"
        write_session(tmp_path / "cache" / "P2", name, *fake_recording(rng, rail_at=30_000 if i == 2 else None))
        split.append({"patient": "P2", "day": f"2022-01-0{i + 1}", "name": name, "split": s, "group": g, "cached": 1})
    (tmp_path / "info").mkdir()
    pd.DataFrame(split).to_csv(tmp_path / "info" / "splits.csv", index=False)
    monkeypatch.setattr(utils.dataset, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(utils.dataset, "INFO_DIR", tmp_path / "info")
    monkeypatch.setattr(utils.train, "RUNS_DIR", tmp_path / "runs")
    return tmp_path


@pytest.fixture
def small_model_env(monkeypatch):
    """A tiny codec (8 samples per token) and short windows."""
    for k, v in dict(STRIDES="4,2", WIDTHS="4,8,8", DILATIONS="1|1", LATENT_DIM="4", DECODER_MULT="1.0",
                     BURN_IN="64", WINDOW="128", BATCH_SIZE="2", VAL_WINDOWS="4", NUM_WORKERS="0",
                     EVAL_INTERVAL="2", LOG_INTERVAL="1", SAVE_EVERY="2", MONITOR_SECONDS="1",
                     WARMUP_STEPS="1", STFT_WINDOWS_30K="16,32", STFT_WINDOWS_1K="4").items():
        monkeypatch.setenv(k, v)
