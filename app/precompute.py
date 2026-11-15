"""Precompute a replay stream without the UI: the part of the app that runs in the project container.

    python -m app.precompute <recording> --electrodes 96                # CPU
    bash cluster/container/run_local.sh python -m app.precompute <recording> --cache /data/cache ...   # Docker, CPU
    CMD='python -m app.precompute <recording> --device cuda' sbatch cluster/tool.slurm     # cluster GPU

Writes <out>/<recording>_<n>el.npz (ŷ and z of the chosen electrodes, compute time per frame), which the app plays
at 1× in Precomputed mode. Same causal stream as Live mode (`app.engine.precompute`). Needs only torch and numpy.
"""
import argparse
import time
from pathlib import Path

import torch

from app.engine import pick_rows, precompute
from utils.config import ROOT, env


def path_env(key, default):
    p = Path(env(key, default)).expanduser()
    return p if p.is_absolute() else ROOT / p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording")
    ap.add_argument("--electrodes", type=int, default=int(env("APP_ELECTRODES", 96)))
    ap.add_argument("--cache", type=Path, default=None, help="default APP_CACHE_DIR (<patient>/<recording>.npy|.npz)")
    ap.add_argument("--checkpoint", type=Path, default=None, help="default APP_CHECKPOINT")
    ap.add_argument("--out", type=Path, default=None, help="default <cache>/../precomputed (where the app looks)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    cache = a.cache or path_env("APP_CACHE_DIR", "data/app/cache")
    ckpt = a.checkpoint or path_env("APP_CHECKPOINT", "data/app/runs/s8-ref-s1/step_200000.pt")
    out = a.out or cache.parent / "precomputed"
    out.mkdir(parents=True, exist_ok=True)
    rows, _ = pick_rows(cache, a.recording, a.electrodes)
    dst = out / f"{a.recording}_{a.electrodes}el.npz"
    t0, last = time.time(), [0.0]

    def say(f):
        if f - last[0] >= 0.1 or f >= 1:
            print(f"{100 * f:5.1f} %  {time.time() - t0:6.0f} s", flush=True)
            last[0] = f
    print(f"{a.recording}: {len(rows)} electrodes on {a.device} -> {dst}", flush=True)
    precompute(cache, a.recording, ckpt, rows, dst, progress=say, device=a.device)
    print(f"done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
