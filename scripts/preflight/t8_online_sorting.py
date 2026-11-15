"""T8 · The four numbers of the online sorter, from data (training recordings only, CPU).

Stand-in for ŷ until a trained decoder exists: the ideal y = m·bp(x). One factor at a time from the
defaults (calibration 60 s, 50 spikes, tau = 95th percentile, alpha 0.01). Recordings last ~2-5 min,
so calibration is swept in seconds (30 / 60 / 90), not in minutes as first planned.
Rule: the shortest calibration / simplest setting within the T7 floor (~0.03) of the best.

    python -m scripts.preflight.t8_online_sorting --max-sessions 20
    python -m scripts.preflight.t8_online_sorting --checkpoint runs/s5-c8/step_015000.pt   # a trained ŷ -> t8_yhat.json
"""
import argparse

import numpy as np

from scripts.preflight import save
from utils import sorting
from utils.config import HZ
from utils.dataset import sessions
from utils.dsp import bp

DEFAULT = dict(cal_seconds=60, min_spikes=50, tau_pct=95, alpha=0.01)   # the T8 starting point (the sweep is run from here)
SWEEP = {"cal_seconds": [30, 60, 90], "min_spikes": [20, 50, 100], "tau_pct": [90, 95, 99], "alpha": [0, 0.001, 0.01]}


def score(ideal, setting):
    acc, unsorted, total = [], 0, 0
    cal = int(setting["cal_seconds"] * HZ)
    for s, y, T in ideal:
        for r in np.unique(s.spike_row):
            k = (s.spike_row == r) & (s.spike_t >= cal) & (s.spike_t < T)
            if not k.any():
                continue
            t, lab = sorting.online(y[r], s.sigma_bp[r], **setting)
            o = np.argsort(s.spike_t[k])
            acc += list(sorting.unit_accuracy(t, lab, s.spike_t[k][o], s.spike_unit[k][o]).values())
            unsorted, total = unsorted + int((lab == sorting.UNSORTED).sum()), total + len(lab)
    return {"acc_median": float(np.median(acc)) if acc else None, "n_units": len(acc),
            "pct_unsorted": 100 * unsorted / max(total, 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-sessions", type=int, default=20)
    ap.add_argument("--seconds", type=float, default=300)
    ap.add_argument("--checkpoint", default=None, help="score a trained ŷ instead of the ideal y")
    a = ap.parse_args()
    model = None
    if a.checkpoint:
        import torch

        from utils.train import load_checkpoint
        model, _ = load_checkpoint(a.checkpoint, ema=True, device="cuda" if torch.cuda.is_available() else "cpu")
    ss = [s for s in sessions("train") if s.T >= 150 * HZ and len(s.spike_t)][:a.max_sessions]
    ideal = []
    for s in ss:
        T = min(s.T, int(a.seconds * HZ))
        if model is None:
            ideal.append((s, s.m(0, T) * bp(s.x(0, T)), T))
        else:
            y_hat = model.reconstruct(s.x(0, T))[1]                    # T is cut to whole tokens
            ideal.append((s, y_hat, y_hat.shape[1]))
    results = {"default": score(ideal, DEFAULT)}
    print("default", results["default"], flush=True)
    for key, values in SWEEP.items():
        for v in values:
            if v != DEFAULT[key]:
                results[f"{key}={v}"] = score(ideal, {**DEFAULT, key: v})
                print(f"{key}={v}", results[f"{key}={v}"], flush=True)
    save("t8_yhat" if model is not None else "t8", {"question": "online sorter settings (on " + (a.checkpoint or "the ideal y") + ")",
                "sessions": [s.name for s in ss],
                "default": DEFAULT, "results": results})


if __name__ == "__main__":
    main()
