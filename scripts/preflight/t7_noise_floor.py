"""T7 · Noise floor over seeds: score 5 runs of ONE configuration at the SAME steps.

Train first (5 jobs):
    for k in 1 2 3 4 5; do EXP=t7-s$k EXTRA="SEED=$k DATA_SEED=$((10 + k)) MAX_STEPS=15000 WANDB_GROUP=t7" cluster/submit.sh; done
Then score them (one tool job):
    python -m scripts.preflight.t7_noise_floor --runs t7-s1 t7-s2 t7-s3 t7-s4 t7-s5

Floor = the widest same-step spread (max - min over seeds) of each metric. Every run is still rising at
15 k steps, so the spread across checkpoints of one run is a trend, not noise (as in the old T7).
"""
import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from scripts.preflight import save
from utils.config import ROOT, RUNS_DIR, VAL_GROUPS

METRICS = {"f1_x_hat": ("detect_x_hat", "f1"), "recall_x_hat": ("detect_x_hat", "recall"),
           "f1_y_hat": ("detect_y_hat", "f1"),
           "retention_y_hat": ("retention_y_hat", "retention"), "retention_x_hat": ("retention_x_hat", "retention"),
           "wave_r": ("waveform", "all", "r_median"),
           "sorting_acc": ("sorting_x_hat", "acc_median"), "mua_r": ("mua_r_median",), "probe_f1": ("probe_f1",)}


def get(d, path):
    for k in path:
        d = d.get(k) if isinstance(d, dict) else None
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--steps", nargs="+", type=int, default=[5000, 10000, 15000])
    ap.add_argument("--jobs", type=int, default=1, help="evaluations run in parallel")
    ap.add_argument("--name", default="t7", help="output experiments/<name>.json (t7 = the floor compare.py reads)")
    a = ap.parse_args()
    path = lambda run, step, g: ROOT / "experiments" / "eval" / f"t7_{run}_{step}_{g}.json"

    def evaluate(cell):
        run, step, g = cell
        if not path(*cell).exists():
            subprocess.run([sys.executable, "-m", "scripts.evaluate", str(RUNS_DIR / run / f"step_{step:06d}.pt"),
                            "--group", g, "--out", str(path(*cell))], cwd=ROOT, check=True)

    with ThreadPoolExecutor(a.jobs) as pool:
        list(pool.map(evaluate, [(r, s, g) for g in VAL_GROUPS for r in a.runs for s in a.steps]))
    summary = {}
    for g in VAL_GROUPS:
        scores, ceiling = {}, None
        for run in a.runs:
            for step in a.steps:
                d = json.loads(path(run, step, g).read_text())
                scores[run, step] = {m: get(d, p) for m, p in METRICS.items()}
                ceiling = d["detect_x"]["f1"]
        summary[g] = {"f1_ceiling": ceiling}
        for m in METRICS:
            by_step = {s: [scores[r, s][m] for r in a.runs if scores[r, s][m] is not None] for s in a.steps}
            last = by_step[a.steps[-1]]
            summary[g][m] = {"mean": float(np.mean(last)), "sd": float(np.std(last, ddof=1)) if len(last) > 1 else None,
                             "floor": float(max(np.ptp(v) for v in by_step.values() if v)), "by_step": by_step}
            print(f"{g} {m}: {summary[g][m]['mean']:.3f} +- {summary[g][m]['sd'] or 0:.3f}  floor {summary[g][m]['floor']:.3f}")
    save(a.name, {"question": "noise floor over seeds", "runs": a.runs, "steps": a.steps, "groups": summary})


if __name__ == "__main__":
    main()
