"""Sums of every training frame, for the PCA fitted on all of them (`pcafull` and `pcawfull-<w>`).

    python -m scripts.pca_stats --frame 240     # 480 = 62.5 Hz tokens, 960 = 31.3 Hz

Writes <RUNS_DIR>/pca_stats_F<frame>.npz. P4 is never read.
"""
import argparse

import numpy as np

from models.baselines import frame_stats
from utils.config import RUNS_DIR, TRAIN_PATIENTS, apply_overrides
from utils.dataset import sessions
from utils.train import check_no_held_out

if __name__ == "__main__":
    rest = apply_overrides()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frame", type=int, default=240)
    a = ap.parse_args(rest)
    check_no_held_out()
    stats = frame_stats(sessions("train", patients=TRAIN_PATIENTS), a.frame)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(RUNS_DIR / f"pca_stats_F{a.frame}.npz", **stats)
    print(f"RESULT frames {stats['n']:.0f} with a spike {stats['n_spk']:.0f}")
