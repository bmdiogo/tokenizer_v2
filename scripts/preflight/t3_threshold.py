"""T3 · The reference detector: F1 of -K·MAD(bp(x)) crossings, K swept, on 3 P3 TRAINING recordings
(never val / test). Recall vs NIX Good, precision vs NIX Good + Bad (S9). The K with the best pooled
F1 goes to K_THRESHOLD in .env.

    python -m scripts.preflight.t3_threshold
"""
import numpy as np

from scripts.preflight import save
from utils.config import HZ, env
from utils.dataset import sessions
from utils.dsp import bp
from utils.metrics import COUNTS, detect, prf

KS = [3.5, 4.0, 4.5, 5.0, 5.5, 6.0]


def main():
    ss = [s for s in sessions("train", patients="P3") if len(s.spike_t)][:3]
    counts = {k: np.zeros(len(COUNTS)) for k in KS}
    for s in ss:
        n = min(s.T, int(env("EVAL_SECONDS", 120) * HZ))
        b = bp(s.x(0, n))
        art = s.art(0, n)
        for r in np.unique(s.spike_row):
            ref = np.sort(s.spike_t[(s.spike_row == r) & (s.spike_t < n)])
            bad = np.sort(s.bad_t[(s.bad_row == r) & (s.bad_t < n)])
            for k in KS:
                counts[k] += detect(b[r], s.sigma_bp[r], ref, art[r], k=k, bad=bad)
    table = {str(k): {**prf(*c), **{name: int(v) for name, v in zip(COUNTS, c)}} for k, c in counts.items()}
    best = max(KS, key=lambda k: table[str(k)]["f1"])
    for k in KS:
        print(f"K {k}: " + "  ".join(f"{m} {table[str(k)][m]:.3f}" for m in ("precision", "recall", "f1")))
    save("t3", {"question": "reference detector", "sessions": [s.name for s in ss], "by_k": table, "best_k": best})


if __name__ == "__main__":
    main()
