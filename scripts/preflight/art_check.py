"""Artifact-rule check: does the hyper-synchronous (hse) rule flag real neural activity?

A loose hse rule removes real synchrony. Test on the most-flagged recordings: if
NIX Good spikes fall inside flagged spans much more often than the flagged share of time, the rule is
removing bursts of real spikes, not artifacts. Also plots bp(x) around a few flagged spans to look at.

    python -m scripts.preflight.art_check --names <recording> ...      # train / val recordings only
"""
import argparse

import numpy as np

from scripts.preflight import save
from utils.config import HZ, ROOT
from utils.dataset import sessions
from utils.dsp import bp


def flagged_mask(s):
    """bool [C, T]: flagged by any rule (spans are per channel, or -1 = all channels)."""
    f = np.zeros((s.C, s.T), bool)
    for r, a, b in s.art_spans:
        f[slice(None) if r < 0 else r, a:b] = True
    return f


def check(s, n_plots=4, half=0.05):
    f = flagged_mask(s)
    good_in = f[s.spike_row, s.spike_t].mean() if len(s.spike_t) else None
    bad_in = f[s.bad_row, s.bad_t].mean() if len(s.bad_t) else None
    out = {"pct_time_flagged": 100 * float(f.mean()),
           "pct_good_spikes_inside": None if good_in is None else 100 * float(good_in),
           "pct_bad_spikes_inside": None if bad_in is None else 100 * float(bad_in),
           "n_spans": int(len(s.art_spans))}
    out["spike_enrichment"] = (None if good_in is None or not f.mean() else float(good_in / f.mean()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(0)
    pick = s.art_spans[rng.choice(len(s.art_spans), min(n_plots, len(s.art_spans)), replace=False)] if len(s.art_spans) else []
    fig, axes = plt.subplots(1, max(len(pick), 1), figsize=(4 * max(len(pick), 1), 6), squeeze=False)
    h = int(half * HZ)
    for ax, (r, a, b) in zip(axes[0], pick):
        t0 = max(a - h, 0)
        seg = bp(s.x(t0, min(b + h, s.T) - t0)) / s.sigma_bp[:, None]
        ax.imshow(np.clip(seg, -8, 8), aspect="auto", cmap="RdBu_r", vmin=-8, vmax=8,
                  extent=[(t0 - a) / HZ * 1e3, (t0 + seg.shape[1] - a) / HZ * 1e3, s.C, 0])
        ax.axvspan(0, (b - a) / HZ * 1e3, color="k", alpha=0.1)
        k = (s.spike_t >= t0) & (s.spike_t < t0 + seg.shape[1])
        ax.scatter((s.spike_t[k] - a) / HZ * 1e3, s.spike_row[k] + 0.5, s=4, c="k", label="NIX Good")
        ax.set_xlabel("ms from flag onset")
        ax.set_title(f"span {a / HZ:.1f} s, row {r}")
    axes[0][0].set_ylabel("electrode (bp(x) / sigma, clipped at ±8)")
    fig.suptitle(f"{s.name}: {out['pct_time_flagged']:.1f} % flagged; Good spikes inside {out['pct_good_spikes_inside']}")
    png = ROOT / "experiments" / f"art_check_{s.name}.png"
    fig.savefig(png, dpi=110, bbox_inches="tight")
    out["figure"] = png.name
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--names", nargs="+", required=True)
    a = ap.parse_args()
    res = {}
    for s in sessions(split=["train", "val"]):          # never a test recording: a rule tuned on test is tuning on test
        if s.name in a.names:
            res[s.name] = check(s)
            print(s.name, res[s.name], flush=True)
    save("art_check", {"question": "does the hse rule flag real spikes?", "recordings": res})


if __name__ == "__main__":
    main()
