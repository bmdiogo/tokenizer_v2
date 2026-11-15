"""T1 · Which signal space the NIX waveforms live in, and where the NIX time sits (CPU).

For the Good units of a few P3 training recordings: NIX snippet vs bp(x) (µV) at the NIX time,
shifted by -15..+15 samples. Reports the best lag, r at that lag, the amplitude ratio NIX / bp(x),
and the share of spikes that overlap a neighbour on the same electrode (< 61 samples apart).
Expected (old T1): r = 1.000, amplitude 0.17-0.25x (NIX is whitened), NIX time = sample 21.

    python -m scripts.preflight.t1_nix_space --n 3
"""
import argparse

import neo
import numpy as np
import pandas as pd

from scripts.preflight import save
from utils.cache import electrode_map
from utils.config import DATA_PATH, HZ, INFO_DIR
from utils.dataset import sessions
from utils.dsp import WF_POST, WF_PRE, bp


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--seconds", type=float, default=120)
    a = ap.parse_args()
    manifest = pd.read_csv(INFO_DIR / "splits.csv").set_index("name")
    lags, rs, ratios, overlap, per_unit = [], [], [], [], []
    for s in sessions("train", patients="P3")[:a.n]:
        n = min(s.T, int(a.seconds * HZ))
        x_uv = np.asarray(s.counts[:, :n], np.float32) * s.scale[:, None]
        b = bp(x_uv - np.median(x_uv, 1, keepdims=True))
        _, eid2row = electrode_map("P3")
        with neo.NixIO(str(DATA_PATH / manifest.loc[s.name, "spikes"]), "ro") as nio:
            trains = [t for t in nio.read_block().segments[0].spiketrains if t.annotations.get("Label") == "Good"]
        for st in trains:
            eid = st.annotations.get("new_electrode_id") or st.annotations.get("channel_id")
            if eid is None or int(eid) not in eid2row:
                continue
            r = eid2row[int(eid)]
            wf = st.waveforms.magnitude.astype(np.float32)
            wf = wf[:, 0] if wf.ndim == 3 else wf
            t = np.rint(st.times.rescale("s").magnitude * HZ).astype(np.int64)
            ok = (t >= WF_PRE + 15) & (t + WF_POST + 15 < n)
            t, wf = t[ok], wf[ok][:, :WF_PRE + WF_POST]
            if len(t) < 20:
                continue
            best = max(range(-15, 16), key=lambda L: np.corrcoef(
                wf.mean(0), b[r][(t + L)[:, None] + np.arange(-WF_PRE, WF_POST)].mean(0))[0, 1])
            seg = b[r][(t + best)[:, None] + np.arange(-WF_PRE, WF_POST)]
            rr = np.corrcoef(wf.mean(0), seg.mean(0))[0, 1]
            ratio = np.ptp(wf.mean(0)) / np.ptp(seg.mean(0))
            lags.append(best), rs.append(rr), ratios.append(ratio)
            overlap.append(np.mean(np.diff(np.sort(s.spike_t[s.spike_row == r])) < WF_PRE + WF_POST))
            per_unit.append({"session": s.name, "row": r, "n": len(t), "lag": best, "r": rr, "ratio": ratio})
    save("t1", {"question": "NIX waveform space and time offset", "n_units": len(rs),
                "lag_median": float(np.median(lags)), "r_median": float(np.median(rs)),
                "amp_ratio_nix_over_bp": [float(np.percentile(ratios, 10)), float(np.median(ratios)), float(np.percentile(ratios, 90))],
                "pct_spikes_overlapping": 100 * float(np.mean(overlap)), "units": per_unit})


if __name__ == "__main__":
    main()
