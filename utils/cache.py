"""NS5 + NIX -> cache, one pair of files per recording (every `cached=1` row of info/splits.csv):

    <CACHE_DIR>/<patient>/<name>.npy   int16 [96, T]   the recording (raw NS5 counts by default)
    <CACHE_DIR>/<patient>/<name>.npz   scale [96] (µV per count) · mu, sigma [96] (median / MAD·1.4826
                                       of the first 60 s, µV) · sigma_bp [96] (MAD of bp(x_n)) ·
                                       NIX spikes of every unit (spike_row, spike_t, spike_unit, and
                                       spike_good = its Label is Good) · artifact spans
                                       (art_row, art_start, art_end; row -1 = every channel) · pct_*

Only NIX spike TIMES are kept (T1): the NIX waveforms are whitened and never a target. Good units
are the ground truth (m, y, every metric); Bad units are kept for analyses of the rest of the sort.

    python -m utils.cache --patients P2 P3 --workers 8
    python -m utils.cache --report            # % time flagged per artifact rule, per session
"""
import argparse
import json
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import signal as sps

from utils.config import CACHE_DIR, DATA_PATH, HZ, INFO_DIR, ROOT, env
from utils.dsp import (
    ART_AFTER,
    ART_BEFORE,
    artifact_flags,
    bp,
    mad_sigma,
    robust_stats,
    spans,
)

CHUNK, PAD = 10 * HZ, HZ // 10        # artifacts: 10 s chunks, 100 ms of filter context on each side


def electrode_map(patient):
    """(NS5 rows of the array, {electrode id: cached row}). Unresolved electrodes are absent."""
    raw = json.loads((INFO_DIR / f"electrode_to_row_{patient}.json").read_text())
    rows = [int(r) for r in raw["rows"]]
    return rows, {int(e): rows.index(int(r)) for e, r in raw["electrode_to_row"].items() if int(r) in rows}


def load_ns5(path, rows):
    """-> (int16 counts [len(rows), T], µV per count [len(rows)])."""
    import neo
    reader = neo.io.BlackrockIO(str(path))
    reader.parse_header()
    counts = reader.get_analogsignal_chunk(0, 0, stream_index=0, channel_indexes=rows)
    chans = reader.header["signal_channels"]
    chans = chans[chans["stream_id"] == reader.header["signal_streams"]["id"][0]]
    return np.ascontiguousarray(counts.T), chans["gain"][rows].astype(np.float32)


def nix_spikes(nix_path, eid2row):
    """(row, sample, unit index, is Good) of every spike of every NIX unit on a resolved electrode.
    NIX labels each unit Good or Bad; the unit index is the train's position in the file."""
    import neo
    with neo.NixIO(str(nix_path), "ro") as nio:
        trains = nio.read_block().segments[0].spiketrains
    rows, times, units, good = [], [], [], []
    for u, st in enumerate(trains):
        eid = st.annotations.get("new_electrode_id") or st.annotations.get("channel_id")
        if eid is None or int(eid) not in eid2row:
            continue                                   # a missing row is skipped; a wrong one would corrupt m
        t = np.rint(st.times.rescale("s").magnitude * HZ).astype(np.int64)
        rows.append(np.full(t.size, eid2row[int(eid)])), times.append(t), units.append(np.full(t.size, u))
        good.append(np.full(t.size, st.annotations.get("Label") == "Good"))
    cat = lambda v, d: np.concatenate(v).astype(d) if v else np.zeros(0, d)
    return cat(rows, np.int16), cat(times, np.int64), cat(units, np.int16), cat(good, bool)


def union_length(starts, ends):
    order = np.argsort(starts)
    total, reach = 0, -1
    for s, e in zip(starts[order], ends[order]):
        total += max(0, e - max(s, reach))
        reach = max(reach, e)
    return total


def artifact_spans(counts, gain, mu, sigma, sigma_bp):
    """The artifact rules (`dsp.artifact_flags`) chunk by chunk over the whole recording -> dilated spans, % flagged."""
    C, T = counts.shape
    found = {"sat": [], "cm": [], "hse": []}                   # (row, start, end)
    for a in range(0, T, CHUNK):
        lo, hi = max(a - PAD, 0), min(a + CHUNK + PAD, T)
        c = counts[:, lo:hi]
        x_n = (c * gain[:, None] - mu[:, None]) / sigma[:, None]
        flags = artifact_flags(c, x_n, bp(x_n), sigma_bp)
        for rule, f in flags.items():
            f = f[..., a - lo:a - lo + min(CHUNK, T - a)]
            per_channel = f.ndim == 2                          # sat is per channel; cm and hse flag them all
            for r, row in enumerate(f if per_channel else [f]):
                s, e = spans(row)
                found[rule] += [(r if per_channel else -1, a + si, a + ei) for si, ei in zip(s, e)]
    out, pct = [], {}
    for rule, sp in found.items():                             # pad onset and recovery tail
        sp = np.array(sp, np.int64).reshape(-1, 3)
        sp[:, 1] = np.maximum(sp[:, 1] - int(ART_BEFORE * HZ), 0)
        sp[:, 2] = np.minimum(sp[:, 2] + int(ART_AFTER * HZ), T)
        per_row = [union_length(*sp[(sp[:, 0] == r) | (sp[:, 0] == -1)][:, 1:].T) for r in range(C)]
        pct[f"pct_{rule}"] = 100 * np.mean(per_row) / T
        out.append(sp)
    sp = np.concatenate(out)
    pct["pct_any"] = 100 * np.mean([union_length(*sp[(sp[:, 0] == r) | (sp[:, 0] == -1)][:, 1:].T)
                                    for r in range(C)]) / T
    return sp, pct


def build(row):
    """Cache one recording (skipped if both files exist). Returns its name."""
    out = CACHE_DIR / row["patient"]
    if (out / f"{row['name']}.npz").exists() and (out / f"{row['name']}.npy").exists():
        return row["name"]
    rows, eid2row = electrode_map(row["patient"])
    counts, gain = load_ns5(DATA_PATH / row["signal"], rows)
    n60 = min(counts.shape[1], int(env("NORM_SECONDS", 60) * HZ))
    raw_mu, raw_sigma = robust_stats(counts[:, :n60] * gain[:, None])
    raw_sigma_bp = mad_sigma(bp((counts[:, :n60] * gain[:, None] - raw_mu[:, None]) / raw_sigma[:, None]))
    art, pct = artifact_spans(counts, gain, raw_mu, raw_sigma, raw_sigma_bp)   # always on the raw recording
    band = env("INPUT_BAND", "raw")
    if band == "raw":
        stored, scale, mu, sigma, sigma_bp = counts, gain, raw_mu, raw_sigma, raw_sigma_bp
    else:                                                  # causal band-pass, requantised per channel
        lo, hi = map(float, band.split(","))
        sos = sps.butter(4, [lo, hi], btype="band", fs=HZ, output="sos")
        x = np.stack([sps.sosfilt(sos, c * g).astype(np.float32) for c, g in zip(counts, gain)])
        scale = np.maximum(np.abs(x).max(1) / 32767, 1e-6).astype(np.float32)
        stored = np.rint(x / scale[:, None]).astype(np.int16)
        mu, sigma = robust_stats(x[:, :n60])
        sigma_bp = mad_sigma(bp((x[:, :n60] - mu[:, None]) / sigma[:, None]))
    s_row, s_t, s_unit, s_good = nix_spikes(DATA_PATH / row["spikes"], eid2row)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"{row['name']}.tmp.npy", stored)
    np.savez(out / f"{row['name']}.tmp.npz", scale=scale, mu=mu, sigma=sigma, sigma_bp=sigma_bp, band=band,
             spike_row=s_row, spike_t=s_t, spike_unit=s_unit, spike_good=s_good,
             art_row=art[:, 0], art_start=art[:, 1], art_end=art[:, 2], **pct)
    (out / f"{row['name']}.tmp.npy").replace(out / f"{row['name']}.npy")     # the .npz last: it marks "done"
    (out / f"{row['name']}.tmp.npz").replace(out / f"{row['name']}.npz")
    return row["name"]


def report():
    """% of time flagged per rule, Good and Bad units per recording -> experiments/cache_report.csv."""
    df = pd.read_csv(INFO_DIR / "splits.csv")
    rows = []
    for _, r in df[df["cached"] == 1].iterrows():
        p = CACHE_DIR / r["patient"] / f"{r['name']}.npz"
        if p.exists():
            z = np.load(p)
            rows.append({"name": r["name"], "patient": r["patient"], "split": r["split"], "group": r["group"],
                         "seconds": np.load(p.with_suffix(".npy"), mmap_mode="r").shape[1] / HZ,
                         "good_units": len(np.unique(z["spike_unit"][z["spike_good"]])),
                         "good_spikes": int(z["spike_good"].sum()),
                         "bad_units": len(np.unique(z["spike_unit"][~z["spike_good"]])),
                         "bad_spikes": int((~z["spike_good"]).sum()),
                         **{k: float(z[k]) for k in z.files if k.startswith("pct_")}})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "experiments" / "cache_report.csv", index=False)
    print(out.groupby(["split", "group"], dropna=False)[[c for c in out if c.startswith("pct_")]]
          .describe().loc[:, (slice(None), ["50%", "max"])].round(3).to_string())
    print(f"{len(out)} / {int((df['cached'] == 1).sum())} cached recordings present")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patients", nargs="+", default=["P2", "P3", "P4"])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        return report()
    df = pd.read_csv(INFO_DIR / "splits.csv")
    todo = [r for _, r in df[(df["cached"] == 1) & df["patient"].isin(a.patients)].iterrows()]
    print(f"caching {len(todo)} recordings of {a.patients} into {CACHE_DIR} (band {env('INPUT_BAND', 'raw')})")
    with Pool(a.workers) as pool:
        for i, name in enumerate(pool.imap_unordered(build, todo), 1):
            print(f"[{i}/{len(todo)}] {name}", flush=True)


if __name__ == "__main__":
    main()
