"""S11: the arrays the paper figures need that the eval JSONs do not keep .

    python -m scripts.figure_data all              # every part, test groups, frozen checkpoints (one GPU job)
    python -m scripts.figure_data streaming psd    # some parts
    python -m scripts.figure_data all --quick      # smoke test: one recording, 20 s

Never re-scores a headline metric (those are S10's, in experiments/eval/). Writes small arrays to
experiments/figures/data/<part>.npz and a json beside it. Reads only the frozen checkpoints of s10_frozen.json.
`threshold` also reads the validation groups: it picks ŷ's operating point there before reading the test groups.
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import scipy.signal as sps
import torch

from scripts.evaluate import codec_fn, sort_scores, unit_recall
from utils import dataset
from utils.config import (
    HZ,
    ROOT,
    RUNS_DIR,
    TEST_GROUPS,
    VAL_GROUPS,
    env,
    git_hash,
    split_hash,
)
from utils.dataset import sessions
from utils.dsp import LFP_BANDS, WF_POST, WF_PRE, bp, down_1k, lfp_band, mua
from utils.metrics import detect, detections, found_on_x, match, pearson, prf
from utils.metrics import (
    waveforms as unit_waveforms,  # `waveforms` below is the figure part
)
from utils.sorting import cluster, online, snippets

OUT = ROOT / "experiments" / "figures" / "data"
CKPT = {"125": "s8-ref-s{}", "62.5": "s8-s6-625hz-lr1-s{}", "31.3": "s8-s6-313hz-lr1-s{}"}
DEV = "cuda" if torch.cuda.is_available() else "cpu"
Q = {"quick": False}


def ckpt(rate, seed=1):
    return str(RUNS_DIR / CKPT[rate].format(seed) / "step_200000.pt")


def seconds():
    return 20 if Q["quick"] else 120


def group_sessions(g):
    s = sessions(groups=g)
    return s[:1] if Q["quick"] else s


def load(s, n=None):
    n = min(s.T, int((n or seconds()) * HZ)) // 960 * 960          # a multiple of every rate's frame
    return s.x(0, n)


def save(part, arrays, meta):
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{part}.npz", **arrays)
    (OUT / f"{part}.json").write_text(json.dumps({"git": git_hash(), "split_hash": split_hash(), "quick": Q["quick"],
                                                  **meta}, indent=1, default=float))
    print(f"wrote {OUT / part}.npz", flush=True)


def example_session():
    """The fixed example: the p4_unseen recording with the most Good spikes, and its 4 busiest electrodes."""
    ss = group_sessions("p4_unseen")
    s = max(ss, key=lambda s: len(s.spike_t))
    rows, counts = np.unique(s.spike_row[s.spike_t < load(s).shape[1]], return_counts=True)
    return s, rows[np.argsort(-counts)[:4]]


# ------------------------------------------------------------------ F4 / F5: per-channel correlations

def channel_r():
    """Per channel and recording: r of MUA and of each LFP band envelope, codec (125 Hz seed 1) and PCA at its budget."""
    out, meta = {}, {"sources": ["codec_125", "pca_F240"], "bands": list(LFP_BANDS), "groups": {}}
    fns = {"codec_125": codec_fn(ckpt("125"), 240, 32), "pca_F240": codec_fn("pca", 240, 32)}
    for g in TEST_GROUPS:
        rec = {k: {"mua": [], **{b: [] for b in LFP_BANDS}} for k in fns}
        names = []
        for s in group_sessions(g):
            x = load(s)
            mx, x1k = mua(x), down_1k(x)
            env_x = {b: lfp_band(x1k, *f) for b, f in LFP_BANDS.items()}
            for k, fn in fns.items():
                xh = fn(x)[0][:, :x.shape[1]]
                rec[k]["mua"].append(pearson(mx, mua(xh)))
                h1k = down_1k(xh)
                for b, f in LFP_BANDS.items():
                    rec[k][b].append(pearson(env_x[b], lfp_band(h1k, *f)))
            names.append(s.name)
            print(f"  channel_r {g} {s.name}", flush=True)
        for k in fns:
            for m, v in rec[k].items():
                out[f"{g}/{k}/{m}"] = np.stack(v).astype(np.float32)             # [recordings, channels]
        meta["groups"][g] = names
    save("channel_r", out, meta)


# ------------------------------------------------------------------ F3: spectrum

def psd():
    """Welch PSD (median over channels, then over recordings) of x, x̂ at each rate and PCA; first 30 s."""
    from scipy.signal import welch
    fns = {**{f"codec_{r}": codec_fn(ckpt(r), 240, 32) for r in CKPT}, "pca_F240": codec_fn("pca", 240, 32)}
    out = {}
    for g in TEST_GROUPS:
        acc = {k: [] for k in ["x", *fns]}
        for s in group_sessions(g):
            x = load(s, 30)
            f, p = welch(x, HZ, nperseg=8192, axis=-1)
            acc["x"].append(np.median(p, 0))
            for k, fn in fns.items():
                acc[k].append(np.median(welch(fn(x)[0], HZ, nperseg=8192, axis=-1)[1], 0))
        for k, v in acc.items():
            out[f"{g}/{k}"] = np.median(np.stack(v), 0).astype(np.float32)
        out["freq"] = f.astype(np.float32)
        print(f"  psd {g}", flush=True)
    save("psd", out, {"nperseg": 8192, "seconds": 30, "sources": ["x", *fns]})


# ------------------------------------------------------------------ F2, F4c, F7, F12, F13: one real example

def examples():
    """1 s on the 4 busiest electrodes of the example recording (after the 60 s calibration, so ŷ has online IDs):
    x, x̂, ŷ, bp(x), bp(x̂), NIX Good times and units, crossings on each, online labels on ŷ. Plus 10 s of MUA and
    of the theta and gamma envelopes on the busiest electrode."""
    s, rows = example_session()
    x = load(s)
    xh, yh, _ = codec_fn(ckpt("125"), 240, 32)(x)
    x = x[:, :xh.shape[1]]
    T = x.shape[1]
    t0 = (70 if T > 80 * HZ else 5) * HZ
    # the 1 s window with the most Good spikes on these electrodes, after t0
    k = np.isin(s.spike_row, rows) & (s.spike_t >= t0) & (s.spike_t < T - HZ)
    starts = np.arange(t0, T - HZ, HZ // 10)
    a = int(starts[np.argmax([((s.spike_t[k] >= b) & (s.spike_t[k] < b + HZ)).sum() for b in starts])])
    out = {"rows": rows, "sigma_bp": s.sigma_bp[rows], "window_start": np.int64(a)}
    for i, r in enumerate(rows):
        bx, bxh = bp(x[r]), bp(xh[r])
        sl = slice(a, a + HZ)
        for name, v in (("x", x[r]), ("x_hat", xh[r]), ("y_hat", yh[r]), ("bp_x", bx), ("bp_x_hat", bxh)):
            out[f"{i}/{name}"] = v[sl].astype(np.float32)
        good = (s.spike_row == r) & (s.spike_t >= a) & (s.spike_t < a + HZ)
        bad = (s.bad_row == r) & (s.bad_t >= a) & (s.bad_t < a + HZ)
        out[f"{i}/nix_t"], out[f"{i}/nix_unit"] = s.spike_t[good] - a, s.spike_unit[good]
        out[f"{i}/nix_bad_t"] = s.bad_t[bad] - a
        for name, v in (("x", bx), ("x_hat", bxh), ("y_hat", yh[r])):
            d = detections(v, s.sigma_bp[r])
            out[f"{i}/det_{name}"] = d[(d >= a) & (d < a + HZ)] - a
        ts, lab = online(yh[r], s.sigma_bp[r])
        w = (ts >= a) & (ts < a + HZ)
        out[f"{i}/online_t"], out[f"{i}/online_label"] = ts[w] - a, lab[w]
    r = rows[0]
    m_x, m_h = mua(x[r]), mua(xh[r])
    x1k, h1k = down_1k(x[r]), down_1k(xh[r])
    b = slice(a // 30 - 5000, a // 30 + 5000)
    out.update({"mua_x": m_x[b], "mua_x_hat": m_h[b], "mua_r": np.float32(pearson(m_x, m_h))})
    for band in ("theta", "gamma"):
        ex, eh = lfp_band(x1k, *LFP_BANDS[band]), lfp_band(h1k, *LFP_BANDS[band])
        out.update({f"{band}_x": ex[b], f"{band}_x_hat": eh[b], f"{band}_r": np.float32(pearson(ex, eh))})
    save("examples", out, {"session": s.name, "rows": rows.tolist(), "window_start_s": a / HZ})


# ------------------------------------------------------------------ F6: waveforms

def waveforms():
    """Every Good unit of p4_unseen (125 Hz seed 1): mean ± SD of bp(x), bp(x̂), ŷ around its spikes (at most 500),
    the unit's SNR and the per-spike r median."""
    fn = codec_fn(ckpt("125"), 240, 32)
    keys = ("bp_x", "bp_x_hat", "y_hat")
    acc = {f"{k}_{s}": [] for k in keys for s in ("mean", "sd")}
    info = {"snr": [], "wave_r": [], "n": [], "session": [], "unit": []}
    rng = np.random.default_rng(0)
    for s in group_sessions("p4_unseen"):
        x = load(s)
        xh, yh, _ = fn(x)
        x = x[:, :xh.shape[1]]
        art = s.art(0, x.shape[1])
        for u in np.unique(s.spike_unit[s.spike_t < x.shape[1]]):
            sel = (s.spike_unit == u) & (s.spike_t < x.shape[1])
            r = int(s.spike_row[sel][0])
            t = np.sort(s.spike_t[sel])
            t = t[(t >= WF_PRE) & (t + WF_POST < x.shape[1])]
            t = t[art[r, t] == 0]
            if len(t) < 20:
                continue
            t = np.sort(rng.choice(t, min(500, len(t)), replace=False))
            idx = t[:, None] + np.arange(-WF_PRE, WF_POST)[None]
            sig = {"bp_x": bp(x[r]), "bp_x_hat": bp(xh[r]), "y_hat": yh[r]}
            sn = {k: v[idx] / s.sigma_bp[r] for k, v in sig.items()}          # in units of sigma
            for k in keys:
                acc[f"{k}_mean"].append(sn[k].mean(0)), acc[f"{k}_sd"].append(sn[k].std(0))
            info["snr"].append(float(np.median(np.ptp(sn["bp_x"], 1))))
            info["wave_r"].append(float(np.median(pearson(sn["bp_x_hat"], sn["bp_x"]))))
            info["n"].append(int(sel.sum())), info["session"].append(s.name), info["unit"].append(int(u))
        print(f"  waveforms {s.name}", flush=True)
    save("waveforms", {k: np.stack(v).astype(np.float32) for k, v in acc.items()} |
         {k: np.asarray(info[k]) for k in ("snr", "wave_r", "n")}, {"session": info["session"], "unit": info["unit"]})


# ------------------------------------------------------------------ F8, F12: sorting

def sorting_units():
    """Per NIX Good unit (125 Hz seed 1, every test group): offline accuracy on x and x̂, online on ŷ. Plus the
    snippet embedding (PCA 2) of the example electrode on bp(x) and bp(x̂), with cluster and matched NIX unit."""
    fn = codec_fn(ckpt("125"), 240, 32)
    acc = {}
    for g in TEST_GROUPS:
        res = {"x": {}, "x_hat": {}, "y_hat_online": {}}
        for s in group_sessions(g):
            x = load(s)
            xh, yh, _ = fn(x)
            one = sort_scores(s, x[:, :xh.shape[1]], xh, yh)
            for k in res:
                res[k].update(one[k])
            print(f"  sorting {g} {s.name}", flush=True)
        keys = sorted(res["x"])
        acc[g] = {"unit": keys, **{k: [res[k].get(u) for u in keys] for k in res}}
    s, rows = example_session()
    x = load(s)
    xh = fn(x)[0]
    x = x[:, :xh.shape[1]]
    r = rows[0]
    emb = {}
    sel = (s.spike_row == r) & (s.spike_t < x.shape[1])
    o = np.argsort(s.spike_t[sel])
    ref_t, ref_u = s.spike_t[sel][o], s.spike_unit[sel][o]
    for name, v in (("x", bp(x[r])), ("x_hat", bp(xh[r]))):
        t, sn = snippets(v, detections(v, s.sigma_bp[r]))
        c = sn - sn.mean(0)
        feats = c @ np.linalg.svd(c, full_matrices=False)[2][:2].T
        nix = np.full(len(t), -1)
        pairs = match(t, ref_t)
        nix[pairs[:, 0]] = ref_u[pairs[:, 1]]
        emb.update({f"{name}/feat": feats.astype(np.float32), f"{name}/cluster": cluster(sn),
                    f"{name}/nix_unit": nix, f"{name}/snip": (sn[:2000] / s.sigma_bp[r]).astype(np.float32),
                    f"{name}/snip_nix": nix[:2000]})
    save("sorting", emb, {"per_unit": acc, "example": {"session": s.name, "row": int(r)}})


# ------------------------------------------------------------------ F10: streaming

def frame_times(model, x, device, n, step=None):
    """Wall time of every frame when the 96 electrodes are streamed one frame at a time (n frames).
    `step(frame, state)`: what runs per frame (default the whole codec, `model.stream`)."""
    model = model.to(device)
    step = step or model.stream
    f = model.ratio
    state, ms = {}, []
    xt = torch.as_tensor(x, dtype=torch.float32)[None].to(device)
    with torch.no_grad():
        for i in range(n + 50):
            fr = xt[..., (i * f) % (xt.shape[-1] - f):][..., :f]
            if device == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter()
            step(fr, state)
            if device == "cuda":
                torch.cuda.synchronize()
            if i >= 50:
                ms.append((time.perf_counter() - t) * 1e3)
    return np.asarray(ms, np.float32)


def streaming():
    """Per rate (frozen seed-1 models): per-frame compute time on GPU and CPU; streamed (frame by frame) vs one
    offline pass; the perturbation test on real data (how far back does a change at t0 reach?); start-up from an
    empty state vs a warm one."""
    from utils.train import load_checkpoint
    s, rows = example_session()
    x = load(s, 8 if Q["quick"] else 60)
    out, meta = {}, {"session": s.name, "torch_threads": torch.get_num_threads(), "rates": {}}
    if DEV == "cuda":
        meta["gpu"] = torch.cuda.get_device_name(0)
    for rate in CKPT:
        model, _ = load_checkpoint(ckpt(rate), ema=True, device=DEV)
        f, mr = model.ratio, {"frame_samples": model.ratio, "budget_ms": 1e3 * model.ratio / HZ}
        n_frames = 200 if Q["quick"] else 2000
        out[f"{rate}/ms_gpu"] = frame_times(model, x[:, :HZ * 10], DEV, n_frames) if DEV == "cuda" else np.zeros(0)
        out[f"{rate}/ms_cpu"] = frame_times(model, x[:, :HZ * 10], "cpu", max(100, n_frames // 4))
        model = model.to(DEV)
        # streamed one frame at a time == one offline pass?
        seg = torch.as_tensor(x[:, :HZ * 2 // f * f], dtype=torch.float32)[None].to(DEV)
        with torch.no_grad():
            off = model.run(seg)[0]
            state, parts = {}, []
            for i in range(0, seg.shape[-1], f):
                parts.append(model.stream(seg[..., i:i + f], state)[0])
            on = torch.cat(parts, -1)
        mr["stream_vs_offline_max_abs"] = float((on - off).abs().max())
        # perturbation: from t0 (inside a frame, not on its edge) the input is replaced by another stretch of x
        n = HZ // 4 // f * f
        t0 = n // 2 + f // 3
        a = torch.as_tensor(x[:, HZ:HZ + n], dtype=torch.float32)[None].to(DEV)
        b = a.clone()
        b[..., t0:] = torch.as_tensor(x[:, 2 * HZ:2 * HZ + n - t0], dtype=torch.float32).to(DEV)
        with torch.no_grad():
            ya, yb = model.run(a)[0][0].cpu().numpy(), model.run(b)[0][0].cpu().numpy()
        diff = np.abs(ya - yb).max(0)
        moved = np.flatnonzero(diff > 1e-4 * np.abs(ya).max())
        first = int(moved[0]) if len(moved) else n
        mr.update({"t0": t0, "first_changed_sample": first, "lookback_samples": t0 - first,
                   "frame_start_of_t0": t0 // f * f, "max_abs_before_frame": float(diff[:t0 // f * f].max())})
        r = rows[0]
        out.update({f"{rate}/pert_diff": diff.astype(np.float32), f"{rate}/pert_x_a": a[0, r].cpu().numpy(),
                    f"{rate}/pert_x_b": b[0, r].cpu().numpy(), f"{rate}/pert_y_a": ya[r], f"{rate}/pert_y_b": yb[r]})
        # start-up: stream from an empty state at 30 s vs the same stretch inside a stream started at 0
        c0, L = (3 if Q["quick"] else 30) * HZ // f * f, (2 if Q["quick"] else 10) * HZ // f * f
        xt = torch.as_tensor(x[:, :c0 + L], dtype=torch.float32)[None].to(DEV)
        def stream_from(start):
            st, parts = {}, []
            with torch.no_grad():
                for i in range(start, c0 + L, f * 32):
                    parts.append(model.stream(xt[..., i:min(i + f * 32, c0 + L)], st)[0])
            return torch.cat(parts, -1)[0, :, -L:].cpu().numpy()
        warm, cold = stream_from(0), stream_from(c0)
        xs = x[:, c0:c0 + L]
        b10 = HZ // 100                                                   # 10 ms bins
        nb = L // b10
        rel = lambda h: np.sqrt(((h - warm)[:, :nb * b10].reshape(len(h), nb, b10) ** 2).mean((0, 2))) / warm.std()
        out[f"{rate}/startup_rel_err"] = rel(cold).astype(np.float32)
        # spikes kept per 1 s bin: cold vs warm, against x
        kept = {"cold": np.zeros(10), "warm": np.zeros(10)}
        found = np.zeros(10)
        for rr in np.unique(s.spike_row):
            ref = np.sort(s.spike_t[(s.spike_row == rr) & (s.spike_t >= c0) & (s.spike_t < c0 + L)] - c0)
            dets = {("x", rr): detections(bp(xs[rr]), s.sigma_bp[rr])}
            fx = found_on_x(dets, rr, ref)
            found += np.histogram(fx, np.arange(11) * HZ)[0]
            for k, h in (("cold", cold), ("warm", warm)):
                d = detections(bp(h[rr]), s.sigma_bp[rr])
                kept[k] += np.histogram(fx[match(d, fx)[:, 1]], np.arange(11) * HZ)[0]
        out[f"{rate}/startup_found"] = found
        out[f"{rate}/startup_kept_cold"], out[f"{rate}/startup_kept_warm"] = kept["cold"], kept["warm"]
        meta["rates"][rate] = mr
        print(f"  streaming {rate}: {mr}", flush=True)
    save("streaming", out, meta)


# ------------------------------------------------------------------ Fig. 2: which spikes are lost (lost-spike diagnostic)

DEPTH = [5.0, 5.25, 5.5, 6.0, 7.0, 9.0, 12.0, np.inf]


def margin():
    """Per test group, seed and output: spikes found on x and kept, binned by the trough depth on bp(x) in sigma.
    The 125 Hz seeds and PCA at its budget. Same detector and matching as retention (utils.metrics)."""
    def trough(sig, t):
        idx = np.clip(t[:, None] + np.arange(-15, 16)[None], 0, len(sig) - 1)
        return sig[idx].min(1)

    srcs = {**{f"codec_s{k}": codec_fn(ckpt("125", k), 240, 32) for k in range(1, 6)},
            "pca_F240": codec_fn("pca", 240, 32)}
    out = {}
    for g in TEST_GROUPS:
        acc = {f"{k}/{a}": np.zeros(len(DEPTH) - 1) for k in srcs for a in ("found", "x_hat", "y_hat")}
        for s in group_sessions(g):
            x = load(s)
            res = {k: fn(x) for k, fn in srcs.items()}
            T = min(v[0].shape[1] for v in res.values())
            art = s.art(0, T)
            for r in np.unique(s.spike_row[s.spike_t < T]):
                ref = np.sort(s.spike_t[(s.spike_row == r) & (s.spike_t < T)])
                ref = ref[art[r, ref] == 0]
                bx = bp(x[r, :T])
                dets = {("x", r): detections(bx, s.sigma_bp[r], art[r])}
                found = found_on_x(dets, r, ref)
                if not len(found):
                    continue
                depth = -trough(bx, found) / s.sigma_bp[r]
                b = np.digitize(depth, DEPTH) - 1
                for k, (xh, yh, _) in res.items():
                    acc[f"{k}/found"] += np.bincount(b, minlength=len(DEPTH) - 1)[:len(DEPTH) - 1]
                    for a, v in (("x_hat", bp(xh[r, :T])), ("y_hat", None if yh is None else yh[r, :T])):
                        if v is None:
                            continue
                        kept = np.zeros(len(found), bool)
                        kept[match(detections(v, s.sigma_bp[r], art[r]), found)[:, 1]] = True
                        acc[f"{k}/{a}"] += np.bincount(b[kept], minlength=len(DEPTH) - 1)[:len(DEPTH) - 1]
            print(f"  margin {g} {s.name}", flush=True)
        out.update({f"{g}/{k}": v for k, v in acc.items()})
    save("margin", out, {"edges_sigma": [float(e) for e in DEPTH], "sources": list(srcs)})


# ------------------------------------------------------------------ timing between electrodes: cross-correlograms

def ccg_all(trains, half, bw):
    """Cross-correlograms of every electrode pair at once: [C, C, nb] counts of t_j - t_i in bins of bw samples
    centred on 0, |lag| <= half. trains: one sorted array of spike times (samples) per electrode."""
    C, nb = len(trains), int(round(2 * half / bw))
    t = np.concatenate(trains)
    c = np.concatenate([np.full(len(v), i) for i, v in enumerate(trains)])
    o = np.argsort(t, kind="stable")
    t, c = t[o], c[o]
    out = np.zeros((C, C * nb), np.int64)
    for i, ti in enumerate(trains):
        if not len(ti):
            continue
        lo, hi = np.searchsorted(t, ti - half, "left"), np.searchsorted(t, ti + half, "left")
        n = hi - lo
        idx = np.repeat(lo - np.cumsum(np.r_[0, n[:-1]]), n) + np.arange(n.sum())
        b = np.clip(((t[idx] - np.repeat(ti, n) + half) // bw).astype(np.int64), 0, nb - 1)
        out[i] = np.bincount(c[idx] * nb + b, minlength=C * nb)
    return out.reshape(C, C, nb)


def ccg_peak(h, centre, flank):
    """Peak height over the flank baseline in Poisson SDs, and the lag bin of the peak (within the centre bins)."""
    base = h[..., flank].mean(-1)
    k = h[..., centre].argmax(-1)
    peak = np.take_along_axis(h[..., centre], k[..., None], -1)[..., 0]
    return (peak - base) / np.sqrt(base + 1), centre[k]


def ccg():
    """Is the timing BETWEEN electrodes kept? Per electrode pair (both with >= CCG_MIN_SPIKES detections on x):
    the cross-correlogram of the detected spike trains (same detector, sigma from bp(x), artifacts out) on x, x̂ and ŷ
    (125 / 62.5 / 31.3 Hz, seed 1), PCA at the 125 Hz budget and the NIX Good times. Each source is compared with x:
    shape r of the correlogram, and whether a peak (>= CCG_Z Poisson SDs over the flanks, |lag| <= CCG_PEAK_MS)
    is found at the same lag. x first half vs second half gives the measure's own reproducibility."""
    from utils.config import env
    bw = int(round(float(env("CCG_BIN_MS", 0.5)) * HZ / 1000))
    K = int(round(float(env("CCG_WINDOW_MS", 25)) / float(env("CCG_BIN_MS", 0.5))))
    half = bw * K + bw / 2                                       # bins centred on lag 0: [-half, half)
    lag_ms = (np.arange(2 * K + 1) - K) * bw / HZ * 1000
    centre = np.flatnonzero(np.abs(lag_ms) <= float(env("CCG_PEAK_MS", 5)))
    flank = np.flatnonzero(np.abs(lag_ms) >= float(env("CCG_FLANK_MS", 15)))
    zmin, nmin = float(env("CCG_Z", 5)), int(env("CCG_MIN_SPIKES", 100))
    fns = {**{f"codec_{r}": codec_fn(ckpt(r), {"125": 240, "62.5": 480, "31.3": 960}[r], 32) for r in CKPT},
           "pca_F240": codec_fn("pca", 240, 32)}
    out, summary = {}, {}
    for g in TEST_GROUPS:
        per = {}                                                 # source -> list of per-pair arrays
        keep = {}                                                # CCGs of the pairs significant on x, every source
        for si, s in enumerate(group_sessions(g)):
            x = load(s)
            res = {k: fn(x) for k, fn in fns.items()}
            T = min([x.shape[1]] + [v[0].shape[1] for v in res.values()])
            art = s.art(0, T)
            det = lambda sig, r: detections(sig, s.sigma_bp[r], art[r])
            trains = {"x": [det(bp(x[r, :T]), r) for r in range(s.C)]}
            for k, (xh, yh, _) in res.items():
                trains[f"{k}/x_hat"] = [det(bp(xh[r, :T]), r) for r in range(s.C)]
                if yh is not None:
                    trains[f"{k}/y_hat"] = [det(yh[r, :T], r) for r in range(s.C)]
            nix = []
            for r in range(s.C):
                v = np.sort(s.spike_t[(s.spike_row == r) & (s.spike_t < T)])
                nix.append(v[art[r, v] == 0])
            trains["nix"] = nix
            trains["x_first_half"] = [v[v < T // 2] for v in trains["x"]]
            trains["x_second_half"] = [v[v >= T // 2] - T // 2 for v in trains["x"]]
            n_x = np.array([len(v) for v in trains["x"]])
            i, j = np.triu_indices(s.C, 1)
            ok = (n_x[i] >= nmin) & (n_x[j] >= nmin)
            i, j = i[ok], j[ok]
            H = {k: ccg_all(v, half, bw)[i, j].astype(np.float64) for k, v in trains.items()}
            zx, lx = ccg_peak(H["x"], centre, flank)
            sig_x = zx >= zmin
            for k, h in H.items():
                if k in ("x", "x_second_half"):
                    continue
                ref = H["x_second_half"] if k == "x_first_half" else H["x"]
                hc, rc = h - h.mean(1, keepdims=True), ref - ref.mean(1, keepdims=True)
                r_shape = (hc * rc).sum(1) / np.sqrt((hc ** 2).sum(1) * (rc ** 2).sum(1) + 1e-12)
                z, lag = ccg_peak(h, centre, flank)
                per.setdefault(k, []).append(np.stack([r_shape, z, lag_ms[lag], zx, lag_ms[lx]], 1))
            for k, h in H.items():
                keep.setdefault(k, []).append(h[sig_x].astype(np.uint32))
            keep.setdefault("pair", []).append(np.stack([np.full(sig_x.sum(), si), i[sig_x], j[sig_x]], 1))
            print(f"  ccg {g} {s.name}: {len(i)} pairs, {sig_x.sum()} with a peak on x", flush=True)
        summary[g] = {}
        for k, v in per.items():
            a = np.concatenate(v)
            r_shape, z, lag, zx, lx = a.T
            sx, ss = zx >= zmin, z >= zmin
            both = sx & ss
            summary[g][k] = {
                "n_pairs": int(len(a)), "n_peak_x": int(sx.sum()), "n_peak_source": int(ss.sum()),
                "r_median_all": float(np.nanmedian(r_shape)),
                "r_median_peak_x": float(np.nanmedian(r_shape[sx])) if sx.any() else None,
                "peaks_kept": float(both.sum() / max(sx.sum(), 1)),        # of x's peaks, found on the source
                "peaks_new": float((ss & ~sx).sum() / max(ss.sum(), 1)),   # of the source's peaks, not on x
                "lag_diff_median_ms": float(np.median(np.abs(lag - lx)[both])) if both.any() else None,
                "lag_same_bin": float((np.abs(lag - lx) < 1e-9)[both].mean()) if both.any() else None,
                "lag_within_0.5ms": float((np.abs(lag - lx) <= 0.5 + 1e-9)[both].mean()) if both.any() else None}
            out[f"{g}/{k}/pairs"] = a.astype(np.float32)
        for k, v in keep.items():
            out[f"{g}/{k}/ccg_peak_x"] = np.concatenate(v)
        print(json.dumps({g: summary[g]}, indent=1, default=float), flush=True)
    save("ccg", {"lag_ms": lag_ms, **out},
         {"summary": summary, "pair_columns": ["r_shape_vs_x", "z", "peak_lag_ms", "z_x", "peak_lag_x_ms"],
          "bin_ms": bw / HZ * 1000, "window_ms": K * bw / HZ * 1000, "z_min": zmin, "min_spikes": nmin,
          "peak_ms": float(env("CCG_PEAK_MS", 5)), "flank_ms": float(env("CCG_FLANK_MS", 15)),
          "sources": ["x_first_half (vs x_second_half)", "nix", *[f"{k}/x_hat" for k in fns],
                      *[f"{k}/y_hat" for k in fns if k != "pca_F240"]]})


def timing():
    """Real-time cost after the demo app's optimisations (S12-C), on the S11 recording, 96 electrodes, one
    8 ms frame per call: the codec as trained (= S11), with weight norm folded into plain weights (same outputs,
    checked), and the app's pipeline (encoder + spike decoder on every electrode; x̂ only where it is shown)."""
    from app.engine import load_model
    from utils.train import load_checkpoint
    s, _ = example_session()
    x = load(s, 8 if Q["quick"] else 60)
    n = {"cuda": 200 if Q["quick"] else 2000, "cpu": 50 if Q["quick"] else 500}
    meta = {"session": s.name, "torch_threads": torch.get_num_threads(), "frame_ms": 8.0, "electrodes": int(x.shape[0])}
    if DEV == "cuda":
        meta["gpu"] = torch.cuda.get_device_name(0)
    out = {}
    for dev in dict.fromkeys([DEV, "cpu"]):
        trained, _ = load_checkpoint(ckpt("125"), device=dev)
        folded = load_model(ckpt("125"), device=dev)
        with torch.no_grad():                                     # folding changes no output
            xt = torch.as_tensor(x[:, :2400], dtype=torch.float32)[None].to(dev)
            a, b = trained.stream(xt, {}), folded.stream(xt, {})
            meta[f"{dev}_folded_max_abs_diff"] = float(max((p - q).abs().max() for p, q in zip(a, b)))

        def enc_y(fr, state, m=folded):
            with m.streaming(state):
                return m.spike_decoder(m.encoder(fr[0][:, None]))
        for name, model, step in (("as_trained", trained, None), ("folded", folded, None), ("encoder_y", folded, enc_y)):
            ms = frame_times(model, x, dev, n[dev], step)
            out[f"{dev}/{name}"] = ms
            meta[f"{dev}/{name}"] = {"p50_ms": float(np.median(ms)), "p99_ms": float(np.percentile(ms, 99))}
            print(f"  timing {dev} {name}: p50 {np.median(ms):.2f} ms, p99 {np.percentile(ms, 99):.2f} ms per 8 ms frame",
                  flush=True)
    save("timing", out, meta)


# ------------------------------------------------------------------ the threshold on ŷ (supervisor, 16 Oct)

def sweep_counts(srcs, g, ks):
    """Per source and K (crossings of -K·sigma, sigma from bp(x)): COUNTS summed over the group's electrodes with Good
    spikes, electrode-minutes, and per NIX Good unit its recall and SNR. srcs: name -> codec fn (None = bp(x))."""
    acc = {(k, K): np.zeros(4, np.int64) for k in srcs for K in ks[k]}
    rec = {(k, K): [] for k in srcs for K in ks[k]}
    snr, minutes = [], 0.0
    for s in group_sessions(g):
        x = load(s)
        y = {k: fn(x)[1] for k, fn in srcs.items() if fn is not None}
        T = min([x.shape[1]] + [v.shape[1] for v in y.values()])
        art = s.art(0, T)
        inside = s.spike_t < T
        for r in np.unique(s.spike_row[inside]):
            ref = np.sort(s.spike_t[inside & (s.spike_row == r)])
            bad = np.sort(s.bad_t[(s.bad_t < T) & (s.bad_row == r)])
            bx = bp(x[r, :T])
            sig = {k: bx if fn is None else y[k][r, :T] for k, fn in srcs.items()}
            minutes += T / HZ / 60
            us = [np.sort(t[art[r, t] == 0]) for t in (s.spike_t[inside & (s.spike_unit == u)]
                                                       for u in np.unique(s.spike_unit[inside & (s.spike_row == r)]))]
            us = [t for t in us if len(wf_ptp(bx, t))]
            snr += [float(np.median(wf_ptp(bx, t)) / s.sigma_bp[r]) for t in us]
            for k, v in sig.items():
                for K in ks[k]:
                    det = detections(v, s.sigma_bp[r], art[r], k=K)
                    acc[k, K] += detect(v, s.sigma_bp[r], ref, art[r], bad=bad, det=det)
                    rec[k, K] += [len(match(det, t)) / len(t) for t in us]
        print(f"  threshold {g} {s.name}", flush=True)
    return acc, rec, np.array(snr), minutes


def wf_ptp(bx, t):
    return unit_waveforms(bx, bx, t)[2]


def sweep_summary(acc, rec, snr, minutes):
    out = {}
    for (k, K), (tp, det, ref, tp_any) in acc.items():
        r = np.array(rec[k, K])
        out.setdefault(k, {})[f"{K:g}"] = {**prf(tp, det, ref, tp_any), "false_alarms_per_min": (det - tp_any) / minutes,
                                           "pct_units_recall_ge_0.8": 100 * float((r >= 0.8).mean()),
                                           **unit_recall(r, snr)}
    return out


def threshold():
    """Is −5σ too strict for ŷ (≈ 0 between spikes)? The 125 Hz seeds' ŷ and bp(x) (threshold crossing, the baseline)
    at every K of SWEEP_K on the validation groups. Rule, fixed before any number: per arm, K* = the K with the highest
    F1 in the worse of the two validation groups (groups are never pooled; ŷ = mean of the 5 seeds). The test groups
    are then read once, at K = 5 (S10) and K* only: a post-hoc operating point, reported as such."""
    grid = [float(v) for v in str(env("SWEEP_K", "0.5,1,1.5,2,2.5,3,3.5,4,4.5,5")).split(",")]
    srcs = {"x": None, **{f"y_hat_s{i}": codec_fn(ckpt("125", i), 240, 32) for i in range(1, 6)}}
    seeds = [k for k in srcs if k != "x"]
    val, out = {}, {}
    for g in VAL_GROUPS:
        acc, rec, snr, minutes = sweep_counts(srcs, g, {k: grid for k in srcs})
        val[g] = sweep_summary(acc, rec, snr, minutes)
        out.update({f"{g}/{k}/K{K:g}/recall_per_unit": np.array(v) for (k, K), v in rec.items()})
        out[f"{g}/snr"] = snr
    f1 = lambda g, arm, K: (val[g]["x"][f"{K:g}"]["f1"] if arm == "x"
                            else np.mean([val[g][k][f"{K:g}"]["f1"] for k in seeds]))
    pick = {arm: max(grid, key=lambda K: min(f1(g, arm, K) for g in VAL_GROUPS)) for arm in ("x", "y_hat")}
    print(f"  threshold: K* = {pick}", flush=True)
    test = {}
    for g in TEST_GROUPS:
        ks = {k: sorted({5.0, pick["x" if k == "x" else "y_hat"]}) for k in srcs}
        acc, rec, snr, minutes = sweep_counts(srcs, g, ks)
        test[g] = sweep_summary(acc, rec, snr, minutes)
        out.update({f"{g}/{k}/K{K:g}/recall_per_unit": np.array(v) for (k, K), v in rec.items()})
        out[f"{g}/snr"] = snr
    save("threshold", out, {"grid": grid, "rule": "per arm, max over K of the lower validation-group F1 (ŷ: 5-seed mean)",
                            "k_star": pick, "val": val, "test": test})


# ------------------------------------------------------------------ the latent space: what the 32 numbers hold

LATENT_F, LATENT_MAP, LATENT_LAGS, LATENT_FIT = 240, 10_000, 8, 200_000  # samples per token · map points per class · lags · readout rows
BINARY = ("spike", "crossing", "artifact")
SPIKE_ONLY = ("position", "trough")                                      # defined only on tokens that hold a spike


def cached_sessions(g):
    """The recordings of a group present in this cache; a partial cache gives a draft. The JSON lists them."""
    df = pd.read_csv(dataset.INFO_DIR / "splits.csv", keep_default_na=False)
    df = df[(df["cached"] == 1) & (df["group"] == g)]
    have = [dataset.Session(r) for _, r in df.iterrows() if (dataset.CACHE_DIR / r["patient"] / f"{r['name']}.npz").exists()]
    return (have[:1] if Q["quick"] else have), len(df)


def token_features(s, x, n_tokens, F=LATENT_F):
    """What each token could hold, [C, tokens] per feature. Token i covers samples [i F, (i + 1) F) (`streaming`).
    NIX Good times only; sigma from bp(x) at cache time; artifact tokens are flagged, not removed."""
    C, n = s.C, n_tokens * F
    per = F // 30                                                        # 1 kHz samples per token
    frame = lambda a: a[..., :n_tokens * per].reshape(*a.shape[:-1], n_tokens, per).mean(-1)
    out = {k: np.zeros((C, n_tokens), bool) for k in ("spike", "crossing")}
    out.update(unit=np.full((C, n_tokens), -1, np.int64), position=np.full((C, n_tokens), np.nan, np.float32),
               trough=np.full((C, n_tokens), np.nan, np.float32), spike_rms=np.zeros((C, n_tokens), np.float32),
               mua=np.zeros((C, n_tokens), np.float32), voltage=np.zeros((C, n_tokens), np.float32),
               **{k: np.zeros((C, n_tokens), np.float32) for b in LFP_BANDS for k in (b, f"{b}_signed")})
    out["artifact"] = s.art(0, n)[:, :n].reshape(C, n_tokens, F).any(-1)
    keep = s.spike_t < n
    for r in range(C):
        bx = bp(x[r, :n])
        out["crossing"][r, detections(bx, s.sigma_bp[r]) // F] = True
        out["spike_rms"][r] = np.sqrt((bx.reshape(n_tokens, F) ** 2).mean(-1)) / s.sigma_bp[r]
        slow = down_1k(x[r, :n])
        out["voltage"][r] = frame(slow)
        for b, (lo, hi) in LFP_BANDS.items():
            sos = sps.butter(4, [lo, hi], btype="band", fs=1000.0, output="sos")
            out[f"{b}_signed"][r] = frame(sps.sosfiltfilt(sos, slow))   # zero phase: it also sees ahead, a causal token cannot
            out[b][r] = np.log10(frame(lfp_band(slow, lo, hi)) + 1e-6)
        out["mua"][r] = np.log10(frame(mua(x[r, :n])) + 1e-6)
        k = keep & (s.spike_row == r)
        t, u = s.spike_t[k], s.spike_unit[k]
        order = np.argsort(-t)                                           # the earliest spike of a token writes last
        tok = t[order] // F
        out["spike"][r, tok] = True
        out["unit"][r, tok] = u[order]
        out["position"][r, tok] = (t[order] % F) / HZ * 1e3              # ms into the token's frame
        idx = np.clip(t[order][:, None] + np.arange(-15, 16)[None], 0, n - 1)
        out["trough"][r, tok] = -bx[idx].min(1) / s.sigma_bp[r]
    return out


def latent_scores(Z, feats, fit, seed=0):
    """Rows of Z = tokens. Per dimension |Spearman rho|; whole token: readouts fitted on `fit`, scored on the rest
    (the first and second half of each recording, as probe_f1). R2 for continuous features, AUC for binary ones."""
    from scipy.stats import rankdata
    from sklearn.ensemble import (
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
    )
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import r2_score, roc_auc_score

    rng = np.random.default_rng(seed)
    mu, sd = Z[fit].mean(0), Z[fit].std(0) + 1e-9
    Zn = (Z - mu) / sd
    pick = lambda m: rng.permutation(np.flatnonzero(m))[:LATENT_FIT]
    names, rho, linear, nonlinear = list(feats), [], [], []
    for name in names:
        y = feats[name]
        ok = np.isfinite(y.astype(float)) if name in SPIKE_ONLY else (name == "artifact") | ~feats["artifact"]
        tr, te = pick(ok & fit), pick(ok & ~fit)
        if len(tr) < 50 or len(te) < 50 or (name in BINARY and (y[tr].all() or not y[tr].any() or y[te].all() or not y[te].any())):
            rho.append(np.full(Z.shape[1], np.nan)), linear.append(np.nan), nonlinear.append(np.nan)
            continue
        ry = rankdata(y[te])
        rho.append([abs(np.corrcoef(rankdata(Zn[te, d]), ry)[0, 1]) for d in range(Z.shape[1])])
        if name in BINARY:
            a = LogisticRegression(class_weight="balanced", max_iter=300).fit(Zn[tr], y[tr])
            b = HistGradientBoostingClassifier(class_weight="balanced", random_state=seed).fit(Zn[tr], y[tr])
            linear.append(roc_auc_score(y[te], a.decision_function(Zn[te])))
            nonlinear.append(roc_auc_score(y[te], b.predict_proba(Zn[te])[:, 1]))
        else:
            a, b = Ridge(1.0).fit(Zn[tr], y[tr]), HistGradientBoostingRegressor(random_state=seed).fit(Zn[tr], y[tr])
            linear.append(r2_score(y[te], a.predict(Zn[te])))
            nonlinear.append(r2_score(y[te], b.predict(Zn[te])))
    return {"names": names, "rho": np.asarray(rho, np.float32), "linear": np.asarray(linear, np.float32),
            "nonlinear": np.asarray(nonlinear, np.float32), "binary": np.array([n in BINARY for n in names])}


def knn_agreement(points, labels, k=15):
    """Share of each point's k nearest neighbours with its own label. 0.5 by chance on two balanced classes."""
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k + 1).fit(points).kneighbors(points, return_distance=False)[:, 1:]
    return float((labels[nn] == labels[:, None]).mean())


def token_map(Zn, seed):
    from sklearn.manifold import TSNE
    return TSNE(init="pca", perplexity=min(30, max(2, (len(Zn) - 1) // 4)), learning_rate="auto", random_state=seed).fit_transform(Zn)


def latent():
    """Frozen 125 Hz model, seed 1. Dimensions are not comparable between seeds. Groups stay apart."""
    fn = codec_fn(ckpt("125"), LATENT_F, 32)
    out, meta = {}, {"model": "s8-ref-s1 step 200000", "seconds": seconds(), "sampling": f"{LATENT_MAP} spike and "
                     f"{LATENT_MAP} non-spike tokens per map, artifacts left out", "groups": {}}
    for g in TEST_GROUPS:
        ss, n_group = cached_sessions(g)
        if not ss:
            continue
        Z, F_, fit, where = [], [], [], []
        for i, s in enumerate(ss):
            x = load(s)
            z = np.asarray(fn(x)[2], np.float32)
            assert z.ndim == 3 and z.shape[0] == s.C, z.shape     # [C, K, tokens]
            f = token_features(s, x, z.shape[-1])
            C, K, T = z.shape
            Z.append(z.transpose(0, 2, 1).reshape(-1, K))
            F_.append({k: v.reshape(-1) for k, v in f.items()})
            half = np.arange(T) < T // 2
            fit.append(np.tile(half, C))
            where.append(np.stack([np.full(C * T, i), np.repeat(np.arange(C), T), np.tile(np.arange(T), C)], 1).astype(np.int32))
            print(f"  latent {g} {s.name}", flush=True)
        Z, fit, where = np.concatenate(Z).astype(np.float32), np.concatenate(fit), np.concatenate(where)
        feats = {k: np.concatenate([p[k] for p in F_]) for k in F_[0]}
        del F_
        unit = feats.pop("unit")
        scores = latent_scores(Z, feats, fit)

        # PCA of the z-scored tokens
        clean = ~feats["artifact"]
        mu, sd = Z[clean].mean(0), Z[clean].std(0) + 1e-9
        Zn = (Z - mu) / sd
        rng = np.random.default_rng(0)
        fitrows = rng.permutation(np.flatnonzero(clean))[:LATENT_FIT]
        val, vec = np.linalg.eigh(np.cov(Zn[fitrows].T))
        val, vec = val[::-1], vec[:, ::-1]
        pr = float(val.sum() ** 2 / (val ** 2).sum())

        # two maps: different tokens and t-SNE seed, the same balanced rule
        maps, agree = {}, {}
        for seed in (0, 1):
            r = np.random.default_rng(seed)
            spk, non = np.flatnonzero(clean & feats["spike"]), np.flatnonzero(clean & ~feats["spike"])
            n = min(LATENT_MAP // 4 if Q["quick"] else LATENT_MAP, len(spk), len(non))
            rows = np.r_[r.permutation(spk)[:n], r.permutation(non)[:n]]
            xy = token_map(Zn[rows], seed)
            maps[seed] = (rows, xy)
            lab = feats["spike"][rows]
            agree[f"seed{seed}"] = {"map": knn_agreement(xy, lab), "tokens_32d": knn_agreement(Zn[rows], lab), "pc12": knn_agreement(Zn[rows] @ vec[:, :2], lab), "n": int(2 * n)}
        rows, xy = maps[0]
        out.update({f"{g}/map": xy.astype(np.float32), f"{g}/map_seed1": maps[1][1].astype(np.float32),
                    f"{g}/map_spike": feats["spike"][rows], f"{g}/map_spike_seed1": feats["spike"][maps[1][0]],
                    f"{g}/map_pc12": (Zn[rows] @ vec[:, :2]).astype(np.float32),
                    f"{g}/map_voltage": feats["voltage"][rows], f"{g}/map_mua": feats["mua"][rows],
                    f"{g}/map_position": feats["position"][rows], f"{g}/map_trough": feats["trough"][rows]})

        # one electrode with the most units: its own map, unit IDs mean nothing across electrodes or recordings
        spikes = clean & (unit >= 0)
        key = where[:, 0].astype(np.int64) * 1000 + where[:, 1]
        best, unit_rows, units = None, np.zeros(0, int), []
        for k in np.unique(key[spikes]):
            mine = spikes & (key == k)
            u, c = np.unique(unit[mine], return_counts=True)
            top = u[np.argsort(-c)[:3]]
            if len(top) >= 2 and np.sort(c)[-len(top):].min() >= 30 and (best is None or (len(top), c.sum()) > best):
                best, units = (len(top), c.sum()), top
                unit_rows = np.flatnonzero(mine & np.isin(unit, top))
        if len(unit_rows):
            take = np.random.default_rng(0).permutation(unit_rows)[:LATENT_MAP]
            out.update({f"{g}/unit_map": token_map(Zn[take], 0).astype(np.float32), f"{g}/unit_label": np.searchsorted(np.sort(units), unit[take])})

        # spike-triggered mean token in PC space: the same example units, then all spikes
        order = np.lexsort((where[:, 2], key))
        lags = np.arange(-LATENT_LAGS, LATENT_LAGS + 1)
        same = key[order]
        pos = np.empty(len(order), np.int64)
        pos[order] = np.arange(len(order))
        def triggered(sel):
            p = pos[np.flatnonzero(sel)]
            p = p[(p + lags[0] >= 0) & (p + lags[-1] < len(order))]
            ok = same[p + lags[0]] == same[p + lags[-1]]
            p = p[ok][np.random.default_rng(0).permutation(ok.sum())[:LATENT_FIT // 10]]
            if not len(p):
                return np.full((len(lags), 3), np.nan, np.float32), 0
            return np.stack([(Zn[order[p + l]] @ vec[:, :3]).mean(0) for l in lags]).astype(np.float32), len(p)
        trig = [triggered(spikes)] + [triggered(spikes & np.isin(key, key[unit_rows[:1]]) & (unit == u)) for u in units]
        out.update({f"{g}/trig": np.stack([t[0] for t in trig]), f"{g}/rho": scores["rho"], f"{g}/linear": scores["linear"],
                    f"{g}/nonlinear": scores["nonlinear"], f"{g}/binary": scores["binary"], f"{g}/eigvals": val.astype(np.float32),
                    f"{g}/z_var": Z[clean].var(0).astype(np.float32)})
        meta["groups"][g] = {
            "recordings": [s.name for s in ss], "n_recordings_in_group": n_group, "tokens": int(len(Z)),
            "spike_rate": float(feats["spike"][clean].mean()), "n_spike_tokens": int(spikes.sum()),
            "n_good_spikes_in_span": int(sum((s.spike_t < load(s).shape[1]).sum() for s in ss)),
            "features": scores["names"], "participation_ratio": pr, "knn_agreement": agree, "lags": lags.tolist(),
            "n_trig": [t[1] for t in trig], "units_in_unit_map": [int(u) for u in units]}
    save("latent", out, meta)


PARTS = {"channel_r": channel_r, "psd": psd, "examples": examples, "waveforms": waveforms,
         "sorting": sorting_units, "streaming": streaming, "margin": margin, "ccg": ccg,
         "timing": timing, "threshold": threshold, "latent": latent}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("parts", nargs="+", choices=["all", *PARTS])
    ap.add_argument("--quick", action="store_true", help="smoke test: one recording per group, 20 s")
    a = ap.parse_args()
    Q["quick"] = a.quick
    frozen = {r for runs in json.loads((ROOT / "experiments" / "s10_frozen.json").read_text())["rates"].values() for r in runs}
    used = {CKPT[r].format(s) for r in CKPT for s in range(1, 6)}
    assert used <= frozen, f"not frozen for S10: {used - frozen}"
    print("device:", DEV, flush=True)
    failed = []
    for p in (PARTS if "all" in a.parts else a.parts):
        t = time.time()
        try:                                                  # one failing part must not cost the others
            PARTS[p]()
        except Exception:
            import traceback
            traceback.print_exc()
            failed.append(p)
        print(f"{p}: {time.time() - t:.0f} s", flush=True)
    if failed:
        raise SystemExit(f"failed parts: {failed}")


if __name__ == "__main__":
    main()
