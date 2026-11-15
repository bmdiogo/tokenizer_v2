"""Score a checkpoint or a baseline on ONE group (S9 on p2_late / p3_late, S10 once on the test groups).

    python -m scripts.evaluate runs/t7-s1/step_015000.pt --group p2_late
    python -m scripts.evaluate pca --group p3_late            # also: pcaw-<w>, pcafull, pcawfull-<w>, lowrate, wavelet[-db4|-haar], identity (= the x ceiling)
    python -m scripts.evaluate runs/lae-125-s1/step_200000.pt --group p3_late   # the trained linear autoencoder
    python -m scripts.evaluate pca --group p2_late --frame 480 # the PCA codec at 62.5 Hz tokens

Every arm is scored against NIX Good units with sigma from bp(x) (never from the output).
Groups are never pooled. Writes experiments/eval/<source>_<group>.json and prints `RESULT {json}`.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from models.baselines import LinearAE, PCACodec, WaveletCodec, low_rate
from utils import sorting
from utils.config import (
    HZ,
    ROOT,
    RUNS_DIR,
    TEST_GROUPS,
    TRAIN_PATIENTS,
    VAL_GROUPS,
    apply_overrides,
    env,
    split_hash,
)
from utils.dataset import sessions
from utils.dsp import LFP_BANDS, bp
from utils.metrics import COUNTS, prf, probe_f1, taus_ms


def bootstrap_ci(v, n=10_000, seed=0):
    v = np.asarray(v, float)
    if len(v) < 2:
        return [None, None]
    m = np.random.default_rng(seed).choice(v, (n, len(v))).mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def codec_fn(source, frame, k):
    """x_n [C, T] -> (x̂, ŷ or None, z or None)."""
    if source == "identity":
        return lambda x: (x, None, None)
    if source == "lowrate":
        return lambda x: (low_rate(x, frame, k), None, None)
    if source.startswith("wavelet"):                    # wavelet = sym4; pick the family on validation only
        wc = WaveletCodec(frame, k, source.partition("-")[2] or "sym4")
        return lambda x: (wc(x), None, None)
    if source == "pca" or source.startswith("pcaw-"):   # pcaw-<w>: spike-weighted PCA; pick w on validation only
        w = float(source.partition("-")[2] or 1)
        pca = PCACodec(frame, k).fit(sessions("train", patients=TRAIN_PATIENTS), spike_weight=w)
        return lambda x: (pca(x), None, None)
    if source == "pcafull" or source.startswith("pcawfull-"):   # every training frame (scripts.pca_stats first)
        w = float(source.partition("-")[2] or 1)
        pca = PCACodec.from_stats(np.load(RUNS_DIR / f"pca_stats_F{frame}.npz"), k, w)
        return lambda x: (pca(x), None, None)
    import torch
    if Path(source).parent.name.startswith("lae-"):
        ck = torch.load(source, map_location="cpu", weights_only=False)
        lae = LinearAE(**ck["config"]["linear_ae"])
        lae.load_state_dict(ck["ema"])
        lae.eval()
        return lambda x: (lae.reconstruct(x), None, None)
    from utils.train import load_checkpoint
    model, _ = load_checkpoint(source, ema=env("EVAL_EMA", True), device="cuda" if torch.cuda.is_available() else "cpu")
    return model.reconstruct


def sort_scores(s, x, x_hat, y_hat):
    """Per NIX unit: offline sorting accuracy on bp(x) and bp(x̂); online on ŷ after calibration."""
    out = {"x": {}, "x_hat": {}, "y_hat_online": {}, "unsorted": [0, 0]}
    cal = int(env("CAL_SECONDS", 60) * HZ)
    T = x.shape[1]
    for r in np.unique(s.spike_row[s.spike_t < T]):
        k = (s.spike_row == r) & (s.spike_t < T)
        t, u = s.spike_t[k], s.spike_unit[k]
        o = np.argsort(t)
        t, u = t[o], u[o]
        for arm, sig in (("x", bp(x[r])), ("x_hat", bp(x_hat[r]))):
            out[arm].update({f"{s.name}/{a}": v for a, v in sorting.offline(sig, s.sigma_bp[r], t, u).items()})
        if y_hat is not None and T > cal:
            ts, lab = sorting.online(y_hat[r], s.sigma_bp[r])
            late = t >= cal
            if late.any():
                acc = sorting.unit_accuracy(ts, lab, t[late], u[late])
                out["y_hat_online"].update({f"{s.name}/{a}": v for a, v in acc.items()})
                out["unsorted"][0] += int((lab == sorting.UNSORTED).sum())
                out["unsorted"][1] += len(lab)
    return out


def unit_recall(rec, snr, name="recall"):
    """Per NIX Good unit: % with `name` >= 0.95 and its median per SNR bin (SNR_BINS, in sigma)."""
    rec, snr = np.asarray(rec, float), np.asarray(snr, float)
    edges = [0.0] + [float(v) for v in str(env("SNR_BINS", "6,9,12,18")).split(",")] + [np.inf]
    bins = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (snr >= lo) & (snr < hi)
        bins[f"{lo:g}-{hi:g}"] = {f"{name}_median": float(np.median(rec[sel])) if sel.any() else None, "n_units": int(sel.sum())}
    return {"n_units": len(rec), f"pct_{name}_ge_0.95": 100 * float((rec >= 0.95).mean()),
            f"{name}_median": float(np.median(rec)), f"{name}_by_snr": bins}


def summarise(res, units, sort, lfp, mua_r, snr, art, probe, cross):
    out = {}
    for arm in ("x", "x_hat", "y_hat"):
        if f"{arm}_tp" not in res[0]:
            continue
        tp, det, ref, tp_any = (sum(r[f"{arm}_{c}"] for r in res) for c in COUNTS)
        per = [prf(*(r[f"{arm}_{c}"] for c in COUNTS))["f1"] for r in res]
        out[f"detect_{arm}"] = {**prf(tp, det, ref, tp_any), "f1_ci95": bootstrap_ci(per), "tp": tp,
                                "tp_any": tp_any, "n_det": det, "n_ref": ref,     # false = matches no NIX spike
                                "false_alarms_per_min": (det - tp_any) / sum(r["minutes"] for r in res),
                                "victor_purpura_median": {f"{t:g}ms": float(np.median([r[f"{arm}_vp{t:g}"] for r in res]))
                                                          for t in taus_ms()}}
        if units and f"{arm}_recall" in units[0]:
            out[f"units_{arm}"] = unit_recall([u[f"{arm}_recall"] for u in units], [u["amp_sigma"] for u in units])
        if f"{arm}_kept" in res[0]:                          # the headline: spikes found on x that the output keeps
            found, kept = sum(r["x_found"] for r in res), sum(r[f"{arm}_kept"] for r in res)
            out[f"retention_{arm}"] = {
                "retention": kept / found if found else None, "kept": kept, "found_on_x": found,
                "ci95": bootstrap_ci([r[f"{arm}_kept"] / r["x_found"] for r in res if r["x_found"]]),
                "extra_per_min": sum(r[f"{arm}_extra"] for r in res) / sum(r["minutes"] for r in res)}
            ru = [(u[f"{arm}_retention"], u["amp_sigma"]) for u in units if u.get(f"{arm}_retention") is not None]
            if ru:
                out[f"retention_{arm}"]["units"] = unit_recall(*zip(*ru), name="retention")
        if cross and f"{arm}_kept" in cross[0][0]:              # label-free: every crossing on x, no NIX
            n, kept = sum(c["x_crossings"] for c, _ in cross), sum(c[f"{arm}_kept"] for c, _ in cross)
            out[f"crossings_{arm}"] = {
                "retention": kept / n if n else None, "kept": kept, "x_crossings": n,
                "ci95": bootstrap_ci([c[f"{arm}_kept"] / c["x_crossings"] for c, _ in cross if c["x_crossings"]]),
                "new_per_min": sum(c[f"{arm}_new"] for c, _ in cross) / sum(m for _, m in cross)}
    if units:
        amp = np.array([u["amp_sigma"] for u in units])
        cut = float(np.median(amp))
        wave = lambda sel: {"r_median": float(np.median([u["wave_r"] for u, s in zip(units, sel) if s])),
                            "amp_ratio_median": float(np.median([u["wave_amp"] for u, s in zip(units, sel) if s])),
                            "n_units": int(sel.sum())}
        out["waveform"] = {"all": wave(amp >= 0), "low_amplitude": wave(amp < cut), "high_amplitude": wave(amp >= cut),
                           "amp_split_sigma": cut}
    for arm, acc in sort.items():
        if arm != "unsorted" and acc:
            v = np.array(list(acc.values()))
            out[f"sorting_{arm}"] = {"acc_median": float(np.median(v)), "n_units": len(v), "n_acc_ge_0.8": int((v >= 0.8).sum())}
    if sort.get("unsorted", [0, 0])[1]:
        out["sorting_y_hat_online"]["pct_unsorted"] = 100 * sort["unsorted"][0] / sort["unsorted"][1]
    out["lfp_r_median"] = {b: float(np.median(np.concatenate(v))) for b, v in lfp.items()}
    out["mua_r_median"] = float(np.median(np.concatenate(mua_r)))
    out["snr_db"] = {b: float(10 * np.log10(e[0] / max(e[1], 1e-12))) for b, e in snr.items()}
    out["artifacts"] = {"pct_flagged": float(np.mean([a["pct"] for a in art])),
                        "l1_inside": float(np.nanmean([a["l1_in"] if a["l1_in"] is not None else np.nan for a in art])),
                        "l1_outside": float(np.mean([a["l1_out"] for a in art]))}
    if probe:
        f, lab, fit = (np.concatenate(v) for v in zip(*probe))
        out["probe_f1"] = probe_f1(f, lab, fit)
    return out


def main():
    rest = apply_overrides()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="checkpoint path | pca | pcaw-<w> | pcafull | pcawfull-<w> | lowrate | wavelet[-db4|-haar] | identity")
    ap.add_argument("--group", required=True, choices=VAL_GROUPS + TEST_GROUPS)
    ap.add_argument("--frame", type=int, default=240, help="baselines: samples per frame (240 = 125 Hz)")
    ap.add_argument("--k", type=int, default=32, help="baselines: numbers per frame")
    ap.add_argument("--no-sorting", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(rest)
    from utils.metrics import score_session

    fn = codec_fn(a.source, a.frame, a.k)
    res, units, lfp, mua_r, art, probe, cross = [], [], {b: [] for b in LFP_BANDS}, [], [], [], []
    snr, sort = {"lfp": [0.0, 0.0], "spike": [0.0, 0.0]}, {"x": {}, "x_hat": {}, "y_hat_online": {}, "unsorted": [0, 0]}
    sel = sessions(groups=a.group)
    for s in sel:
        n = min(s.T, int(env("EVAL_SECONDS", 120) * HZ)) // 240 * 240
        x = s.x(0, n)
        x_hat, y_hat, z = fn(x)
        x = x[:, :x_hat.shape[1]]
        r = score_session(s, x, x_hat, y_hat, z)
        res += r["rows"]
        units += r["units"]
        mua_r.append(r["mua_r"])
        art.append(r["art"])
        cross.append((r["crossings"], r["seconds"] / 60))
        for b in lfp:
            lfp[b].append(r["lfp_r"][b])
        for b in snr:
            snr[b] = [snr[b][0] + r["snr"][b][0], snr[b][1] + r["snr"][b][1]]
        if "probe" in r:
            probe.append(r["probe"])
        if not a.no_sorting:
            one = sort_scores(s, x, x_hat, y_hat)
            for k in ("x", "x_hat", "y_hat_online"):
                sort[k].update(one[k])
            sort["unsorted"] = [sort["unsorted"][i] + one["unsorted"][i] for i in (0, 1)]
        print(f"  {s.name}: {len(r['rows'])} electrodes with Good spikes", flush=True)
    summary = summarise(res, units, sort, lfp, mua_r, snr, art, probe, cross)
    name = Path(a.source).parent.name + "_" + Path(a.source).stem if a.source.endswith(".pt") else f"{a.source}_F{a.frame}_K{a.k}"
    doc = {"source": a.source, "group": a.group, "n_recordings": len(sel), "split_hash": split_hash(),
           "k_threshold": env("K_THRESHOLD", 5.0), "eval_seconds": env("EVAL_SECONDS", 120), **summary}
    out = Path(a.out or ROOT / "experiments" / "eval" / f"{name}_{a.group}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({**doc, "electrodes": res, "units": units}, indent=1))
    print("RESULT " + json.dumps(doc))


if __name__ == "__main__":
    main()
