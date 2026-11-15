"""Evaluation metrics, each computed on x (the ceiling) and on the model's output.

Detection  crossings of -K·sigma (sigma = MAD of bp(x), NEVER of a reconstruction), dead time 1 ms,
           greedy one-to-one match within ±0.5 ms. Recall vs NIX Good; precision vs EVERY NIX unit
           (Good + Bad): a detection is false only if it matches no NIX spike of any label (S9,
           SpikeForest). -> P, R, F1, false alarms / min
Units      per NIX Good unit: recall on each arm vs its SNR (peak-to-peak / sigma of bp(x));
           % of units with recall >= 0.95 (the 95 % claim per unit, not pooled)
Timing     Victor-Purpura distance per electrode between detections and NIX Good times, at the
           timescales VP_TAU_MS: shifting by dt costs 2·dt/tau, so a shift beyond tau costs more than
           dropping and adding the spike. Normalised by the number of spikes: 0 = same train, 1 = none paired
Waveform   per Good spike: Pearson r and peak-to-peak ratio of bp(x̂) vs bp(x) on [t-21, t+40)
Crossings  label-free retention: of the crossings on bp(x) (every electrode), the share each output keeps
Features   median over channels of r(LFP_b(x), LFP_b(x̂)) per band and r(MUA(x), MUA(x̂))
SNR        per band: LFP (< 300 Hz) and spike band
Probe      logistic regression "spike in this 8 ms frame?" on the tokens z (secondary)
Streaming  p50 / p99 compute time per frame, streamed == offline
Headline detection and waveform numbers are computed OUTSIDE flagged artifacts.
"""
import time

import numpy as np
import torch

from utils.config import HZ, env
from utils.dsp import (
    LFP_BANDS,
    WF_POST,
    WF_PRE,
    bp,
    down_1k,
    lfp_band,
    lowpass_fft,
    mua,
)

TOL = int(round(0.0005 * HZ))          # ±0.5 ms = 15 samples
DEAD = int(round(0.001 * HZ))          # 1 ms dead time = 30 samples
COUNTS = ("tp", "det", "ref", "tp_any")  # matches to Good · detections · Good spikes · matches to Good + Bad


# ------------------------------------------------------------------ detection

def crossings(s, thr, dead=DEAD):
    """Samples where s goes below thr, at most one per dead time. s: [T]."""
    below = s < thr
    on = np.flatnonzero(below[1:] & ~below[:-1]) + 1
    keep, last = [], -dead - 1
    for t in on:
        if t - last > dead:
            keep.append(t)
            last = t
    return np.asarray(keep, np.int64)


def match(det, ref, tol=TOL):
    """Greedy one-to-one matching, nearest pairs first. -> (det index, ref index) of the matches."""
    cand = []
    j = np.searchsorted(ref, det)
    for i, jj in enumerate(j):
        for r in (jj - 1, jj):
            if 0 <= r < len(ref) and abs(det[i] - ref[r]) <= tol:
                cand.append((abs(det[i] - ref[r]), i, r))
    used_d, used_r, pairs = set(), set(), []
    for _, i, r in sorted(cand):
        if i not in used_d and r not in used_r:
            used_d.add(i), used_r.add(r), pairs.append((i, r))
    return np.array(pairs, np.int64).reshape(-1, 2)


def prf(tp, n_det, n_ref, tp_any=None):
    """Recall = tp / Good spikes. Precision = tp_any / detections when tp_any is given (Good + Bad),
    else tp / detections; `precision_good` (vs Good only) is kept to compare with older numbers."""
    r = tp / n_ref if n_ref else 0.0
    p_good = tp / n_det if n_det else 0.0
    p = p_good if tp_any is None else (tp_any / n_det if n_det else 0.0)
    out = {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}
    return out if tp_any is None else {**out, "precision_good": p_good}


def detections(s, sigma, art=None, k=None):
    """Crossings of -K·sigma on one electrode, outside artifacts."""
    det = crossings(s, -(k or env("K_THRESHOLD", 5.0)) * sigma)
    return det if art is None else det[art[det] == 0]


def detect(s, sigma, ref, art=None, k=None, bad=None, det=None):
    """COUNTS of one electrode: ref = its NIX Good times, bad = its NIX Bad times (both sorted).
    Spikes and detections on artifacts are left out. `det` skips the crossings when already known."""
    det = detections(s, sigma, art, k) if det is None else det
    bad = np.zeros(0, np.int64) if bad is None else bad
    if art is not None:
        ref, bad = ref[art[ref] == 0], bad[art[bad] == 0]
    return len(match(det, ref)), len(det), len(ref), len(match(det, np.sort(np.concatenate([ref, bad]))))


def taus_ms():
    return [float(v) for v in str(env("VP_TAU_MS", "0.5,1,5")).split(",")]


def victor_purpura(a, b, tau):
    """Victor-Purpura distance between two sorted spike trains (samples), timescale tau (samples),
    divided by len(a) + len(b). Adding or dropping a spike costs 1, shifting it by dt costs 2·dt/tau.
    Events further apart than tau are never worth pairing, so the trains are cut at every gap >= tau
    and each short segment is solved exactly (Victor & Purpura 1996)."""
    n = len(a) + len(b)
    if n == 0:
        return 0.0
    ev = np.concatenate([a, b]).astype(np.float64)
    src = np.r_[np.zeros(len(a), bool), np.ones(len(b), bool)]
    o = np.argsort(ev, kind="stable")
    ev, src = ev[o], src[o]
    cuts = np.flatnonzero(np.diff(ev) >= tau) + 1
    d = 0.0
    for e, s in zip(np.split(ev, cuts), np.split(src, cuts)):
        u, v = e[~s], e[s]
        if not len(u) or not len(v):
            d += len(e)
            continue
        D = np.zeros((len(u) + 1, len(v) + 1))
        D[:, 0], D[0, :] = np.arange(len(u) + 1), np.arange(len(v) + 1)
        for i in range(1, len(u) + 1):
            for j in range(1, len(v) + 1):
                D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + 2 * abs(u[i - 1] - v[j - 1]) / tau)
        d += D[-1, -1]
    return d / n


def waveforms(a, b, times):
    """Per spike: r and peak-to-peak ratio of a vs b (a = bp(x̂), b = bp(x)) on [t-21, t+40),
    and the peak-to-peak of b (the spike's amplitude on the recording)."""
    times = times[(times >= WF_PRE) & (times + WF_POST <= len(a))]
    idx = times[:, None] + np.arange(-WF_PRE, WF_POST)[None]
    A, B = a[idx].astype(np.float64), b[idx].astype(np.float64)
    r = pearson(A, B)
    ptp_b = np.ptp(B, 1)
    return r, np.ptp(A, 1) / np.maximum(ptp_b, 1e-12), ptp_b


def pearson(a, b):
    a, b = a - a.mean(-1, keepdims=True), b - b.mean(-1, keepdims=True)
    return (a * b).sum(-1) / (np.sqrt((a * a).sum(-1) * (b * b).sum(-1)) + 1e-12)


def snr_db(ref, est):
    return float(10 * np.log10((ref ** 2).sum() / max(((est - ref) ** 2).sum(), 1e-12)))


# ------------------------------------------------------------------ one recording

def found_on_x(dets, r, ref):
    """The NIX Good spikes (sorted) that the detector finds on the original x of electrode r."""
    return np.sort(ref[match(dets["x", r], ref)[:, 1]])


def retention_counts(dets, r, ref, arms):
    """Retention (the headline): of the Good spikes the detector finds on x, how many it also finds on each
    output. x_found = found on x · {arm}_kept = found on x AND on the arm · {arm}_extra = found on the arm
    but missed on x (ŷ can add spikes: it is trained on the NIX times; reported apart, never in retention)."""
    found = found_on_x(dets, r, ref)
    out = {"x_found": len(found)}
    for a in arms:
        if a != "x":
            hit = ref[match(dets[a, r], ref)[:, 1]]
            out[f"{a}_kept"] = len(match(dets[a, r], found))
            out[f"{a}_extra"] = len(np.setdiff1d(hit, found))
    return out


def unit_retention(dets, r, t, arms):
    """Per unit: kept / found on x for each output (None if x finds none of the unit's spikes)."""
    found = found_on_x(dets, r, t)
    return {f"{a}_retention": (len(match(dets[a, r], found)) / len(found) if len(found) else None)
            for a in arms if a != "x"}


def crossing_retention(arms, sigma, art):
    """Label-free retention: of the crossings on x (EVERY electrode, outside artifacts), how many each output also
    crosses within ±0.5 ms. No NIX: hash and Bad units count too, so ŷ (trained on Good times) is low by design.
    x_crossings · {arm}_kept · {arm}_new = crossings on the arm that x does not have."""
    out = {"x_crossings": 0, **{f"{a}_{c}": 0 for a in arms if a != "x" for c in ("kept", "new")}}
    for r in range(len(sigma)):
        found = detections(arms["x"][r], sigma[r], art[r])
        out["x_crossings"] += len(found)
        for a, sig in arms.items():
            if a != "x":
                det = detections(sig[r], sigma[r], art[r])
                kept = len(match(det, found))
                out[f"{a}_kept"] += kept
                out[f"{a}_new"] += len(det) - kept
    return out


def score_session(s, x, x_hat, y_hat=None, z=None, t0=0):
    """Everything for one recording. x, x_hat, y_hat: [C, T] normalised, starting at sample t0.

    -> {"rows": per-electrode detection counts, "units": per-unit waveform medians, "crossings": label-free retention,
        "lfp_r"/"mua_r": per-channel r, "snr": energies, "probe": (features, labels), "art": ...}
    """
    C, T = x.shape
    art = s.art(t0, T)
    arms = {"x": bp(x), "x_hat": bp(x_hat)}
    if y_hat is not None:
        arms["y_hat"] = y_hat                          # ŷ is already a spike-band trace: no bp
    inside = (s.spike_t >= t0) & (s.spike_t < t0 + T)
    bad_inside = (s.bad_t >= t0) & (s.bad_t < t0 + T)
    rows, units, dets = [], [], {}
    for r in np.unique(s.spike_row[inside]):                 # electrodes with Good spikes (recall is defined)
        ref = np.sort(s.spike_t[inside & (s.spike_row == r)] - t0)
        bad = np.sort(s.bad_t[bad_inside & (s.bad_row == r)] - t0)
        row = {"session": s.name, "row": int(r), "minutes": T / HZ / 60}
        for arm, sig in arms.items():
            dets[arm, r] = det = detections(sig[r], s.sigma_bp[r], art[r])
            counts = detect(sig[r], s.sigma_bp[r], ref, art[r], bad=bad, det=det)
            row.update(zip((f"{arm}_{c}" for c in COUNTS), counts))
            for tau in taus_ms():
                row[f"{arm}_vp{tau:g}"] = victor_purpura(det, ref[art[r, ref] == 0], tau * HZ / 1000)
        row.update(retention_counts(dets, r, ref[art[r, ref] == 0], arms))
        rows.append(row)
    for u in np.unique(s.spike_unit[inside]):
        k = inside & (s.spike_unit == u)
        r = int(s.spike_row[k][0])
        t = s.spike_t[k] - t0
        t = np.sort(t[art[r, t] == 0])
        wr, amp, ptp = waveforms(arms["x_hat"][r], arms["x"][r], t)
        if len(wr):
            units.append({"session": s.name, "unit": int(u), "row": r, "n": len(wr),
                          "wave_r": float(np.median(wr)), "wave_amp": float(np.median(amp)),
                          "amp_sigma": float(np.median(ptp) / s.sigma_bp[r]),      # the unit's SNR
                          **{f"{a}_recall": len(match(dets[a, r], t)) / len(t) for a in arms},
                          **unit_retention(dets, r, t, arms)})
    lo_x, lo_h = lowpass_fft(x), lowpass_fft(x_hat)
    x1k, h1k = down_1k(x), down_1k(x_hat)
    out = {"rows": rows, "units": units, "seconds": T / HZ,
           "lfp_r": {b: pearson(lfp_band(x1k, *f), lfp_band(h1k, *f)) for b, f in LFP_BANDS.items()},
           "mua_r": pearson(mua(x), mua(x_hat)),
           "crossings": crossing_retention(arms, s.sigma_bp, art),
           "snr": {"lfp": (float((lo_x ** 2).sum()), float(((lo_h - lo_x) ** 2).sum())),
                   "spike": (float((arms["x"] ** 2).sum()), float(((arms["x_hat"] - arms["x"]) ** 2).sum()))},
           "art": {"pct": 100 * float(art.mean()),
                   "l1_in": float(np.abs(x_hat - x)[art > 0].mean()) if art.any() else None,
                   "l1_out": float(np.abs(x_hat - x)[art == 0].mean())}}
    if z is not None:                                  # frame label: a Good spike in this 8 ms frame
        ratio = T // z.shape[-1]
        lab = np.zeros((C, z.shape[-1]), bool)
        lab[s.spike_row[inside], (s.spike_t[inside] - t0) // ratio] = True
        out["probe"] = (z.transpose(0, 2, 1).reshape(-1, z.shape[1]), lab.reshape(-1),
                        np.tile(np.arange(z.shape[-1]) < z.shape[-1] // 2, C))   # first half fits
    return out


def probe_f1(feats, labels, fit):
    """Linear probe: fitted on the first half of each recording, scored on the second half."""
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(class_weight="balanced", max_iter=300).fit(feats[fit], labels[fit])
    pred = clf.predict(feats[~fit])
    tp = int((pred & labels[~fit]).sum())
    return prf(tp, int(pred.sum()), int(labels[~fit].sum()))["f1"]


# ------------------------------------------------------------------ training-time validation windows

def window_scores(batch, x_hat, y_hat):
    """Cheap scores on the validation windows (loss span only). -> sums to accumulate over batches."""
    x, m, art, good, bad, sig = (batch[k].float().cpu().numpy() for k in ("x", "m", "art", "good", "bad", "sigma_bp"))
    x = x[..., -x_hat.shape[-1]:]
    x_hat, y_hat = x_hat.float().cpu().numpy(), y_hat.float().cpu().numpy()
    arms = {"x": bp(x), "x_hat": bp(x_hat), "y_hat": y_hat}
    out = {f"{a}_{c}": 0 for a in arms for c in COUNTS}
    out["wave_r"] = []
    for b, c in zip(*np.nonzero(good.any(-1))):
        ref, bad_t = np.flatnonzero(good[b, c]), np.flatnonzero(bad[b, c])
        for a, s in arms.items():
            for key, v in zip(COUNTS, detect(s[b, c], sig[b, c], ref, art[b, c], bad=bad_t)):
                out[f"{a}_{key}"] += v
        out["wave_r"] += list(waveforms(arms["x_hat"][b, c], arms["x"][b, c], ref[art[b, c, ref] == 0])[0])
    err = np.abs(x_hat - x)
    out.update(std=float(x_hat.std()), corr=float(np.mean(pearson(x_hat, x))),
               l1_in=(float(err[art > 0].sum()), int((art > 0).sum())),
               l1_out=(float(err[art == 0].sum()), int((art == 0).sum())))
    return out


# ------------------------------------------------------------------ streaming

@torch.no_grad()
def stream_timing(model, n_frames=2000, channels=96, device="cpu"):
    """p50 / p99 ms to process one frame of all electrodes (after 100 warm-up frames)."""
    model.eval().to(device)
    state, ms = {}, []
    frame = torch.randn(1, channels, model.ratio, device=device)
    for i in range(n_frames + 100):
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t = time.perf_counter()
        model.stream(frame, state)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        if i >= 100:
            ms.append((time.perf_counter() - t) * 1e3)
    return {"p50_ms": float(np.percentile(ms, 50)), "p99_ms": float(np.percentile(ms, 99)),
            "budget_ms": 1e3 * model.ratio / HZ}
