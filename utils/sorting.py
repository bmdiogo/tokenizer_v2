"""Units: offline sorting of bp(x̂) (vs the same sorter on bp(x)) and the online sorter on ŷ.

Offline: crossings -> 61-sample snippets aligned on the trough -> PCA(3) -> isosplit (ElcheSort's
clusterer), one electrode at a time.
Online: templates from the first CAL_SECONDS of the recording, then
every spike gets the nearest template if it is closer than tau, else UNSORTED; templates follow
slow drift with rate alpha (OSort).
Score: Hungarian matching of sorted units to NIX Good units on the same electrode, then SpikeForest
accuracy per NIX unit, tp / (tp + fn + fp).
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

from utils.config import HZ, env
from utils.dsp import WF_POST, WF_PRE
from utils.metrics import crossings, match

UNSORTED = -1


def snippets(s, times):
    """Trough-aligned 61-sample snippets. -> (aligned times, [n, 61]) for the times that fit."""
    times = times[(times + 15 < len(s)) & (times >= WF_PRE)]
    trough = times + np.array([np.argmin(s[t:t + 15]) for t in times], np.int64)
    trough = trough[trough + WF_POST <= len(s)]
    return trough, s[trough[:, None] + np.arange(-WF_PRE, WF_POST)[None]]


def cluster(sn):
    """PCA(3) + isosplit6. Fewer than 10 snippets -> one cluster."""
    if len(sn) < 10:
        return np.zeros(len(sn), int)
    from isosplit6 import isosplit6
    c = sn - sn.mean(0)
    feats = c @ np.linalg.svd(c, full_matrices=False)[2][:3].T
    return np.asarray(isosplit6(feats.astype(np.float64)), int)


def unit_accuracy(times, labels, ref_times, ref_units):
    """{NIX unit: tp / (tp + fn + fp)} after the best one-to-one matching of labels to units."""
    units, labs = np.unique(ref_units), [l for l in np.unique(labels) if l != UNSORTED]
    tp = np.zeros((len(labs), len(units)))
    for i, l in enumerate(labs):
        for j, u in enumerate(units):
            tp[i, j] = len(match(np.sort(times[labels == l]), np.sort(ref_times[ref_units == u])))
    acc = {int(u): 0.0 for u in units}
    for i, j in zip(*linear_sum_assignment(-tp)):
        n_l, n_u = (labels == labs[i]).sum(), (ref_units == units[j]).sum()
        acc[int(units[j])] = tp[i, j] / (n_l + n_u - tp[i, j]) if n_l + n_u - tp[i, j] else 0.0
    return acc


def offline(s, sigma, ref_times, ref_units, k=None):
    """Sort one electrode's spike-band trace s and score it against its NIX Good units."""
    t, sn = snippets(s, crossings(s, -(k or env("K_THRESHOLD", 5.0)) * sigma))
    return unit_accuracy(t, cluster(sn), ref_times, ref_units)


def online(y, sigma, cal_seconds=None, min_spikes=50, tau_pct=99, alpha=0.0, k=None, clusterer=None):
    """The online sorter on one electrode's ŷ. -> (times, labels) after the calibration window.
    T8 (ideal y and the trained ŷ of s5-c8): tau at the 99th percentile sorts better on both (+0.024 / +0.015)
    with a third of the unsorted spikes; alpha = 0 (no template drift) is as good as 0.01, so the simpler one."""
    cal = int((cal_seconds or env("CAL_SECONDS", 60)) * HZ)
    t, sn = snippets(y, crossings(y, -(k or env("K_THRESHOLD", 5.0)) * sigma))
    first = t < cal
    lab = (clusterer or cluster)(sn[first])          # clusterer: the app's fallback where isosplit6 cannot build
    keep = [u for u in np.unique(lab) if (lab == u).sum() >= min_spikes]
    if not keep:
        return t[~first], np.full((~first).sum(), UNSORTED)
    W = np.stack([sn[first][lab == u].mean(0) for u in keep])
    dist = lambda v: np.linalg.norm(W - v, axis=1)
    tau = np.percentile([dist(v).min() for v in sn[first]], tau_pct)       # how far a unit's own spikes go
    labels = []
    for v in sn[~first]:
        d = dist(v)
        u = int(np.argmin(d))
        if d[u] > tau:
            labels.append(UNSORTED)                   # a new unit or an overlap: a spike without an ID
        else:
            labels.append(u)
            W[u] += alpha * (v - W[u])                # follow slow drift
    return t[~first], np.asarray(labels, int)
