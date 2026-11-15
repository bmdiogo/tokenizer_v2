"""Detection, waveform, sorting and baselines on synthetic spikes."""
import numpy as np
import pytest

from models.baselines import PCACodec, WaveletCodec, dwt_matrix, low_rate
from utils.config import HZ
from utils.dataset import sessions
from utils.metrics import crossings, match, prf, score_session
from utils.sorting import unit_accuracy


def test_crossings_respect_the_dead_time():
    s = np.zeros(200)
    s[[10, 20, 100]] = -10
    assert list(crossings(s, -5)) == [10, 100]


def test_match_is_one_to_one_and_nearest_first():
    pairs = match(np.array([10, 12, 50]), np.array([11, 80]))
    assert len(pairs) == 1 and tuple(pairs[0]) in [(0, 0), (1, 0)]
    assert prf(1, 3, 2)["precision"] == pytest.approx(1 / 3)


def test_the_recording_scores_itself_perfectly(fake_cache):
    s = sessions("val")[0]
    x = s.x(0, s.T)
    r = score_session(s, x, x.copy(), None, None)
    tp = sum(e["x_hat_tp"] for e in r["rows"])
    assert tp == sum(e["x_tp"] for e in r["rows"]) > 0
    assert all(u["wave_r"] > 0.999 for u in r["units"])
    assert np.median(r["mua_r"]) > 0.999


def test_unit_accuracy_ignores_label_names():
    t = np.arange(0, 3000, 100)
    u = np.repeat([0, 1], 15)
    assert unit_accuracy(t, 7 - u, t, u) == {0: 1.0, 1: 1.0}


def test_pca_with_every_component_is_exact_and_low_rate_keeps_the_shape(fake_cache):
    tr = sessions("train")
    pca = PCACodec(F=24, K=24).fit(tr, seconds=0.2)
    x = tr[0].x(0, 2400)
    assert np.allclose(pca(x), x, atol=1e-3)
    assert low_rate(x).shape == x.shape


def test_evaluate_writes_every_metric_family(fake_cache, monkeypatch, tmp_path):
    import json
    import sys

    from scripts import evaluate
    out = tmp_path / "e.json"
    monkeypatch.setattr(sys, "argv", ["evaluate", "identity", "--group", "p2_late", "--no-sorting", "--out", str(out),
                                      "EVAL_SECONDS=2"])
    evaluate.main()
    d = json.loads(out.read_text())
    assert d["detect_x_hat"]["f1"] == d["detect_x"]["f1"] > 0.5
    x = d["detect_x"]                                         # the synthetic Bad spikes are real spikes too:
    assert x["tp_any"] > x["tp"] and x["precision"] > x["precision_good"]   # finding them is not a false alarm
    assert {"waveform", "lfp_r_median", "mua_r_median", "snr_db", "artifacts"} <= set(d)
    assert d["artifacts"]["pct_flagged"] > 0
    u = d["units_x_hat"]                                      # identity: every unit keeps what x keeps
    assert u == d["units_x"] and u["n_units"] > 0 and sum(b["n_units"] for b in u["recall_by_snr"].values()) == u["n_units"]
    assert d["detect_x_hat"]["victor_purpura_median"] == d["detect_x"]["victor_purpura_median"]
    c = d["crossings_x_hat"]                                  # label-free: identity keeps every crossing
    assert c["retention"] == 1.0 and c["x_crossings"] > 0 and c["new_per_min"] == 0


def test_offline_sorting_separates_two_units():
    pytest.importorskip("isosplit6")
    from utils.sorting import offline
    rng = np.random.default_rng(1)
    s = rng.normal(0, 1, 60_000)
    t = np.sort(rng.choice(np.arange(100, 59_900, 200), 200, replace=False))
    u = np.arange(200) % 2
    for ti, ui in zip(t, u):                         # odd length: ONE trough sample. A flat 2-sample trough
        s[ti - 5:ti + 6] -= (12 if ui else 25) * np.hanning(11)   # lets noise pick the alignment, and isosplit
                                                                    # then splits every unit in two (seen on the cluster)
    acc = offline(s, 1.0, t, u)
    assert min(acc.values()) > 0.8


def test_a_detection_on_a_bad_unit_is_not_a_false_positive():
    from utils.metrics import detect
    s = np.zeros(3000)
    s[[500, 1500, 2500]] = -10                       # 3 crossings: a Good spike, a Bad spike, nothing
    tp, n_det, n_ref, tp_any = detect(s, 1.0, np.array([501]), bad=np.array([1502]), k=5)
    assert (tp, n_det, n_ref, tp_any) == (1, 3, 1, 2)
    m = prf(tp, n_det, n_ref, tp_any)
    assert m["recall"] == 1.0                        # recall: Good only
    assert m["precision"] == pytest.approx(2 / 3)    # precision: false only if it matches no NIX spike
    assert m["precision_good"] == pytest.approx(1 / 3)


def test_victor_purpura_costs_and_cuts():
    from utils.metrics import victor_purpura
    a = np.array([100, 500, 900])
    assert victor_purpura(a, a, 30) == 0.0
    assert victor_purpura(a, np.array([2000, 3000]), 30) == 1.0          # nothing paired
    assert victor_purpura(np.array([100]), np.array([115]), 30) == pytest.approx(1 / 2)   # shift 15 of 30 = cost 1, / 2 spikes
    assert victor_purpura(np.array([100]), np.array([140]), 30) == 1.0    # beyond tau: drop + add


def test_victor_purpura_cut_equals_the_full_dynamic_program():
    from utils.metrics import victor_purpura
    def full(u, v, tau):
        D = np.zeros((len(u) + 1, len(v) + 1))
        D[:, 0], D[0, :] = np.arange(len(u) + 1), np.arange(len(v) + 1)
        for i in range(1, len(u) + 1):
            for j in range(1, len(v) + 1):
                D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + 2 * abs(u[i - 1] - v[j - 1]) / tau)
        return D[-1, -1] / (len(u) + len(v))
    rng = np.random.default_rng(0)
    for _ in range(20):
        u = np.sort(rng.choice(3000, rng.integers(1, 40), replace=False))
        v = np.sort(rng.choice(3000, rng.integers(1, 40), replace=False))
        assert victor_purpura(u, v, 30) == pytest.approx(full(u, v, 30))


def test_retention_counts_only_spikes_the_original_finds():
    from utils.metrics import retention_counts, unit_retention
    ref = np.array([100, 300, 500, 700])
    dets = {("x", 0): np.array([101, 301, 502]),           # x finds 3 of the 4 Good spikes
            ("x_hat", 0): np.array([100, 300, 501]),        # keeps all 3
            ("y_hat", 0): np.array([300, 700])}             # keeps 1 of the 3, and finds 700, which x missed
    arms = ["x", "x_hat", "y_hat"]
    c = retention_counts(dets, 0, ref, arms)
    assert c == {"x_found": 3, "x_hat_kept": 3, "x_hat_extra": 0, "y_hat_kept": 1, "y_hat_extra": 1}
    u = unit_retention(dets, 0, ref, arms)
    assert u["x_hat_retention"] == 1.0 and u["y_hat_retention"] == pytest.approx(1 / 3)   # extra spikes never raise it
    assert unit_retention({**dets, ("x", 0): np.zeros(0, int)}, 0, ref, arms)["x_hat_retention"] is None


def test_crossing_retention_needs_no_labels():
    from utils.metrics import crossing_retention
    x = np.zeros((2, 3000))
    x[0, [500, 1500]] = x[1, 800] = -10                      # 3 crossings on x, over two electrodes
    h = np.zeros((2, 3000))
    h[0, 505] = h[0, 2500] = -10                             # keeps 1 (within ±0.5 ms), adds 1
    out = crossing_retention({"x": x, "x_hat": h}, np.ones(2), np.zeros((2, 3000), np.int8))
    assert out == {"x_crossings": 3, "x_hat_kept": 1, "x_hat_new": 1}


def test_identity_keeps_every_spike(fake_cache):
    s = sessions("val")[0]
    x = s.x(0, 30_000)
    r = score_session(s, x, x.copy(), x.copy())
    assert sum(row["x_hat_kept"] for row in r["rows"]) == sum(row["x_found"] for row in r["rows"]) > 0


def test_ccg_all_matches_brute_force_and_finds_a_lagged_pair():
    from scripts.figure_data import ccg_all, ccg_peak
    rng = np.random.default_rng(0)
    base = np.sort(rng.choice(300_000, 400, replace=False))
    trains = [np.sort(rng.choice(300_000, 300 + 50 * i, replace=False)) for i in range(4)]
    trains[1] = np.unique(np.r_[trains[1], base])
    trains[2] = np.unique(np.r_[trains[2], base + 30])                 # 1 ms after electrode 1
    bw, K = 15, 50
    half = bw * K + bw / 2
    H = ccg_all(trains, half, bw)
    for i in range(4):
        for j in range(4):
            d = (trains[j][None] - trains[i][:, None]).ravel()
            d = d[(d >= -half) & (d < half)]
            assert (H[i, j] == np.bincount(((d + half) // bw).astype(int), minlength=2 * K + 1)).all()
    lag = (np.arange(2 * K + 1) - K) * bw / 30
    z, k = ccg_peak(H[[1, 0], [2, 3]], np.flatnonzero(np.abs(lag) <= 5), np.flatnonzero(np.abs(lag) >= 15))
    assert lag[k[0]] == 1.0 and z[0] > 20 and z[1] < 5


def test_wavelet_basis_is_orthonormal_and_keeping_everything_is_exact(fake_cache):
    for w in ("haar", "db4", "sym4"):
        for F in (240, 480):
            W = dwt_matrix(F, w)
            assert np.allclose(W @ W.T, np.eye(F), atol=1e-9)
    x = sessions("train")[0].x(0, 2400)
    assert np.allclose(WaveletCodec(F=240, K=240)(x), x, atol=1e-3)
    assert WaveletCodec(F=240, K=32)(x).shape == x.shape


def test_spike_weighted_pca_fits_spike_frames_better(fake_cache):
    tr = sessions("train")
    n = int(0.2 * HZ) // 24 * 24
    X = np.concatenate([s.x(0, n).reshape(-1, 24) for s in tr])
    spk = np.concatenate([s.m(0, n).reshape(-1, 24).any(1) for s in tr])
    assert 0 < spk.sum() < len(spk)
    err = lambda p, f: float(np.mean((((f - p.mu) @ p.U.T) @ p.U + p.mu - f) ** 2))
    plain = PCACodec(F=24, K=4).fit(tr, seconds=0.2)
    weighted = PCACodec(F=24, K=4).fit(tr, seconds=0.2, spike_weight=100.0)
    assert err(weighted, X[spk]) < err(plain, X[spk])
