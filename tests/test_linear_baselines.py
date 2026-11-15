"""PCA on every training frame and the trained linear autoencoder."""
import json
import sys

import numpy as np

import scripts.train_linear_ae as tl
from models.baselines import PCACodec, frame_stats
from scripts import evaluate
from utils.dataset import sessions


def stats_on_disk(tmp_path, monkeypatch, F):
    monkeypatch.setattr(tl, "RUNS_DIR", tmp_path / "runs")
    (tmp_path / "runs").mkdir()
    np.savez(tmp_path / "runs" / f"pca_stats_F{F}.npz", **frame_stats(sessions("train"), F))


def test_pca_on_every_frame_equals_the_direct_fit_with_or_without_weights(fake_cache):
    tr = sessions("train")
    stats = frame_stats(tr, 24, read=4800)                    # the pieces must not change the sums
    assert stats["n"] == sum(s.C * (s.T // 24) for s in tr)
    x = tr[0].x(0, 2400)
    for w in (1.0, 100.0):
        direct = PCACodec(F=24, K=4).fit(tr, seconds=2.0, spike_weight=w)      # 2 s = the whole fake recording
        assert np.allclose(PCACodec.from_stats(stats, 4, w)(x), direct(x), atol=1e-2)


def test_linear_ae_nears_pca_and_goes_through_evaluate(fake_cache, small_model_env, monkeypatch, tmp_path):
    stats_on_disk(tmp_path, monkeypatch, 8)
    for k, v in dict(MAX_STEPS="3000", EVAL_INTERVAL="3000", SAVE_EVERY="3000", LR="1e-3", EMA_DECAY="0.99").items():
        monkeypatch.setenv(k, v)
    run = tl.train("lae-test", "cpu")
    # 128 frames per step on nearly white fake data: Adam noise sets a plateau near 1.05. The real runs have
    # 12 288 frames per step (1.0001 on synthetic data at that size), and the 5 % convergence rule applies to them.
    assert json.loads((run / "history.json").read_text())["val"][-1]["ratio"] < 1.1
    out = tmp_path / "e.json"
    monkeypatch.setattr(sys, "argv", ["evaluate", str(run / "step_003000.pt"), "--group", "p2_late", "--no-sorting",
                                      "--out", str(out), "EVAL_SECONDS=2"])
    evaluate.main()
    d = json.loads(out.read_text())
    assert d["detect_x_hat"]["f1"] > 0 and d["detect_x"]["f1"] > 0


def test_linear_ae_resumes_at_its_step(fake_cache, small_model_env, monkeypatch, tmp_path):
    stats_on_disk(tmp_path, monkeypatch, 8)
    monkeypatch.setenv("MAX_STEPS", "4")
    run = tl.train("lae-a", "cpu")
    monkeypatch.setenv("MAX_STEPS", "6")
    tl.train("lae-a", "cpu")
    assert [v["step"] for v in json.loads((run / "history.json").read_text())["val"]] == [2, 4, 6]
