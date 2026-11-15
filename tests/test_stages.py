"""The plan's stage layer: which runs each step submits, how compare judges them, S7-D's mask, S8's legs."""
import json

import numpy as np
import pytest
import torch

from scripts import compare, stage


def test_s5_draws_are_reproducible_and_inside_the_search_ranges():
    a, b = stage.s5(), stage.s5()
    assert a == b and len(a) == 8
    for extra in a.values():
        kv = dict(w.split("=") for w in extra.split())
        assert 5e-5 <= float(kv["LR"]) <= 2e-3
        assert kv["WEIGHT_DECAY"] in {"0", "1e-4", "1e-3", "1e-2"} and kv["SCHEDULER"] in {"cosine", "constant", "wsd"}
        assert kv["LAMBDA_BG"] in {"0.1", "0.3", "1.0"} and "LAMBDA_STFT" not in kv          # λ_STFT stays 0 (T9)


def test_s6_appends_one_stage_per_halving_with_the_t2_burn_in(monkeypatch):
    monkeypatch.setenv("LR", "2e-4")
    runs = stage.s6()
    assert len(runs) == 9
    kv = dict(w.split("=", 1) for w in runs["s6-313hz-lr2"].split())
    assert kv["STRIDES"] == "8,5,3,2,2,2" and kv["BURN_IN"] == "15360" and float(kv["LR"]) == 4e-4
    assert len(kv["WIDTHS"].split(",")) == 7 and kv["DILATIONS"].count("|") == 5


def test_no_stage_touches_test_groups_or_p4(monkeypatch, tmp_path):
    monkeypatch.setattr(stage, "ROOT", tmp_path)
    (tmp_path / "experiments").mkdir()
    (tmp_path / "experiments" / "s7_arms.json").write_text(json.dumps({"runs": stage.s7()}))
    every = {**stage.s5(), **stage.s6(), **stage.s7(), **stage.s8(["ref", "s7-e-box"])}
    text = " ".join(every.values())
    assert "P4" not in text and "final" not in text and "unseen" not in text and "TRAIN_PATIENTS" not in text
    s8 = stage.s8(["s7-e-box"])
    assert len(s8) == 5 and s8["s8-s7-e-box-s3"] == "MASK_ALPHA=0 SEED=3 DATA_SEED=13 MAX_STEPS=200000"
    with pytest.raises(SystemExit):
        stage.s8(["nope"])


def test_compare_keeps_arms_within_the_floor_and_reads_seeds_as_one_arm():
    assert compare.arm_of("s8-ref-s3") == "s8-ref" and compare.arm_of("s5-c3") == "s5-c3"
    empty = {m: [] for m in compare.ALL}
    scores = {"a": {**empty, "f1_y_hat": [0.70, 0.72]}, "b": {**empty, "f1_y_hat": [0.69]}, "c": {**empty, "f1_y_hat": [0.60]}}
    out = compare.judge(scores, {"f1_y_hat": 0.03}, ref="a")
    assert out["a"]["f1_y_hat"]["mean"] == pytest.approx(0.71) and out["a"]["f1_y_hat"]["n"] == 2
    assert out["b"]["f1_y_hat"]["within_floor_of_best"] and not out["c"]["f1_y_hat"]["within_floor_of_best"]
    assert out["b"]["f1_y_hat"]["vs_ref_verdict"] == "no difference" and out["c"]["f1_y_hat"]["vs_ref_verdict"] == "worse"


def test_crossings_mask_finds_the_spikes_without_labels(fake_cache, monkeypatch):
    from utils.dataset import sessions
    s = sessions("train")[0]
    nix = s.m(1000, 20_000)
    monkeypatch.setenv("MASK_SOURCE", "crossings")
    cross = s.m(1000, 20_000)
    assert cross.sum() > 0
    overlap = (np.minimum(nix, cross).sum() / nix.sum())
    assert overlap > 0.5                                       # the big synthetic spikes cross -5 sigma near their NIX time


def test_a_finished_run_resumed_again_takes_no_step(fake_cache, small_model_env, monkeypatch):
    from utils.train import train
    monkeypatch.setenv("MAX_STEPS", "2")
    run = train("done")
    before = torch.load(run / "last.pt", weights_only=False)
    train("done")
    after = torch.load(run / "last.pt", weights_only=False)
    assert after["step"] == before["step"] == 2
    for k, v in before["model"].items():
        assert torch.equal(v, after["model"][k]), k


def test_art_check_counts_spikes_inside_flagged_spans(fake_cache, monkeypatch):
    from scripts.preflight import art_check
    from utils.dataset import sessions
    monkeypatch.setattr(art_check, "ROOT", fake_cache)
    (fake_cache / "experiments").mkdir(exist_ok=True)
    out = art_check.check(sessions("val")[0])                  # the rail run is flagged
    assert out["n_spans"] > 0 and 0 < out["pct_time_flagged"] < 100
    assert out["pct_good_spikes_inside"] is not None and (fake_cache / "experiments" / out["figure"]).exists()


def test_floor_is_t7_spread_at_the_step_else_the_reference_seeds():
    t7g = {"f1_y_hat": {"by_step": {"15000": [0.70, 0.705]}, "floor": 0.05}}
    assert abs(compare.floor_at(t7g, 15000)["f1_y_hat"] - 0.005) < 1e-9
    assert abs(compare.floor_at(t7g, 200000, {"f1_y_hat": [0.80, 0.81, 0.79]})["f1_y_hat"] - 0.02) < 1e-9
    assert compare.floor_at(t7g, 200000)["f1_y_hat"] == 0.05


def test_test_groups_have_no_t7_floor_so_the_reference_seeds_set_it():
    assert abs(compare.floor_at({}, 200000, {"f1_y_hat": [0.80, 0.81]})["f1_y_hat"] - 0.01) < 1e-9
    assert compare.floor_at({}, 200000) == {}
