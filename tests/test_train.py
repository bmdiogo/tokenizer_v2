"""A few real training steps on the synthetic cache: logging, checkpoints, true resume."""
import torch

from utils.train import TrainConfig, lr_factor, train


def test_schedulers():
    c = TrainConfig(scheduler="wsd", warmup_steps=10, max_steps=100, wsd_decay_frac=0.2)
    assert lr_factor(0, c) < lr_factor(50, c) == 1.0 > lr_factor(90, c) > lr_factor(99, c)
    assert lr_factor(99, TrainConfig(scheduler="cosine", warmup_steps=1, max_steps=100)) < 0.01


def test_resume_continues_exactly(fake_cache, small_model_env, monkeypatch):
    monkeypatch.setenv("MAX_STEPS", "4")
    straight = torch.load(train("straight") / "last.pt", weights_only=False)
    monkeypatch.setenv("MAX_STEPS", "2")
    train("resumed")
    monkeypatch.setenv("MAX_STEPS", "4")          # a resumed run keeps its config: extend it explicitly
    run = fake_cache / "runs" / "resumed"
    ck = torch.load(run / "last.pt", weights_only=False)
    ck["config"]["train"]["max_steps"] = 4
    torch.save(ck, run / "last.pt")
    resumed = torch.load(train("resumed") / "last.pt", weights_only=False)
    assert resumed["step"] == straight["step"] == 4
    for k, v in straight["model"].items():
        assert torch.allclose(v, resumed["model"][k], atol=1e-6), k
    assert (run / "step_000002.pt").exists() and (run / "plots").is_dir()


def test_a_trained_checkpoint_goes_through_evaluate(fake_cache, small_model_env, monkeypatch, tmp_path):
    import json
    import sys

    from scripts import evaluate
    monkeypatch.setenv("MAX_STEPS", "2")
    run = train("tiny")
    out = tmp_path / "e.json"
    monkeypatch.setattr(sys, "argv", ["evaluate", str(run / "step_000002.pt"), "--group", "p2_late",
                                      "--no-sorting", "--out", str(out)])
    evaluate.main()
    d = json.loads(out.read_text())
    assert {"detect_x", "detect_x_hat", "detect_y_hat", "probe_f1", "retention_x_hat", "retention_y_hat"} <= set(d)


def test_frozen_encoder_trains_only_a_fresh_spike_decoder(fake_cache, small_model_env, monkeypatch):
    """S12-A2: the tokens stay exactly those of INIT_CHECKPOINT; only the (re-initialised) spike decoder learns."""
    monkeypatch.setenv("MAX_STEPS", "2")
    src = train("joint") / "step_000002.pt"
    monkeypatch.setenv("INIT_CHECKPOINT", str(src))
    monkeypatch.setenv("FREEZE_ENCODER", "1")
    monkeypatch.setenv("SEED", "7")
    monkeypatch.setenv("MAX_STEPS", "3")
    a, b = torch.load(src, weights_only=False)["ema"], torch.load(train("frozen") / "last.pt", weights_only=False)
    for k, v in a.items():
        if not k.startswith("spike_decoder."):
            assert torch.equal(v, b["model"][k]), k                  # encoder and reconstruction decoder untouched
    assert any(not torch.equal(v, b["model"][k]) for k, v in a.items() if k.startswith("spike_decoder."))
    assert b["config"]["train"]["freeze_encoder"] and b["config"]["train"]["init_checkpoint"] == str(src)
