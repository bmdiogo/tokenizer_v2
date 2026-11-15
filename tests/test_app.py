"""The demo app: its engine equals the offline pass, live and precomputed; the show bundle and its server."""
from dataclasses import asdict

import numpy as np
import pytest
import torch

from app.engine import FRAME, Replay, precompute
from models.codec import Codec, CodecConfig


@pytest.fixture
def ckpt(tmp_path):
    """A tiny codec with the app's 240 samples per token, saved the way training saves it."""
    cfg = CodecConfig(strides="8,5,3,2", widths="2,2,2,2,2", dilations="1|1|1|1", latent_dim=4, decoder_mult=1.0)
    torch.manual_seed(0)
    m = Codec(cfg)
    path = tmp_path / "tiny.pt"
    torch.save({"ema": m.state_dict(), "model": m.state_dict(), "config": {"codec": asdict(cfg)}}, path)
    return path


@pytest.mark.parametrize("chunk", ["1", "4"])
def test_live_replay_equals_the_offline_pass(fake_cache, ckpt, monkeypatch, chunk):
    monkeypatch.setenv("APP_CHUNK", chunk)
    monkeypatch.setenv("APP_WINDOW_S", "0.5")
    r = Replay(fake_cache / "cache", "P2_2022_01_01_fake", ckpt, rows=[1, 3])
    while r.t < 80 * FRAME:
        r.step()
    xh, yh, z = r.model.reconstruct(r.s.x(0, r.t)[[1, 3]])
    b, w = r.snapshot()["buf"], r.window
    assert np.abs(b["y_hat"] - yh[:, -w:]).max() < 1e-4                   # ŷ and z: every electrode
    assert np.abs(r.snapshot()["z"][..., -5:] - z[..., -5:]).max() < 1e-4
    assert np.abs(b["x_hat"][0] - xh[0, -w:]).max() < 1e-4 and not b["x_hat"][1].any()   # x̂: the selected one only


def test_switching_electrode_restarts_its_reconstruction_from_the_tokens(fake_cache, ckpt, monkeypatch):
    monkeypatch.setenv("APP_WINDOW_S", "0.5")
    monkeypatch.setenv("APP_CHUNK", "1")                                 # the switch lands after frame 30
    r = Replay(fake_cache / "cache", "P2_2022_01_01_fake", ckpt, rows=[1, 3])
    for _ in range(30):
        r.step()
    r.select(1)
    while r.t < 100 * FRAME:
        r.step()
    _, _, z = r.model.reconstruct(r.s.x(0, r.t)[[1, 3]])
    with torch.inference_mode(), r.model.streaming({}):                    # a cold start at the switch
        xh = r.model.decoder(torch.from_numpy(z[1:2, :, 30:]))[0, 0].numpy()
    b, w = r.snapshot()["buf"], r.window
    assert np.abs(b["x_hat"][1] - xh[-w:]).max() < 1e-4 and not b["x_hat"][0].any()


def test_precomputed_replay_plays_the_same_stream(fake_cache, ckpt, tmp_path, monkeypatch):
    monkeypatch.setenv("APP_WINDOW_S", "0.5")
    monkeypatch.setenv("APP_CHUNK", "1")
    out = precompute(fake_cache / "cache", "P2_2022_01_01_fake", ckpt, [0, 2], tmp_path / "pre.npz", chunk=8)
    live = Replay(fake_cache / "cache", "P2_2022_01_01_fake", ckpt, rows=[0, 2])
    pre = Replay(fake_cache / "cache", "P2_2022_01_01_fake", ckpt, rows=None, precomputed=out)
    assert list(pre.rows) == [0, 2]
    while live.t < 70 * FRAME:                     # past one second: the counters run
        live.step(), pre.step()
    a, b = live.snapshot(), pre.snapshot()
    assert np.abs(a["buf"]["y_hat"] - b["buf"]["y_hat"]).max() < 1e-4
    assert np.abs(a["buf"]["x_hat"] - b["buf"]["x_hat"]).max() < 1e-4          # x̂ decoded from the stored z
    for k in a["count"]:
        assert (a["count"][k] == b["count"][k]).all(), k


def test_precompute_command_writes_what_the_app_plays(fake_cache, ckpt, tmp_path, monkeypatch):
    """The container entry point (no UI): `python -m app.precompute <recording> ...`."""
    import sys

    from app import precompute as cli
    monkeypatch.setattr(sys, "argv", ["precompute", "P2_2022_01_01_fake", "--electrodes", "4", "--cache",
                                      str(fake_cache / "cache"), "--checkpoint", str(ckpt), "--out", str(tmp_path),
                                      "--device", "cpu"])
    cli.main()
    out = tmp_path / "P2_2022_01_01_fake_4el.npz"
    r = Replay(fake_cache / "cache", "P2_2022_01_01_fake", ckpt, rows=None, precomputed=out)
    assert list(r.rows) == [0, 1, 2, 3] and r.step() is not None


def test_export_events_equal_the_s10_detector(fake_cache, ckpt, tmp_path):
    """events.bin gives the counts of `utils.metrics.detect` on the whole recording; x̂ = the offline decoder."""
    import json

    from app.engine import load_model
    from app.export import SCALE, export
    from utils.dataset import Session
    from utils.dsp import bp
    from utils.metrics import detect
    cache, name = fake_cache / "cache", "P2_2022_01_01_fake"
    pre = precompute(cache, name, ckpt, [0, 1, 2, 3], tmp_path / "pre.npz", chunk=8)
    out = export(cache, name, ckpt, pre, tmp_path / "show", chunk=8, say=lambda m: None)
    T = json.loads((out / "meta.json").read_text())["samples"]
    ev = np.fromfile(out / "events.bin", "<i4").reshape(-1, 4)
    assert (np.diff(ev[:, 0]) >= 0).all()
    s, yh = Session({"name": name, "patient": "P2", "split": "", "group": ""}, cache), np.load(pre)["y_hat"]
    x = s.x(0, T)
    for r in range(4):
        e = ev[ev[:, 1] == r]
        good, bad = np.sort(s.spike_t[s.spike_row == r]), np.sort(s.bad_t[s.bad_row == r])
        for kind, sig in ((1, yh[r, :T]), (2, bp(x[r]))):
            tp, det, ref, tp_any = detect(sig, s.sigma_bp[r], good[good < T], s.art(0, T)[r], bad=bad[bad < T])
            d, g = e[e[:, 2] == kind], e[e[:, 2] == 0]
            assert (len(d), d[:, 3].sum(), len(g), (g[:, 3] & kind > 0).sum()) == (det, tp_any, ref, tp)
    assert (ev[ev[:, 2] == 0][:, 3] & 2).sum() > 0                                # bp(x) finds the planted spikes
    sig = np.fromfile(out / "sig" / "2.bin", "<i2").reshape(4, T) / SCALE
    xh, _, _ = load_model(ckpt).reconstruct(x)
    assert np.abs(sig[1] - xh[2]).max() < 1e-2 and np.abs(sig[0] - x[2]).max() < 1e-2
    assert np.fromfile(out / "latent.bin", "<f4").size == 4 * (T // FRAME) * 2


def test_server_serves_the_page_and_the_bundle(tmp_path, monkeypatch):
    import threading
    import urllib.request

    from app import server
    (tmp_path / "meta.json").write_text("{}")
    monkeypatch.setenv("APP_SHOW_DIR", str(tmp_path))
    srv = server.serve(port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    get = lambda p: urllib.request.urlopen(f"http://127.0.0.1:{srv.server_address[1]}{p}")
    try:
        assert get("/show/meta.json").read() == b"{}" and b"<canvas" in get("/").read()
        assert get("/main.js").headers["Content-Type"].startswith("text/javascript")
    finally:
        srv.shutdown()
