"""The labels of the latent figure: a wrong token index would make every panel meaningless."""
import json

import numpy as np

from scripts import figure_data as fd
from utils.dataset import sessions

F = 240


def test_spikes_land_in_their_token_and_rows_match(fake_cache):
    s = sessions("val")[0]
    n_tokens = s.T // F
    x = s.x(0, n_tokens * F)
    f = fd.token_features(s, x, n_tokens)
    assert all(v.shape == (s.C, n_tokens) for v in f.values())
    k = s.spike_t < n_tokens * F
    assert f["spike"].sum() == len(set(zip(s.spike_row[k].tolist(), (s.spike_t[k] // F).tolist())))
    r, t = int(s.spike_row[0]), int(s.spike_t[0])
    assert f["spike"][r, t // F]
    assert f["position"][r, t // F] == (t % F) / 30
    assert f["unit"][r, t // F] == s.spike_unit[0]
    assert np.isnan(f["trough"][~f["spike"]]).all() and (f["trough"][f["spike"]] > 0).all()


def test_artifact_tokens_are_flagged(fake_cache):
    s = sessions("val")[0]                                    # the fake val recording rails at sample 30 000
    n_tokens = s.T // F
    f = fd.token_features(s, s.x(0, n_tokens * F), n_tokens)
    assert f["artifact"][0, 30_000 // F]
    assert not f["artifact"].all()


def test_a_token_changes_only_from_its_own_frame_on():
    """The rule token_features relies on: an input change reaches no token before the frame that holds it."""
    from models.codec import Codec, CodecConfig
    model = Codec(CodecConfig(strides="8,5,3,2", widths="4,8,8,8,8", dilations="1|1|1|1", latent_dim=4)).eval()
    x = np.random.default_rng(0).normal(size=(1, 4800)).astype(np.float32)
    a = model.reconstruct(x)[2]
    x2, t0 = x.copy(), 2500
    x2[0, t0:] += 5.0
    b = model.reconstruct(x2)[2]
    changed = np.flatnonzero(np.abs(a - b).max(1)[0] > 0)
    assert changed[0] == t0 // F


def test_scores_find_a_planted_feature_and_ignore_noise():
    rng = np.random.default_rng(0)
    n = 8000
    z = rng.normal(size=(n, 4)).astype(np.float32)
    fit = np.arange(n) < n // 2
    feats = {"delta": z[:, 2] + 0.1 * rng.normal(size=n), "mua": rng.normal(size=n),
             "spike": z[:, 1] > 1.0, "artifact": np.zeros(n, bool)}
    s = fd.latent_scores(z, feats, fit)
    i = {name: j for j, name in enumerate(s["names"])}
    assert s["rho"][i["delta"]].argmax() == 2 and s["linear"][i["delta"]] > 0.9
    assert abs(s["linear"][i["mua"]]) < 0.05
    assert s["rho"][i["spike"]].argmax() == 1 and s["linear"][i["spike"]] > 0.9
    assert np.isnan(s["linear"][i["artifact"]])               # one class only: no score, not a fake one


def test_only_cached_recordings_are_used_and_listed(fake_cache, monkeypatch):
    ss, n_group = fd.cached_sessions("p2_late")
    assert [s.name for s in ss] and n_group == len(ss)
    (fake_cache / "cache" / "P2" / f"{ss[0].name}.npz").unlink()
    ss, n_group = fd.cached_sessions("p2_late")
    assert ss == [] and n_group == 1
    json.dumps(n_group)
