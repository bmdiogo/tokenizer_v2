"""The bundle the two-array show plays (S12-C): everything the page needs, computed once.

    python -m app.export P4-test-44        # after app.precompute (ŷ and z, 96 electrodes)

Writes into APP_SHOW_DIR/<recording>/ (plain files, so any static web server can serve them) and lists the
bundles of APP_SHOW_DIR in recordings.json, the menu of the page:
  sig/<row>.bin   int16 [4, T]: x, x̂, bp(x), ŷ of one electrode, in units of 1/256 (x̂ decoded from z, streaming)
  events.bin      int32 [n, 4], sorted by time: sample, electrode, kind, flag
                    kind 0 = curated spike (NIX Good)   flag bit 0: ŷ found it, bit 1: bp(x) found it
                    kind 1 = detection on ŷ             flag 1: it matches a NIX spike (Good or Bad), 0: false alarm
                    kind 2 = detection on bp(x)         same flag
  latent.bin      float32 [C, T/240, 2]: the tokens z on their first 2 principal components (display only)
  meta.json       recording, rates, sizes, K, sigma_bp
Detection and matching are the S10 ones (`utils.metrics`): −K·σ crossings with σ from bp(x) stored at cache time,
found = matched to a NIX Good spike, false alarm = matches no NIX spike of any label; spikes and detections on
artifacts are left out.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from app.engine import FRAME, load_model
from utils.config import HZ, env
from utils.config import path as env_path
from utils.dataset import Session
from utils.dsp import bp
from utils.metrics import crossings, match

SCALE = 256                                         # int16 = value × 256 (±128 σ, far beyond what is drawn)


def q(a):
    return np.clip(np.rint(np.asarray(a) * SCALE), -32767, 32767).astype("<i2")


def electrode_events(s, r, T, signals, k):
    """Events of cached row r. signals = {1: ŷ, 2: bp(x)}. -> int [n, 4]: sample, r, kind, flag."""
    art = np.zeros(T, bool)
    for row, a, b in s.art_spans:
        if row < 0 or row == r:
            art[a:b] = True
    clean = lambda t: np.sort(t[t < T][~art[t[t < T]]])
    good, bad = clean(s.spike_t[s.spike_row == r]), clean(s.bad_t[s.bad_row == r])
    found, out = np.zeros(len(good), int), []
    for kind, sig in signals.items():
        det = clean(crossings(sig, -k * s.sigma_bp[r]))
        found[match(det, good)[:, 1]] |= kind
        hit = np.zeros(len(det), int)
        hit[match(det, np.sort(np.r_[good, bad]))[:, 0]] = 1
        out.append(np.c_[det, np.full(len(det), r), np.full(len(det), kind), hit])
    return np.vstack([np.c_[good, np.full(len(good), r), np.zeros(len(good), int), found], *out])


def export(cache_dir, name, checkpoint, precomputed, out, chunk=64, say=print):
    s = Session({"name": name, "patient": name.split("_")[0], "split": "", "group": ""}, cache_dir)
    with np.load(precomputed) as f:
        rows, y_hat, z = f["rows"], f["y_hat"], f["z"]
    if len(rows) != s.C:
        raise SystemExit(f"{precomputed} has {len(rows)} electrodes; the show needs the whole array ({s.C})")
    T, k, out = y_hat.shape[1] // FRAME * FRAME, float(env("K_THRESHOLD", 5.0)), out / name
    (out / "sig").mkdir(parents=True, exist_ok=True)
    sig = [np.memmap(out / "sig" / f"{r}.bin", "<i2", "w+", shape=(4, T)) for r in range(s.C)]

    say("x, bp(x), ŷ and the events of every electrode")
    events = []
    for r in range(s.C):
        x = (np.asarray(s.counts[r, :T], np.float32) * s.scale[r] - s.mu[r]) / s.sigma[r]
        b, y = bp(x), y_hat[r, :T]
        sig[r][0], sig[r][2], sig[r][3] = q(x), q(b), q(y)
        events.append(electrode_events(s, r, T, {1: y, 2: b}, k))
    events = np.vstack(events)
    events[np.argsort(events[:, 0], kind="stable")].astype("<i4").tofile(out / "events.bin")

    say("x̂: the tokens through the streaming decoder")
    model = load_model(checkpoint)
    with torch.inference_mode(), model.streaming({}):
        for a in range(0, T // FRAME, chunk):
            xh = model.decoder(torch.from_numpy(np.ascontiguousarray(z[..., a:a + chunk])))[:, 0]
            for r in range(s.C):
                sig[r][1, a * FRAME:a * FRAME + xh.shape[1]] = q(xh[r])
            if a % (50 * chunk) == 0:
                say(f"  {100 * a * FRAME / T:3.0f} %")
    for m in sig:
        m.flush()

    zz = z[..., :T // FRAME].transpose(0, 2, 1).reshape(-1, z.shape[1])          # PCA of the tokens, all electrodes
    zz = zz - zz.mean(0)
    pcs = np.linalg.eigh(zz.T @ zz)[1][:, :-3:-1]                                # the 2 largest components
    lat = zz @ pcs
    (lat / lat.std(0)).reshape(s.C, -1, 2).astype("<f4").tofile(out / "latent.bin")

    meta = {"recording": name, "patient": s.patient, "hz": HZ, "frame": FRAME, "samples": T, "electrodes": s.C,
            "k": k, "scale": SCALE, "sigma_bp": s.sigma_bp.tolist(), "checkpoint": Path(checkpoint).parent.name}
    (out / "meta.json").write_text(json.dumps(meta))
    (out.parent / "recordings.json").write_text(json.dumps(sorted(p.parent.name for p in out.parent.glob("*/meta.json"))))
    say(f"done: {out}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording")
    ap.add_argument("--cache", type=Path, default=env_path("APP_CACHE_DIR", "data/app/cache"))
    ap.add_argument("--checkpoint", type=Path, default=env_path("APP_CHECKPOINT", "data/app/runs/s8-ref-s1/step_200000.pt"))
    ap.add_argument("--precomputed", type=Path, default=None, help="default <cache>/../precomputed/<recording>_96el.npz")
    ap.add_argument("--out", type=Path, default=env_path("APP_SHOW_DIR", "data/app/show"))
    a = ap.parse_args()
    pre = a.precomputed or a.cache.parent / "precomputed" / f"{a.recording}_96el.npz"
    export(a.cache, a.recording, a.checkpoint, pre, a.out, say=lambda m: print(m.encode("ascii", "replace").decode(), flush=True))


if __name__ == "__main__":
    main()
