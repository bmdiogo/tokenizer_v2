"""The replay engine of the demo app (S12-C): a cached recording streamed frame by frame through the codec.

Each 8 ms frame (240 samples) of the chosen electrodes goes through `Codec.stream` exactly as in the S11 streaming
test, so the outputs equal the offline pass. A worker thread runs the model; the UI reads ring buffers under a lock.
Two modes: live (the model runs now, `APP_CHUNK` frames per call) and precomputed (`precompute()` ran the same
stream once and saved it; the replay then plays at 1× on any CPU).
Only what is shown is computed: the encoder and the spike decoder run on every electrode (ŷ feeds the grid and the
counters); the reconstruction decoder runs, from z, only on the selected electrode (its own causal state, restarted
when the selection changes). Weight norm is folded into plain weights once at load. On a Ryzen 7 7730U this is
~0.1× real time live at 96 electrodes; precomputed playback is 1× at any electrode count.
Live counters per electrode, the S10 metrics of a whole recording, made incremental: spikes found by the detector
(−K·σ, σ from bp(x)) on bp(x) and on ŷ, and false alarms = detections that match no NIX spike (Good or Bad).
"""
import threading
import time

import numpy as np
import torch

from utils.config import HZ, env
from utils.dataset import Session
from utils.dsp import bp
from utils.metrics import TOL, crossings, match
from utils.train import load_checkpoint


def pick_rows(cache_dir, name, n):
    """The n busiest electrodes (most NIX Good spikes first, then the rest of the array): n = 96 is the whole array.
    -> (sorted cached rows, Good-spike count per cached row)."""
    d = cache_dir / name.split("_")[0]
    z = np.load(d / f"{name}.npz")
    C = np.load(d / f"{name}.npy", mmap_mode="r").shape[0]
    counts = np.bincount(z["spike_row"][z["spike_good"]].astype(int), minlength=C)
    return np.sort(np.argsort(-counts, kind="stable")[:n]), counts


def load_model(checkpoint, device="cpu"):
    """The 125 Hz codec for inference: EMA weights, weight norm folded into plain weights (same outputs)."""
    from torch.nn.utils import parametrize
    model, _ = load_checkpoint(str(checkpoint), device=device)
    if model.ratio != 240:
        raise SystemExit(f"the app runs the 125 Hz codec (240 samples per token), not {model.ratio}")
    for m in model.modules():
        if parametrize.is_parametrized(m, "weight"):
            parametrize.remove_parametrizations(m, "weight", leave_parametrized=True)
    return model

FRAME = 240                                   # one token per electrode every 8 ms at 125 Hz
PAD = HZ // 10                                # filter context on each side of a block for bp(x)


class Replay:
    """One recording, one checkpoint, a subset of electrodes (cached rows). `step()` advances one frame."""

    def __init__(self, cache_dir, name, checkpoint, rows, seconds=None, threads=None, precomputed=None):
        torch.set_num_threads(int(threads or env("APP_THREADS", torch.get_num_threads())))
        patient = name.split("_")[0]
        self.s = Session({"name": name, "patient": patient, "split": "", "group": ""}, cache_dir)
        self.pre = None
        if precomputed:                                                   # an npz re-reads an array on every [key]
            with np.load(precomputed) as f:
                self.pre = {k: f[k] for k in ("rows", "y_hat", "z", "ms")}
            rows = self.pre["rows"]
        self.model = load_model(checkpoint)                               # x̂ of the selected electrode, always live
        latent = self.model.cfg.latent_dim
        self.chunk = int(env("APP_CHUNK", 1))
        self.rows = np.asarray(rows, int)
        self.T = min(self.s.T, int((seconds or self.s.T / HZ) * HZ)) // FRAME * FRAME
        if self.pre is not None:
            self.T = min(self.T, self.pre["y_hat"].shape[1])
        self.window = int(float(env("APP_WINDOW_S", 2.0)) * HZ) // FRAME * FRAME   # what the plots show
        self.k = float(env("K_THRESHOLD", 5.0))
        n = len(self.rows)
        self.buf = {k: np.zeros((n, self.window), np.float32) for k in ("x", "x_hat", "y_hat")}
        self.z = np.zeros((n, latent, self.window // FRAME), np.float32)
        self.state, self.t, self.ms = {}, 0, []
        self.sel, self.dec_state = 0, {}                                  # x̂ is decoded for this electrode only
        self.count = {k: np.zeros(n, np.int64) for k in ("found_x", "found_y", "false_x", "false_y", "det_x", "det_y", "nix")}
        self.recent = [[] for _ in range(n)]                              # ŷ detection times, for the rate grid
        self.y_all = [[] for _ in range(n)]                               # ŷ history (online sorter)
        self.lock, self._block = threading.Lock(), []
        nix = lambda rr, tt: {r: np.sort(tt[rr == r]) for r in self.rows}
        self.good = nix(self.s.spike_row, self.s.spike_t)
        self.any = nix(np.r_[self.s.spike_row, self.s.bad_row], np.r_[self.s.spike_t, self.s.bad_t])

    @torch.inference_mode()
    def step(self):
        """The next APP_CHUNK frames (8 ms each) of every chosen electrode through the codec, or from the precomputed
        stream. -> compute time per frame in ms (None at the end)."""
        f = min(self.chunk, (self.T - self.t) // FRAME)
        if f <= 0:
            return None
        n, a = f * FRAME, self.t
        x = self.s.x(a, n)[self.rows]
        if self.pre is not None:
            yh = self.pre["y_hat"][:, a:a + n]
            z = self.pre["z"][..., a // FRAME:a // FRAME + f]
            ms = float(self.pre["ms"][a // FRAME:a // FRAME + f].mean())
        else:
            t0 = time.perf_counter()
            with self.model.streaming(self.state):
                zt = self.model.encoder(torch.from_numpy(x)[:, None])
                yh = self.model.spike_decoder(zt)[:, 0].numpy()
            ms = 1000 * (time.perf_counter() - t0) / f
            z = zt.numpy()                                                 # z: [n, latent, f]
        with self.lock:
            sel, dec = self.sel, self.dec_state
        with self.model.streaming(dec):                                    # x̂: the selected electrode only
            xh = self.model.decoder(torch.from_numpy(np.ascontiguousarray(z[sel:sel + 1])))[0, 0].numpy()
        with self.lock:
            idx = (a + np.arange(n)) % self.window                          # ring buffers: no shifting
            for k, v in (("x", x), ("y_hat", yh)):
                self.buf[k][:, idx] = v
            if sel == self.sel:                                            # not switched meanwhile
                self.buf["x_hat"][sel, idx] = xh
            self.z[..., (a // FRAME + np.arange(f)) % self.z.shape[-1]] = z
            self.ms.append(ms)
            self._block.append((x, yh))
            for i in range(len(self.rows)):
                self.y_all[i].append(yh[i])
            self.t += n
        if sum(b[1].shape[1] for b in self._block) >= HZ:                 # count once per second of signal
            self._count()
        return ms

    def _count(self):
        """Detections on the last block: bp(x) with filter context, ŷ as is (it is already the spike band)."""
        n = sum(b[1].shape[1] for b in self._block)
        t1 = self.t - n
        lo = max(t1 - PAD, 0)
        xs = self.s.x(lo, min(self.t + PAD, self.s.T) - lo)[self.rows]
        y = np.concatenate([b[1] for b in self._block], 1)
        self._block = []
        art = self.s.art(t1, n)[self.rows]
        for i, r in enumerate(self.rows):
            sig = self.s.sigma_bp[r]
            dx = crossings(bp(xs[i])[t1 - lo:t1 - lo + n], -self.k * sig)
            dy = crossings(y[i], -self.k * sig)
            dx, dy = dx[art[i, dx] == 0], dy[art[i, dy] == 0]
            good = self.good[r][(self.good[r] >= t1) & (self.good[r] < self.t)] - t1
            good = good[art[i, good] == 0]
            anyr = self.any[r][(self.any[r] >= t1 - TOL) & (self.any[r] < self.t + TOL)] - t1
            with self.lock:
                self.count["nix"][i] += len(good)
                self.count["det_x"][i] += len(dx)
                self.count["det_y"][i] += len(dy)
                self.count["found_x"][i] += len(match(dx, good))
                self.count["found_y"][i] += len(match(dy, good))
                self.count["false_x"][i] += len(dx) - len(match(dx, anyr))
                self.count["false_y"][i] += len(dy) - len(match(dy, anyr))
                self.recent[i] = [t for t in self.recent[i] if t > self.t - 5 * HZ] + list(dy + t1)

    def snapshot(self, i=None):
        """Copies of what the UI draws, taken under the lock: the buffers of electrode i only (all when None)."""
        with self.lock:
            o = -(self.t % self.window)                                       # oldest sample first
            pick = slice(None) if i is None else slice(i, i + 1)
            return {"t": self.t, "buf": {k: np.roll(v[pick], o, 1) for k, v in self.buf.items()},
                    "z": np.roll(self.z[pick], o // FRAME, -1),
                    "ms": list(self.ms[-250:]), "count": {k: v.copy() for k, v in self.count.items()},
                    "rate": np.array([len(r) / 5.0 for r in self.recent])}

    def select(self, i):
        """Decode x̂ for electrode i from now on: a fresh causal state (a cold start settles at once, S11)."""
        with self.lock:
            if i != self.sel:
                self.buf["x_hat"][self.sel] = 0
                self.sel, self.dec_state = i, {}

    def y_history(self, i):
        with self.lock:
            return np.concatenate(self.y_all[i]) if self.y_all[i] else np.zeros(0, np.float32)


def precompute(cache_dir, name, checkpoint, rows, out, chunk=32, progress=None, device="cpu"):
    """The same causal stream as live mode (encoder + spike decoder on every electrode), run once over the whole
    recording and saved with the compute time per frame; x̂ is decoded from z at playback, for the selected electrode.
    Chunks of `chunk` frames give the same outputs as frame by frame (S11: streaming = offline)."""
    r = Replay(cache_dir, name, checkpoint, rows)
    model, n = r.model.to(device), len(r.rows)
    yh = np.zeros((n, r.T), np.float32)
    z, ms = np.zeros((n, model.cfg.latent_dim, r.T // FRAME), np.float32), np.zeros(r.T // FRAME, np.float32)
    state = {}
    with torch.inference_mode():
        for a in range(0, r.T, chunk * FRAME):
            b = min(a + chunk * FRAME, r.T)
            t0 = time.perf_counter()
            with model.streaming(state):
                zt = model.encoder(torch.from_numpy(r.s.x(a, b - a)[r.rows])[:, None].to(device))
                yh[:, a:b] = model.spike_decoder(zt)[:, 0].float().cpu().numpy()
            if device != "cpu":
                torch.cuda.synchronize()
            ms[a // FRAME:b // FRAME] = 1000 * (time.perf_counter() - t0) / ((b - a) // FRAME)
            z[..., a // FRAME:b // FRAME] = zt.float().cpu().numpy()
            if progress:
                progress(b / r.T)
    np.savez(out, rows=r.rows, y_hat=yh, z=z, ms=ms, name=name, checkpoint=str(checkpoint), chunk=chunk,
             device=str(device))
    return out


class Worker(threading.Thread):
    """Runs `Replay.step` at `speed` × real time (or as fast as the CPU allows), until stopped or the end."""

    def __init__(self, replay, speed=1.0):
        super().__init__(daemon=True)
        self.r, self.speed, self.running, self.alive = replay, speed, False, True

    def run(self):
        start, t_start = time.perf_counter(), self.r.t
        while self.alive:
            if not self.running:
                time.sleep(0.02)
                start, t_start = time.perf_counter(), self.r.t
                continue
            due = start + (self.r.t - t_start) / HZ / self.speed
            wait = due - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
            if self.r.step() is None:
                self.running = False
