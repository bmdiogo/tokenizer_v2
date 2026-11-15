"""Training loop, checkpointing and Weights & Biases monitoring.

AdamW · warm-up + constant / cosine / WSD · bf16 autocast · weight EMA 0.999.
Run folder <RUNS_DIR>/<exp-name>/:
    config.json        every setting of the run (codec, loss, train, split hash, git hash)
    last.pt            weights + EMA + optimizer + scheduler + step: a TRUE resume (same data stream)
    step_NNNNNN.pt     the fixed checkpoint set (weights + EMA), every SAVE_EVERY steps
    history.json, plots/
Re-running the same exp-name resumes from last.pt.
"""
import copy
import json
import math
import os
import time
from collections import defaultdict
from dataclasses import asdict, dataclass

import numpy as np
import torch
import wandb
from tqdm import tqdm

from models.codec import Codec, CodecConfig
from utils import plots
from utils.config import (
    HELD_OUT,
    HZ,
    RUNS_DIR,
    TRAIN_PATIENTS,
    E,
    env,
    git_hash,
    split_hash,
)
from utils.dataset import sessions, train_loader, val_loader
from utils.dsp import LFP_BANDS, down_1k, lfp_band, mua
from utils.losses import LossConfig, losses
from utils.metrics import COUNTS, pearson, prf, window_scores


# ================= CONFIG — every value can be changed in .env =================
@dataclass
class TrainConfig:
    burn_in: int = E("BURN_IN", 11520)           # >= receptive field, multiple of the stride product (T2)
    window: int = E("WINDOW", 7680)              # loss span L
    batch_size: int = E("BATCH_SIZE", 4)
    lr: float = E("LR", 6.07e-4)                 # chosen by the S5 search (run s5-c8)
    weight_decay: float = E("WEIGHT_DECAY", 1e-4)   # chosen by the S5 search (run s5-c8)
    scheduler: str = E("SCHEDULER", "constant")  # constant | cosine | wsd
    warmup_steps: int = E("WARMUP_STEPS", 1000)
    wsd_decay_frac: float = E("WSD_DECAY_FRAC", 0.2)
    max_steps: int = E("MAX_STEPS", 200_000)
    eval_interval: int = E("EVAL_INTERVAL", 1000)
    log_interval: int = E("LOG_INTERVAL", 100)
    save_every: int = E("SAVE_EVERY", 5000)
    ema_decay: float = E("EMA_DECAY", 0.999)
    num_workers: int = E("NUM_WORKERS", 4)
    val_windows: int = E("VAL_WINDOWS", 256)
    monitor_seconds: float = E("MONITOR_SECONDS", 10.0)   # the fixed segment behind the LFP / MUA / PSD plots
    seed: int = E("SEED", 42)                    # model init
    data_seed: int = E("DATA_SEED", 42)          # order of the training windows
    amp: bool = E("AMP", True)
    init_checkpoint: str = E("INIT_CHECKPOINT", "")    # S12-A2: start from these weights (a frozen S10 checkpoint)
    freeze_encoder: bool = E("FREEZE_ENCODER", False)  # S12-A2: only a freshly initialised spike decoder trains
# ===============================================================================

COLLAPSE_STD, COLLAPSE_CORR, COLLAPSE_AFTER = 0.5, 0.3, 1000    # T6: small AND uncorrelated output -> stop
ARMS = ("x", "x_hat", "y_hat")                                   # detection runs on each of these


# ------------------------------------------------------------------ building blocks

def lr_factor(step, c):
    """The learning rate at `step`, as a fraction of LR."""
    warm = min(1.0, (step + 1) / max(c.warmup_steps, 1))            # 1) linear warm-up
    if c.scheduler == "cosine":                                      # 2a) cosine down to 0 at MAX_STEPS
        progress = min(step / c.max_steps, 1)
        return warm * 0.5 * (1 + math.cos(math.pi * progress))
    if c.scheduler == "wsd":                                         # 2b) stable, then linear down to 0
        remaining = (c.max_steps - step) / (c.wsd_decay_frac * c.max_steps)   # over the last WSD_DECAY_FRAC
        return warm * min(1.0, max(0.0, remaining))
    return warm                                                      # 2c) constant


class EMA:
    """Running average of the weights: avg = decay·avg + (1 - decay)·current."""

    def __init__(self, model, decay):
        self.decay = decay
        self.model = copy.deepcopy(model).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for avg, current in zip(self.model.state_dict().values(), model.state_dict().values()):
            if avg.dtype.is_floating_point:
                avg.mul_(self.decay).add_(current.detach(), alpha=1 - self.decay)
            else:
                avg.copy_(current)


def load_checkpoint(path, ema=True, device="cpu"):
    """A trained codec (its EMA weights by default) and the run's config."""
    ck = torch.load(path, map_location=device, weights_only=False)
    model = Codec(CodecConfig(**ck["config"]["codec"]))
    model.load_state_dict(ck["ema" if ema else "model"])
    return model.to(device).eval(), ck["config"]


# ------------------------------------------------------------------ setup

def check_no_held_out():
    if HELD_OUT in TRAIN_PATIENTS:
        raise SystemExit(f"{HELD_OUT} is the held-out patient and never enters training")


def load_or_create_config(run, device):
    """A resumed run keeps ITS settings (from last.pt), not today's .env. -> codec, loss, train config, checkpoint."""
    if not (run / "last.pt").exists():
        tc = TrainConfig()
        if tc.init_checkpoint:                     # S12-A2: the architecture of the checkpoint it starts from
            src = torch.load(tc.init_checkpoint, map_location="cpu", weights_only=False)
            return CodecConfig(**src["config"]["codec"]), LossConfig(), tc, None
        return CodecConfig(), LossConfig(), tc, None
    ck = torch.load(run / "last.pt", map_location=device, weights_only=False)
    c = ck["config"]
    # fields removed since the checkpoint was written (e.g. grad_clip) are dropped
    train_fields = {k: v for k, v in c["train"].items() if k in TrainConfig.__dataclass_fields__}
    return CodecConfig(**c["codec"]), LossConfig(**c["loss"]), TrainConfig(**train_fields), ck


def init_from(model, tc):
    """S12-A2: the codec's weights from INIT_CHECKPOINT (EMA); with FREEZE_ENCODER the spike decoder keeps its fresh
    initialisation, so ŷ is learned again from the frozen tokens alone."""
    src = torch.load(tc.init_checkpoint, map_location="cpu", weights_only=False)
    fresh = {k: v.clone() for k, v in model.state_dict().items() if k.startswith("spike_decoder.")}
    model.load_state_dict(src["ema"])
    if tc.freeze_encoder:
        model.load_state_dict(fresh, strict=False)


def freeze(model, tc):
    """S12-A2: everything but the spike decoder stops training (and AdamW's weight decay cannot touch it)."""
    if tc.freeze_encoder:
        for name, p in model.named_parameters():
            p.requires_grad_(name.startswith("spike_decoder."))


def build_optimizer(model, tc):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=tc.lr, weight_decay=tc.weight_decay, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: lr_factor(step, tc))
    return opt, sched


def prepare_data(tc, start_step, future=0):
    """Training loader (continues at start_step), the fixed validation windows and the monitoring segment.
    `future`: samples after each window, the non-causal ceiling's look-ahead (model.future)."""
    train_s, val_s = sessions("train", patients=TRAIN_PATIENTS), sessions("val")
    print(f"[data] {len(train_s)} train | {len(val_s)} val recordings")
    loader = train_loader(train_s, tc.burn_in, tc.window, tc.batch_size, tc.data_seed, start_step,
                          tc.num_workers, future)
    vloader = val_loader(val_s, tc.burn_in, tc.window, tc.batch_size, tc.val_windows, future)
    seg = monitor_segment(val_s, tc.burn_in, tc.monitor_seconds)
    return loader, vloader, seg


def init_wandb(exp_name, config, ck, model):
    wandb.init(project=env("WANDB_PROJECT", "tokenizer-v2"), entity=env("WANDB_ENTITY", None) or None,
               group=env("WANDB_GROUP", None) or None, mode=env("WANDB_MODE", "online"), name=exp_name,
               id=ck["wandb_id"] if ck else None, resume="allow", config=config)
    wandb.summary["params"] = sum(p.numel() for p in model.parameters())


def past_only(batch, tc):
    """The batch without the non-causal look-ahead: x ends where the loss span ends."""
    return {**batch, "x": batch["x"][..., :tc.burn_in + tc.window]}


# ------------------------------------------------------------------ evaluation

@torch.no_grad()
def validate(model, loader, tc, lc, device, amp):
    """Mean loss terms and the window scores on the fixed validation windows."""
    model.eval()
    loss_sum, counts = defaultdict(float), defaultdict(float)
    std_sum, corr_sum, wave_r = 0.0, 0.0, []
    err_in, n_in, err_out, n_out = 0.0, 0, 0.0, 0      # |x_hat - x| inside / outside artifacts
    n_batches = 0
    for batch in loader:
        x, m = batch["x"].to(device), batch["m"].to(device)
        with torch.autocast(device.split(":")[0], torch.bfloat16, enabled=amp):
            x_hat, y_hat, _ = model(x, tc.burn_in)
        for k, v in losses(x_hat, y_hat, x[..., tc.burn_in:tc.burn_in + tc.window], m, lc).items():
            loss_sum[k] += float(v)

        s = window_scores(past_only(batch, tc), x_hat, y_hat)
        for arm in ARMS:
            for c in COUNTS:
                counts[f"{arm}_{c}"] += s[f"{arm}_{c}"]
        std_sum += s["std"]
        corr_sum += s["corr"]
        wave_r += s["wave_r"]
        err_in, n_in = err_in + s["l1_in"][0], n_in + s["l1_in"][1]
        err_out, n_out = err_out + s["l1_out"][0], n_out + s["l1_out"][1]
        n_batches += 1
    model.train()

    out = {f"val/{k}": v / n_batches for k, v in loss_sum.items()}
    for arm in ARMS:
        for k, v in prf(*(counts[f"{arm}_{c}"] for c in COUNTS)).items():
            out[f"val_detect/{k}_{arm}"] = v
    out["val_wave/r_median"] = float(np.median(wave_r)) if wave_r else float("nan")
    out["health/output_std"] = std_sum / n_batches
    out["health/corr"] = corr_sum / n_batches
    out["health/l1_inside_artifacts"] = err_in / max(n_in, 1)
    out["health/l1_outside_artifacts"] = err_out / max(n_out, 1)
    out["health/pct_flagged"] = 100 * n_in / max(n_in + n_out, 1)
    return out


def monitor_segment(val_sessions, burn_in, seconds):
    """The fixed segment of the monitoring plots: the val recording with the most Good spikes."""
    s = max(val_sessions, key=lambda s: len(s.spike_t))
    n = int(seconds * HZ) // 1920 * 1920           # a multiple of every rate arm (up to 15.6 Hz)
    in_seg = (s.spike_t >= burn_in) & (s.spike_t < burn_in + n)
    ref = {}                                       # Good spike times per electrode row, from the segment start
    for r in np.unique(s.spike_row):
        t = np.sort(s.spike_t[(s.spike_row == r) & in_seg] - burn_in)
        if len(t):
            ref[int(r)] = t
    busiest = max(ref, key=lambda r: len(ref[r])) if ref else 0
    return {"session": s, "x_full": s.x(0, burn_in + n), "art": s.art(burn_in, n), "ref": ref,
            "sigma_bp": s.sigma_bp, "ch": busiest, "burn_in": burn_in}


def monitor_scores(model, seg):
    """Reconstruct the monitoring segment (kept in `seg` for the plots) and score its LFP bands and MUA."""
    x_hat, y_hat, _ = model.reconstruct(seg["x_full"])
    b = seg["burn_in"]
    seg.update(x=seg["x_full"][:, b:], x_hat=x_hat[:, b:], y_hat=y_hat[:, b:])
    x1k, h1k = down_1k(seg["x"]), down_1k(seg["x_hat"])
    seg["lfp_r"] = {k: pearson(lfp_band(x1k, *f), lfp_band(h1k, *f)) for k, f in LFP_BANDS.items()}
    seg["mua_r"] = pearson(mua(seg["x"]), mua(seg["x_hat"]))
    scores = {f"val_lfp/r_{k}": float(np.median(v)) for k, v in seg["lfp_r"].items()}
    scores["val_lfp/r_mua"] = float(np.median(seg["mua_r"]))
    return scores


def is_collapsed(val, step):
    """T6: after COLLAPSE_AFTER steps, an output that is both small and uncorrelated with x."""
    return (step >= COLLAPSE_AFTER and val["health/output_std"] < COLLAPSE_STD
            and val["health/corr"] < COLLAPSE_CORR)


# ------------------------------------------------------------------ saving and logging

def save_checkpoint(path, model, ema, step, config, opt=None, sched=None, history=None):
    """Weights + EMA; with `opt` also everything a true resume needs. Written to .tmp first, then renamed."""
    state = {"model": model.state_dict(), "ema": ema.model.state_dict(), "step": step, "config": config}
    if opt is not None:
        state.update(opt=opt.state_dict(), sched=sched.state_dict(), history=history, wandb_id=wandb.run.id)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def log_figure(run, name, make, step):
    try:                                           # a plot must never stop a run
        fig = make()
    except Exception as e:
        print(f"[plots] {name} failed: {e!r}")
        return
    if fig is not None:
        fig.savefig(run / "plots" / f"{name}_{step:06d}.png", dpi=90, bbox_inches="tight")
        wandb.log({f"plots/{name}": wandb.Image(fig)}, step=step)
        plots.plt.close(fig)


def log_train(running, lr, steps_per_s, step, history):
    """The mean of each loss term since the last log."""
    loss_means = {k: float(np.mean(v)) for k, v in running.items() if k.startswith("loss")}
    wandb.log({**{f"train/{k}": v for k, v in loss_means.items()},
               "health/lr": lr, "health/steps_per_s": steps_per_s}, step=step)
    history["train"].append({"step": step, **loss_means})


# ------------------------------------------------------------------ the loop

def train(exp_name, device=None):
    check_no_held_out()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    run = RUNS_DIR / exp_name
    (run / "plots").mkdir(parents=True, exist_ok=True)

    ## Settings
    cc, lc, tc, ck = load_or_create_config(run, device)
    # the dataset builds m with THIS run's alpha and source (also on resume)
    os.environ["MASK_ALPHA"] = str(lc.mask_alpha)
    os.environ["MASK_SOURCE"] = lc.mask_source
    config = {"exp_name": exp_name, "codec": asdict(cc), "loss": asdict(lc), "train": asdict(tc),
              "split_hash": split_hash(), "git": git_hash(), "train_patients": list(TRAIN_PATIENTS)}

    ## Model, EMA, optimizer (a resumed run restores all of them)
    torch.manual_seed(tc.seed)
    model = Codec(cc).to(device)
    if tc.burn_in % model.ratio or tc.window % model.ratio:
        raise SystemExit(f"BURN_IN and WINDOW must be multiples of {model.ratio}")
    if tc.freeze_encoder and not tc.init_checkpoint:
        raise SystemExit("FREEZE_ENCODER needs INIT_CHECKPOINT (the codec whose tokens are frozen)")
    if tc.init_checkpoint and not ck:
        init_from(model, tc)
    freeze(model, tc)
    ema = EMA(model, tc.ema_decay)
    opt, sched = build_optimizer(model, tc)
    step, history = 0, {"train": [], "val": []}
    if ck:
        model.load_state_dict(ck["model"])
        ema.model.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        step, history = ck["step"], ck["history"]
    if step >= tc.max_steps:                       # an S8 leg chained after the run already finished
        print(f"[train] {run} is already at step {step}")
        return run
    (run / "config.json").write_text(json.dumps(config, indent=2))
    print(f"[train] {run}\n[train] {model.describe()}\n[train] split {config['split_hash']} | git {config['git']}"
          + (f" · resumed at step {step}" if ck else ""))

    ## Data and W&B
    loader, vloader, seg = prepare_data(tc, step, model.future)
    init_wandb(exp_name, config, ck, model)
    amp = tc.amp and device.startswith("cuda")

    pbar = tqdm(total=tc.max_steps, initial=step, unit="step", mininterval=30)
    running, t_last = defaultdict(list), time.time()
    model.train()
    for batch in loader:
        if step == 0:
            log_figure(run, "model_input", lambda: plots.model_input(past_only(batch, tc), tc.burn_in), step)

        # forward
        x, m = batch["x"].to(device, non_blocking=True), batch["m"].to(device, non_blocking=True)
        with torch.autocast(device.split(":")[0], torch.bfloat16, enabled=amp):
            x_hat, y_hat, _ = model(x, tc.burn_in)
        terms = losses(x_hat, y_hat, x[..., tc.burn_in:tc.burn_in + tc.window], m, lc)

        # backward and update
        opt.zero_grad(set_to_none=True)
        terms["loss_total"].backward()
        opt.step()
        sched.step()
        ema.update(model)
        step += 1

        # progress
        for k, v in terms.items():
            running[k].append(float(v.detach()))
        pbar.set_postfix(loss=running["loss_total"][-1], refresh=False)
        pbar.update()

        if step % tc.log_interval == 0:
            log_train(running, sched.get_last_lr()[0], tc.log_interval / (time.time() - t_last), step, history)
            running, t_last = defaultdict(list), time.time()

        if step % tc.eval_interval == 0 or step == tc.max_steps:
            # validate, plot, save
            val = validate(model, vloader, tc, lc, device, amp)
            val.update(monitor_scores(model, seg))
            wandb.log(val, step=step)
            history["val"].append({"step": step, **{k[4:]: v for k, v in val.items() if k.startswith("val/")}})
            print(f"[val {step}] loss {val['val/loss_total']:.4f} | F1 x_hat {val['val_detect/f1_x_hat']:.3f} "
                  f"y_hat {val['val_detect/f1_y_hat']:.3f} (x {val['val_detect/f1_x']:.3f}) | "
                  f"std {val['health/output_std']:.2f} corr {val['health/corr']:.2f}", flush=True)
            for name, make in plots.monitor_figures(seg, history).items():
                log_figure(run, name, make, step)

            save_checkpoint(run / "last.pt", model, ema, step, config, opt, sched, history)
            (run / "history.json").write_text(json.dumps(history))
            if step % tc.save_every == 0:
                save_checkpoint(run / f"step_{step:06d}.pt", model, ema, step, config)

            if is_collapsed(val, step):
                wandb.finish(exit_code=1)
                raise RuntimeError(f"collapsed at step {step}: output std {val['health/output_std']:.2f}, "
                                   f"corr {val['health/corr']:.2f} (T6)")

        if step >= tc.max_steps:
            break
    pbar.close()
    wandb.finish()
    return run
