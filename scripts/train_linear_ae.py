"""Train the linear autoencoder baseline on the codec's data. Settings come from .env; KEY=VALUE overrides.

    python -m scripts.train_linear_ae --exp-name lae-125-s1 SEED=1 DATA_SEED=11
    python -m scripts.train_linear_ae --exp-name lae-625-s1 SEED=1 DATA_SEED=11 STRIDES=8,5,3,2,2 BURN_IN=12480

Same recordings, window draws and optimiser as the codec; plain MSE on the frames of the loss span. Needs
scripts.pca_stats at the same rate: the convergence check compares with PCA on the validation windows.
Writes <RUNS_DIR>/<exp-name>/ as a codec run does; running an existing exp-name again resumes it.
"""
import argparse
import json
from dataclasses import asdict
from math import prod
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as nnf

from models.baselines import LinearAE, PCACodec
from utils.config import (
    RUNS_DIR,
    TRAIN_PATIENTS,
    apply_overrides,
    env,
    git_hash,
    ints,
    split_hash,
)
from utils.dataset import sessions, train_loader, val_loader
from utils.train import EMA, TrainConfig, build_optimizer, check_no_held_out


def save(path, **state):
    torch.save(state, str(path) + ".tmp")
    Path(str(path) + ".tmp").replace(path)


def frames_of(batch, tc, F):
    return batch["x"][..., tc.burn_in:tc.burn_in + tc.window].reshape(-1, F)


@torch.no_grad()
def val_mse(fn, loader, tc, F):
    """Mean squared error of frames -> frames over the fixed validation windows."""
    err, n = 0.0, 0
    for batch in loader:
        f = frames_of(batch, tc, F)
        err, n = err + float(((fn(f) - f) ** 2).sum()), n + f.numel()
    return err / n


def train(exp_name, device=None):
    check_no_held_out()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    F, K, tc = prod(ints(env("STRIDES", "8,5,3,2"))), env("LATENT_DIM", 32), TrainConfig()
    if tc.burn_in % F or tc.window % F:
        raise SystemExit(f"BURN_IN and WINDOW must be multiples of {F}")
    pca = PCACodec.from_stats(np.load(RUNS_DIR / f"pca_stats_F{F}.npz"), K)
    run = RUNS_DIR / exp_name
    run.mkdir(parents=True, exist_ok=True)
    config = {"exp_name": exp_name, "linear_ae": {"F": F, "K": K}, "train": asdict(tc), "split_hash": split_hash(),
              "git": git_hash(), "train_patients": list(TRAIN_PATIENTS)}

    torch.manual_seed(tc.seed)
    model = LinearAE(F, K).to(device)
    ema = EMA(model, tc.ema_decay)
    opt, sched = build_optimizer(model, tc)
    step, history = 0, {"train": [], "val": []}
    if (run / "last.pt").exists():
        ck = torch.load(run / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        ema.model.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        step, history = ck["step"], ck["history"]
    if step >= tc.max_steps:
        print(f"[train] {run} is already at step {step}")
        return run
    (run / "config.json").write_text(json.dumps(config, indent=2))

    train_s, val_s = sessions("train", patients=TRAIN_PATIENTS), sessions("val")
    loader = train_loader(train_s, tc.burn_in, tc.window, tc.batch_size, tc.data_seed, step, tc.num_workers)
    vloader = val_loader(val_s, tc.burn_in, tc.window, tc.batch_size, tc.val_windows)
    pca_mu, pca_u = torch.from_numpy(pca.mu), torch.from_numpy(pca.U)
    mse_pca = val_mse(lambda f: ((f - pca_mu) @ pca_u.T) @ pca_u + pca_mu, vloader, tc, F)
    print(f"[train] {run} F {F} K {K} from step {step} | {len(train_s)} train recordings | val MSE PCA {mse_pca:.5f}", flush=True)

    model.train()
    for batch in loader:
        f = frames_of(batch, tc, F).to(device, non_blocking=True)
        loss = nnf.mse_loss(model(f), f)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        ema.update(model)
        step += 1
        if step % tc.log_interval == 0:
            history["train"].append({"step": step, "loss": float(loss.detach())})
        if step % tc.eval_interval == 0 or step == tc.max_steps:
            mse = val_mse(lambda v: ema.model(v.to(device)).cpu(), vloader, tc, F)
            history["val"].append({"step": step, "mse": mse, "mse_pca": mse_pca, "ratio": mse / mse_pca})
            print(f"[val {step}] MSE {mse:.5f} | PCA {mse_pca:.5f} | ratio {mse / mse_pca:.4f}", flush=True)
            weights = dict(model=model.state_dict(), ema=ema.model.state_dict(), step=step, config=config)
            save(run / "last.pt", **weights, opt=opt.state_dict(), sched=sched.state_dict(), history=history)
            (run / "history.json").write_text(json.dumps(history))
            if step % tc.save_every == 0:
                save(run / f"step_{step:06d}.pt", **weights)
        if step >= tc.max_steps:
            break
    print("RESULT " + json.dumps(history["val"][-1]))
    return run


if __name__ == "__main__":
    rest = apply_overrides()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp-name", required=True)
    ap.add_argument("--device", default=None)
    a = ap.parse_args(rest)
    train(a.exp_name, a.device)
