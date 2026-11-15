"""T9 · Recipe sanity checks (Karpathy): catch silent failures before any long run (GPU, minutes).

  loss at init            output ≈ 0 -> ℒ_time ≈ mean |x|
  input-independent       x = 0 -> corr(x̂, x) ≈ 0 and F1(ŷ) ≈ 0 (also the collapsed state of T6)
  overfit 2 batches       train until corr(x̂, x) > 0.99 and F1(ŷ) >= 0.9 · F1(y) on those batches (plan:
                          "continue until ... F1(ŷ) ≈ 1"). F1(ŷ) ≈ 1 means ŷ reproduces y, and y itself only
                          reaches F1(y) through the T3 detector (0.845 on these batches: some Good spikes
                          never cross -5σ). --steps is a cap; the curve has y_scale = the least-squares
                          scale of ŷ on y inside m (0 = silent, 1 = the right amplitude)
  see what the model sees x_n, m and y for 3 windows -> experiments/t9_model_input.png
The gradient checks (channel independence, causality) are pytest: tests/test_model.py.

    python -m scripts.preflight.t9_sanity --steps 30000
"""
import argparse

import numpy as np
import torch

from models.codec import Codec
from scripts.preflight import save
from utils import plots
from utils.config import ROOT
from utils.dataset import Windows, sessions
from utils.dsp import bp
from utils.losses import LossConfig, losses
from utils.metrics import COUNTS, prf, window_scores
from utils.train import TrainConfig


def f1(scores, arm):
    return prf(*(sum(s[f"{arm}_{c}"] for s in scores) for c in COUNTS))["f1"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--eval-every", type=int, default=1000)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tc, lc = TrainConfig(), LossConfig()
    items = Windows(sessions("train"), tc.burn_in, tc.window, seed=7)
    batches = [{k: torch.stack([items[b * tc.batch_size + i][k] for i in range(tc.batch_size)]) for k in items[0]}
               for b in range(2)]
    plots.model_input(batches[0], tc.burn_in).savefig(ROOT / "experiments" / "t9_model_input.png", dpi=90)
    torch.manual_seed(0)
    model = Codec().to(dev)
    amp = dict(device_type=dev, dtype=torch.bfloat16, enabled=dev == "cuda")

    def run(batch, zero=False):
        x = batch["x"].to(dev)
        with torch.autocast(**amp):
            xh, yh, _ = model(x * 0 if zero else x, tc.burn_in)
        return xh, yh, losses(xh, yh, x[..., tc.burn_in:], batch["m"].to(dev), lc)

    targets = [b["m"] * bp(b["x"][..., tc.burn_in:]) for b in batches]           # y, the ideal ŷ
    f1_y_ideal = f1([window_scores(b, b["x"][..., tc.burn_in:], y) for b, y in zip(batches, targets)], "y_hat")

    def overfit():
        """corr, F1 and y_scale on the two batches, and whether they pass the bar."""
        with torch.no_grad():
            outs = [run(b)[:2] for b in batches]
        sc = [window_scores(b, xh, yh) for b, (xh, yh) in zip(batches, outs)]
        num = sum(float((b["m"] * yh.float().cpu() * y).sum()) for b, (_, yh), y in zip(batches, outs, targets))
        den = sum(float((b["m"] * y * y).sum()) for b, y in zip(batches, targets))
        o = {"corr": float(np.mean([s["corr"] for s in sc])), "f1_x": f1(sc, "x"), "f1_x_hat": f1(sc, "x_hat"),
             "f1_y_hat": f1(sc, "y_hat"), "f1_y_ideal": f1_y_ideal, "y_scale": num / max(den, 1e-12)}
        return o, o["corr"] > 0.99 and f1_y_ideal > 0 and o["f1_y_hat"] >= 0.9 * f1_y_ideal

    with torch.no_grad():
        init = {k: float(v) for k, v in run(batches[0])[2].items()}
    expected = float(batches[0]["x"][..., tc.burn_in:].abs().mean())
    opt = torch.optim.AdamW(model.parameters(), lr=tc.lr, betas=(0.9, 0.95))
    curve, passed_overfit, step = [], False, 0
    while step < a.steps and not passed_overfit:
        L = run(batches[step % 2])[2]
        opt.zero_grad()
        L["loss_total"].backward()
        opt.step()
        step += 1
        if step % a.eval_every == 0 or step == a.steps:
            fitted, passed_overfit = overfit()
            curve.append({"step": step, **{k: float(v.detach()) for k, v in L.items()}, **fitted})
            print(curve[-1], flush=True)
    with torch.no_grad():
        zero = [window_scores(b, *run(b, zero=True)[:2]) for b in batches]
    out = {"init": init, "init_expected_loss_time": expected, "curve": curve, "steps_run": step,
           "overfit": curve[-1] if curve else overfit()[0],
           "zero_input": {"corr": float(np.mean([s["corr"] for s in zero])), "f1_y_hat": f1(zero, "y_hat")}}
    out["passed"] = {"init": abs(init["loss_time"] - expected) < 0.5 * expected,
                     "overfit": passed_overfit,
                     "zero_input": abs(out["zero_input"]["corr"]) < 0.1 and out["zero_input"]["f1_y_hat"] < 0.05}
    print(out["passed"])
    save("t9", {"question": "recipe sanity checks", **out})


if __name__ == "__main__":
    main()
