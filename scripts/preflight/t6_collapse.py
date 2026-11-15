"""T6 (+ S3, S4) · Which loss weighting keeps x̂ alive, per rate, on one frozen batch of the cache.

150 AdamW steps on ONE fixed batch per variant (~2-3 min each on an H100), then the output std and
corr(x̂, x) on that batch. Collapse = small AND uncorrelated output; escape = corr > 0.85.
On the raw cache this also settles S3: raw passes if the default weighting (time only, λ_STFT 0) escapes at 125 Hz.

    python -m scripts.preflight.t6_collapse
"""
import dataclasses

import numpy as np
import torch

from models.codec import Codec, CodecConfig
from scripts.preflight import save
from scripts.preflight.t2_receptive_field import ARMS
from utils.config import env
from utils.dataset import Windows, sessions
from utils.losses import LossConfig, losses
from utils.metrics import pearson

BURN_IN = {"125 Hz": 11520, "62.5 Hz": 12480, "31.3 Hz": 15360, "15.6 Hz": 21120}      # T2 (15.6 Hz: S4 at every S6 rate)
VARIANTS = {"stft 0.05": dict(lambda_stft=0.05), "stft 0.1": dict(lambda_stft=0.1),
            "stft 0.2": dict(lambda_stft=0.2), "stft 1.0": dict(lambda_stft=1.0),
            "stft only": dict(lambda_time=0.0, lambda_wave=0.0, lambda_stft=1.0),
            "time only (default)": dict(lambda_stft=0.0)}


def fit(arm, loss_cfg, batch, steps, dev):
    torch.manual_seed(0)
    s, w, d = ARMS[arm]
    model = Codec(CodecConfig(strides=s, widths=w, dilations=d)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=env("LR", 1e-4), betas=(0.9, 0.95))
    B = BURN_IN[arm]
    x, m = batch["x"].to(dev), batch["m"].to(dev)
    x = x[..., x.shape[-1] - B - m.shape[-1]:]                  # this arm's burn-in in front of the same span
    amp = dict(device_type=dev, dtype=torch.bfloat16, enabled=dev == "cuda")
    for _ in range(steps):
        with torch.autocast(**amp):
            xh, yh, _ = model(x, B)
        opt.zero_grad()
        losses(xh, yh, x[..., B:], m, loss_cfg)["loss_total"].backward()
        opt.step()
    with torch.no_grad(), torch.autocast(**amp):
        xh = model(x, B)[0].float()
    return {"std": float(xh.std()), "corr": float(np.mean(pearson(xh.cpu().numpy(), x[..., B:].cpu().numpy())))}


def main(steps=150):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    items = Windows(sessions("train"), max(BURN_IN.values()), env("WINDOW", 7680), seed=123)
    batch = {k: torch.stack([items[i][k] for i in range(env("BATCH_SIZE", 4))]) for k in ("x", "m")}
    table = {}
    for arm in BURN_IN:
        for name, over in VARIANTS.items():
            r = fit(arm, dataclasses.replace(LossConfig(), **over), batch, steps, dev)
            r["escaped"] = r["corr"] > 0.85
            table.setdefault(arm, {})[name] = r
            print(f"{arm:>8} {name:<20} std {r['std']:.2f} corr {r['corr']:.3f}", flush=True)
    save("t6", {"question": "which loss term collapses the output (and S3: does the raw input train)",
                "band": env("INPUT_BAND", "raw"), "steps": steps,
                "input_std": float(batch["x"][..., -env("WINDOW", 7680):].std()), "table": table,
                "s3_input_passes": table["125 Hz"]["time only (default)"]["escaped"]})


if __name__ == "__main__":
    main()
