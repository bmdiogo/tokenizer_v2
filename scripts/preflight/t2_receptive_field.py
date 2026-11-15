"""T2 · Receptive field, burn-in B and peak memory per rate arm (GPU for the memory).

RF = how many past input samples the LAST output sample depends on (gradient of x̂[-1] w.r.t. a
60 000-sample input). B = RF rounded up to whole tokens. Memory = one training step at the batch
size of .env with W = B + L.

    python -m scripts.preflight.t2_receptive_field
"""
import torch

from models.codec import Codec, CodecConfig
from scripts.preflight import save
from utils.config import env
from utils.losses import LossConfig, losses

ARMS = {"125 Hz": ("8,5,3,2", "32,64,128,256,512", "1,3,9|1,3,9|1,3|1"),
        "62.5 Hz": ("8,5,3,2,2", "32,64,128,256,512,512", "1,3,9|1,3,9|1,3|1|"),
        "31.3 Hz": ("8,5,3,2,2,2", "32,64,128,256,512,512,512", "1,3,9|1,3,9|1,3|1||"),
        "15.6 Hz": ("8,5,3,2,2,2,2", "32,64,128,256,512,512,512,512", "1,3,9|1,3,9|1,3|1|||")}


def receptive_field(model, n=60_000):
    x = torch.zeros(1, 1, n // model.ratio * model.ratio, requires_grad=True)
    model(x)[0][0, 0, -1].backward()
    return x.shape[-1] - int(torch.nonzero(x.grad[0, 0]).min())


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    L, batch = env("WINDOW", 7680), env("BATCH_SIZE", 4)
    out = {}
    for name, (s, w, d) in ARMS.items():
        model = Codec(CodecConfig(strides=s, widths=w, dilations=d)).to(dev)
        rf = receptive_field(model.cpu())
        B = -(-rf // model.ratio) * model.ratio
        row = {"strides": s, "ratio": model.ratio, "params": sum(p.numel() for p in model.parameters()),
               "receptive_field": rf, "rf_ms": rf / 30, "burn_in": B}
        if dev == "cuda":
            model.to(dev)
            torch.cuda.reset_peak_memory_stats()
            x = torch.randn(batch, 96, B + L, device=dev)
            with torch.autocast("cuda", torch.bfloat16):
                xh, yh, _ = model(x, B)
            losses(xh, yh, x[..., B:], torch.zeros_like(xh), LossConfig())["loss_total"].backward()
            row["peak_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
            del x, xh, yh
        out[name] = row
        print(name, row, flush=True)
    save("t2", {"question": "receptive field, burn-in and memory per arm", "window": L, "batch": batch, "arms": out})


if __name__ == "__main__":
    main()
