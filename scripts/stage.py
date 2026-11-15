"""The training runs of one stage of the experiment plan (S5-S8, S12), one line each: `name<TAB>EXTRA`.

    python -m scripts.stage s5                        # LR x scheduler search (+ λ_bg), 8 runs
    python -m scripts.stage s6                        # 62.5 / 31.3 / 15.6 Hz, each at LR x {0.5, 1, 2}
    python -m scripts.stage s7                        # one-factor ablations at 125 Hz
    python -m scripts.stage s8 --finalists s5-c3 s6-31hz-lr1
    python -m scripts.stage s12                       # frozen-encoder spike decoders on the S10 125 Hz tokens

The runs are written to experiments/<stage>_arms.json BEFORE they start, so the draw is fixed before
any number is seen (S0). cluster/stage.sh submits them. Every other setting comes from the cluster's .env:
after each stage the chosen values (LR, scheduler, ...) go into .env, and the next stage builds on them.
15 k-step stages use seed 1 / data order 11 (= t7-s1): the T7 noise floor applies to their differences.
"""
import argparse
import json

import numpy as np

from scripts.preflight.t2_receptive_field import ARMS
from utils.config import ROOT, env

SHORT = "SEED=1 DATA_SEED=11 MAX_STEPS=15000"
BURN_IN = {"62.5 Hz": 12480, "31.3 Hz": 15360, "15.6 Hz": 21120}         # T2


def s5(n=8, seed=5):
    """Random search (S5): LR log-U(5e-5, 2e-3) · weight decay · scheduler · λ_bg. λ_STFT stays 0 (T9)."""
    rng = np.random.default_rng(seed)
    runs = {}
    for i in range(1, n + 1):
        lr = float(np.exp(rng.uniform(np.log(5e-5), np.log(2e-3))))
        wd = rng.choice(["0", "1e-4", "1e-3", "1e-2"])
        sched = rng.choice(["cosine", "constant", "wsd"])
        bg = rng.choice(["0.1", "0.3", "1.0"])
        runs[f"s5-c{i}"] = f"LR={lr:.2e} WEIGHT_DECAY={wd} SCHEDULER={sched} LAMBDA_BG={bg}"
    return runs


def s6():
    """Rate ablation (S6): one ×2 stage appended per halving, B from T2, a small LR check per rate."""
    lr = env("LR", 1e-4)
    runs = {}
    for rate, b in BURN_IN.items():
        strides, widths, dils = ARMS[rate]
        for f in (0.5, 1, 2):
            runs[f"s6-{rate.split()[0].replace('.', '')}hz-lr{f:g}"] = (
                f"STRIDES={strides} WIDTHS={widths} DILATIONS={dils} BURN_IN={b} LR={lr * f:.2e}")
    return runs


def s7():
    """One-factor ablations at 125 Hz (S7). B (2 Hz - 6 kHz input) needs a second cache: not here."""
    return {"s7-a-nospike": "LAMBDA_SPK=0",                # A: does the ℒ_spike gradient through z cost x̂?
            "s7-c-stft005": "LAMBDA_STFT=0.05",            # C: STFT re-added (T6: the balancer ≈ a fixed λ)
            "s7-d-nowave": "LAMBDA_WAVE=0",                # D: ℒ_wave off
            "s7-d-crossings": "MASK_SOURCE=crossings",     # D: m from crossings on bp(x), no labels
            "s7-e-box": "MASK_ALPHA=0",                    # E: edges of m (0.25 = the reference)
            "s7-e-hann": "MASK_ALPHA=1"}


def s8(finalists, seeds=5):
    """Full runs (S8): each finalist's settings × 5 seeds (seed k, data order 10 + k), 200 k steps."""
    known = {}
    for f in sorted((ROOT / "experiments").glob("s[567]_arms.json")):
        known.update(json.loads(f.read_text())["runs"])
    runs = {}
    for name in finalists:
        extra = known.get(name, "") if name != "ref" else ""            # ref = the .env settings alone
        if name != "ref" and name not in known:
            raise SystemExit(f"{name} is not in experiments/s5-s7_arms.json")
        extra = " ".join(w for w in extra.split() if not w.startswith(("SEED=", "DATA_SEED=", "MAX_STEPS=", "WANDB_GROUP=")))
        for k in range(1, seeds + 1):
            runs[f"s8-{name}-s{k}"] = f"{extra} SEED={k} DATA_SEED={10 + k} MAX_STEPS=200000".strip()
    return runs


def s12(seeds=3):
    """S12-A2: a fresh spike decoder on the frozen tokens of s8-ref-s{k} (the S10 125 Hz models), 50 k steps,
    checkpoint 50 k fixed now: does ŷ need the encoder to be shaped for it?"""
    return {f"s12-frozen-s{k}": f"INIT_CHECKPOINT=runs/s8-ref-s{k}/step_200000.pt FREEZE_ENCODER=1 "
                                f"SEED={100 + k} DATA_SEED={110 + k} MAX_STEPS=50000" for k in range(1, seeds + 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["s5", "s6", "s7", "s8", "s12"])
    ap.add_argument("--finalists", nargs="+", default=[], help="s8: run names from s5-s7, or ref")
    a = ap.parse_args()
    runs = s8(a.finalists) if a.stage == "s8" else {"s5": s5, "s6": s6, "s7": s7, "s12": s12}[a.stage]()
    if a.stage not in ("s8", "s12"):
        runs = {k: f"{v} {SHORT}" for k, v in runs.items()}
    runs = {k: f"{v} WANDB_GROUP={a.stage}" for k, v in runs.items()}
    (ROOT / "experiments" / f"{a.stage}_arms.json").write_text(json.dumps({"stage": a.stage, "runs": runs}, indent=1))
    for k, v in runs.items():
        print(f"{k}\t{v}")


if __name__ == "__main__":
    main()
