"""Train the codec. Every setting comes from .env; KEY=VALUE arguments override it for this run.

    python -m scripts.train --exp-name t7-s1 SEED=1 DATA_SEED=11 MAX_STEPS=15000 WANDB_GROUP=t7
    python -m scripts.train --exp-name sanity MAX_STEPS=200 EVAL_INTERVAL=50 VAL_WINDOWS=16

Running an existing exp-name again resumes it from <RUNS_DIR>/<exp-name>/last.pt.
"""
import argparse

from utils.config import apply_overrides

if __name__ == "__main__":
    rest = apply_overrides()                      # before any config is created
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp-name", required=True)
    ap.add_argument("--device", default=None)
    a = ap.parse_args(rest)
    from utils.train import train
    train(a.exp_name, a.device)
