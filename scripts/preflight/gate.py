"""Gate before the long runs (T7 and later): exits non-zero unless T9 passed and T6 shows the input
trains (S3). A SLURM job chained with --dependency=afterok on this never starts after a failure.

    python -m scripts.preflight.gate
"""
import json
import sys

from utils.config import ROOT


def main():
    t6 = json.loads((ROOT / "experiments" / "t6.json").read_text())
    t9 = json.loads((ROOT / "experiments" / "t9.json").read_text())
    problems = [f"T9 {k} failed" for k, ok in t9["passed"].items() if not ok]
    if not t6["s3_input_passes"]:
        problems.append(f"T6/S3: the {t6['band']} input does not escape at 125 Hz with the default weights")
    print("GATE " + ("FAILED: " + "; ".join(problems) if problems else "PASSED"))
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
