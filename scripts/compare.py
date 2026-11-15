"""Score the runs of one plan step (S5, S6, S7, S9) at the SAME steps, against the T7 noise floor.

    python -m scripts.compare --stage s5 --runs s5-c1 s5-c2 ... --jobs 4
    python -m scripts.compare --stage s7 --runs s7-a-nospike ... --ref s5-c3      # deltas vs the reference
    python -m scripts.compare --stage s9 --runs s8-ref-s1 ... s8-ref-s5 --steps 50000 100000 150000 200000
    python -m scripts.compare --stage s10 --runs <frozen runs> --steps 200000 --groups p2_final p3_final p4_unseen --ref s8-ref
    python -m scripts.compare --stage s12 --runs s12-frozen-s1 s12-frozen-s2 s12-frozen-s3 --steps 50000 --groups p2_late p3_late p2_final p3_final p4_unseen

A run missing a checkpoint of the fixed set (collapsed, diverged) is listed as failed, not scored.
Each (run, step, group) is scored by scripts.evaluate on the validation groups (test groups only with --stage s10, or s12 for the frozen-encoder ŷ; once)
and cached in experiments/eval/<stage>_<run>_<step>_<group>.json. Runs named X-s1 ... X-s5 are one arm
(mean ± SD over seeds). For each group and metric, at the last step:
    within_floor_of_best   best - value <= the T7 floor of that metric and group (S5: kept)
    vs_ref                 value - ref; "no difference" when |delta| <= floor (S6 knee, S7 ablations)
Groups are never pooled. Writes experiments/<stage>.json.
"""
import argparse
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from scripts.preflight import save
from scripts.preflight.t7_noise_floor import METRICS, get
from utils.config import ROOT, RUNS_DIR, TEST_GROUPS, VAL_GROUPS
from utils.dsp import LFP_BANDS

ALL = {**METRICS, **{f"lfp_{b}_r": ("lfp_r_median", b) for b in LFP_BANDS}}   # LFP: no floor, reported (S6 knee)
KEEP_ON = ("f1_y_hat", "wave_r")                                               # S5 "done when"


def arm_of(run):
    return re.sub(r"-s\d+$", "", run)


def verdict(delta, floor):
    if floor is None:
        return None
    return "no difference" if abs(delta) <= floor else ("better" if delta > 0 else "worse")


def floor_at(t7g, step, ref_scores=None):
    """{metric: floor} AT the compared step: T7's seed spread there (the 15 k comparisons); beyond T7's steps,
    the spread of the reference arm's own seeds (S8: the 125 Hz runs are the full-horizon floor)."""
    at, out = str(step), {}
    for m in METRICS:
        if m in t7g and at in t7g[m].get("by_step", {}):
            out[m] = float(np.ptp(t7g[m]["by_step"][at]))
        elif ref_scores and len(ref_scores.get(m, [])) > 1:
            out[m] = float(np.ptp(ref_scores[m]))
        elif m in t7g:
            out[m] = t7g[m]["floor"]                           # T7's own floor is set by its 5 k spread
    return out


def judge(scores, floors, ref=None):
    """scores {arm: {metric: [value per seed]}} of ONE group at the last step -> per arm and metric."""
    out = {arm: {} for arm in scores}
    for m in ALL:
        mean = {a: float(np.mean(v[m])) for a, v in scores.items() if v[m]}
        if not mean:
            continue
        best, floor = max(mean.values()), floors.get(m)
        for a, mu in mean.items():
            v = scores[a][m]
            r = {"mean": mu, "sd": float(np.std(v, ddof=1)) if len(v) > 1 else None, "n": len(v),
                 "within_floor_of_best": None if floor is None else best - mu <= floor}
            if ref is not None and ref in mean:
                r["vs_ref"] = mu - mean[ref]
                r["vs_ref_verdict"] = verdict(mu - mean[ref], floor)
            out[a][m] = r
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--steps", nargs="+", type=int, default=[5000, 10000, 15000])
    ap.add_argument("--ref", default=None, help="reference run or arm (S6: 125 Hz, S7: the S5 winner)")
    ap.add_argument("--jobs", type=int, default=1, help="evaluations run in parallel")
    ap.add_argument("--groups", nargs="+", default=VAL_GROUPS, choices=VAL_GROUPS + TEST_GROUPS)
    a = ap.parse_args()
    if set(a.groups) & set(TEST_GROUPS) and a.stage not in ("s10", "s12"):    # s12: a new model, its step fixed before
        raise SystemExit("test groups are scored only by --stage s10 (or s12), once (S10)")
    runs = a.runs + ([a.ref] if a.ref and a.ref not in a.runs and (RUNS_DIR / a.ref).exists() else [])
    ckpt = lambda run, step: RUNS_DIR / run / f"step_{step:06d}.pt"
    failed = [r for r in runs if not all(ckpt(r, s).exists() for s in a.steps)]   # collapsed, diverged or unfinished
    runs = [r for r in runs if r not in failed]
    if failed:
        print("not scored (a checkpoint of the fixed set is missing):", failed, flush=True)
    path = lambda run, step, g: ROOT / "experiments" / "eval" / f"{a.stage}_{run}_{step}_{g}.json"

    def evaluate(cell):
        run, step, g = cell
        if not path(*cell).exists():
            subprocess.run([sys.executable, "-m", "scripts.evaluate", str(ckpt(run, step)),
                            "--group", g, "--out", str(path(*cell))], cwd=ROOT, check=True)

    with ThreadPoolExecutor(a.jobs) as pool:
        list(pool.map(evaluate, [(r, s, g) for g in a.groups for r in runs for s in a.steps]))
    t7 = json.loads((ROOT / "experiments" / "t7.json").read_text())["groups"]
    ref = arm_of(a.ref) if a.ref else None
    doc = {"question": f"{a.stage}: runs at the same steps vs the T7 floor", "runs": runs, "failed": failed, "steps": a.steps,
           "ref": ref, "groups": {}}
    for g in a.groups:
        by_step = {}
        for s in a.steps:
            scores = {}
            for run in runs:
                d = json.loads(path(run, s, g).read_text())
                for m, p in ALL.items():
                    v = get(d, p)
                    scores.setdefault(arm_of(run), {k: [] for k in ALL})[m] += [] if v is None else [v]
                ceiling = d["detect_x"]["f1"]
            by_step[s] = scores
        floors = floor_at(t7.get(g, {}), a.steps[-1], by_step[a.steps[-1]].get(ref))
        last = judge(by_step[a.steps[-1]], floors, ref)
        trend = {arm: {m: [float(np.mean(by_step[s][arm][m])) if by_step[s][arm][m] else None for s in a.steps]
                       for m in KEEP_ON} for arm in last}
        doc["groups"][g] = {"f1_ceiling": ceiling, "floors": floors, "last_step": last, "trend": trend}
        for arm, r in last.items():
            print(g, arm, " ".join(f"{m} {r[m]['mean']:.3f}" + (f" ({r[m]['vs_ref_verdict']})" if r[m].get("vs_ref_verdict") else "")
                                   for m in ("f1_y_hat", "f1_x_hat", "wave_r", "mua_r") if m in r), flush=True)
    arms = list(doc["groups"][a.groups[0]]["last_step"])
    doc["kept"] = [arm for arm in arms if all(doc["groups"][g]["last_step"][arm].get(m, {}).get("within_floor_of_best")
                                              for g in a.groups for m in KEEP_ON)]
    print("kept (within the floor of the best on " + " and ".join(KEEP_ON) + " in every group):", doc["kept"])
    save(a.stage, doc)


if __name__ == "__main__":
    main()
