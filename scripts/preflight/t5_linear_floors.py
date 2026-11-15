"""T5 · Linear floors: PCA codec and low-rate sampling at the codec's budget (32 numbers per frame)
and below, next to the x ceiling, on each validation group. Runs scripts.evaluate per cell.

    python -m scripts.preflight.t5_linear_floors --jobs 4   # F = 240 / 480 / 960 / 1920 -> 4 / 2 / 1 / 0.5 kHz
"""
import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from scripts.preflight import save
from utils.config import ROOT, VAL_GROUPS

FRAMES = (240, 480, 960, 1920)


def run(source, group, frame):
    out = ROOT / "experiments" / "eval" / f"t5_{source}_F{frame}_{group}.json"
    if not out.exists():
        subprocess.run([sys.executable, "-m", "scripts.evaluate", source, "--group", group,
                        "--frame", str(frame), "--out", str(out)], cwd=ROOT, check=True)
    d = json.loads(out.read_text())
    return {"f1": d["detect_x_hat"]["f1"], "precision": d["detect_x_hat"]["precision"],
            "recall": d["detect_x_hat"]["recall"], "f1_ceiling": d["detect_x"]["f1"],
            "wave_r": d.get("waveform", {}).get("all", {}).get("r_median"),
            "sorting_acc": d.get("sorting_x_hat", {}).get("acc_median"),
            "mua_r": d["mua_r_median"], "lfp_r": d["lfp_r_median"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--groups", nargs="+", default=list(VAL_GROUPS))
    ap.add_argument("--jobs", type=int, default=1, help="evaluations run in parallel")
    a = ap.parse_args()
    cells = [(g, "identity", 240) for g in a.groups] + [(g, s, F) for g in a.groups for s in ("pca", "lowrate") for F in FRAMES]
    with ThreadPoolExecutor(a.jobs) as pool:                    # each cell is its own evaluate process
        results = list(pool.map(lambda c: run(c[1], c[0], c[2]), cells))
    table = {}
    for (g, src, F), r in zip(cells, results):
        key = "identity" if src == "identity" else f"{src}_{32 * 30 / F:g}kHz"
        table.setdefault(g, {})[key] = r
        print(g, key, f"F1 {r['f1']:.3f}", flush=True)
    save("t5", {"question": "linear floors at the codec budget and below", "k": 32, "frames": FRAMES,
                "groups": table})


if __name__ == "__main__":
    main()
