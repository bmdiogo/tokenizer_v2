"""The split (S0): every recording gets a split and a group. Writes info/splits.csv — never edit it.

Rules (asserted by `check_integrity`):
  * the unit is the recording DAY: two windows of one day share electrode state, noise and units,
    so a day never crosses train / val / test (no leakage)
  * P4, the latest-recorded participant (2024), is held out entirely: test, group p4_unseen
  * P2 and P3 are cut chronologically by day: train | next 15 % -> val (p2_late, p3_late)
    | last 15 % -> test (p2_final, p3_final)
  * `cached` = the recordings that get cached: a date-stratified subset per split, only on NS5
    widths whose electrode map is verified (`buildable`)

    python -m utils.split                                  # corpus from DATA_PATH/<P>/table.csv, reads NS5 headers
    python -m utils.split --corpus old_splits.csv          # reuse a previous listing (same columns)
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from utils.config import DATA_PATH, HELD_OUT, INFO_DIR, split_hash

PATIENTS = ("P2", "P3", "P4")
VAL_FRAC = TEST_FRAC = 0.15
CACHE_BUDGET = {"train": 110, "val": 24, "test": 60}       # recordings cached per split
COLUMNS = ["patient", "day", "name", "signal", "spikes", "split", "group", "cached",
           "size_gb", "n_channels", "buildable"]


def read_corpus(data_path):
    """One row per recording, from each patient's table.csv, with the NS5 channel count."""
    import neo
    rows = []
    for p in PATIENTS:
        table = pd.read_csv(Path(data_path) / p / "table.csv")
        for sig, spk in zip(table.get("signals", table.get("signal")), table["spikes"]):
            sig, spk = str(sig).strip(), str(spk).strip()
            name = Path(spk).stem.replace("_spikes", "")
            reader = neo.io.BlackrockIO(str(Path(data_path) / p / sig))
            reader.parse_header()
            rows.append({"patient": p, "day": "-".join(re.search(r"(\d{4})_(\d{2})_(\d{2})", name).groups()),
                         "name": name, "signal": f"{p}/{sig}", "spikes": f"{p}/{spk}",
                         "size_gb": (Path(data_path) / p / sig).stat().st_size / 2 ** 30,
                         "n_channels": int(reader.signal_channels_count(0))})
    return pd.DataFrame(rows)


def assign_splits(df):
    split, group = {}, {}
    for patient, sub in df.groupby("patient"):
        days = sorted(sub["day"].unique())
        if patient == HELD_OUT:
            for d in days:
                split[patient, d], group[patient, d] = "test", f"{patient.lower()}_unseen"
            continue
        n_test, n_val = max(1, round(len(days) * TEST_FRAC)), max(1, round(len(days) * VAL_FRAC))
        for i, d in enumerate(days):
            if i >= len(days) - n_test:
                split[patient, d], group[patient, d] = "test", f"{patient.lower()}_final"
            elif i >= len(days) - n_test - n_val:
                split[patient, d], group[patient, d] = "val", f"{patient.lower()}_late"
            else:
                split[patient, d], group[patient, d] = "train", ""
    keys = list(zip(df["patient"], df["day"]))
    return df.assign(split=[split[k] for k in keys], group=[group[k] for k in keys])


def verified_widths(patient):
    raw = json.loads((INFO_DIR / f"electrode_to_row_{patient}.json").read_text())
    return {int(raw["channel_count"]), *map(int, raw.get("verified_widths", []))}


def stratified_days(counts, budget):
    """The most days, evenly spread over the date range, whose recordings fit in `budget`."""
    days = sorted(counts)
    best = []
    for k in range(1, len(days) + 1):
        pick = [days[i] for i in sorted(set(np.linspace(0, len(days) - 1, k).round().astype(int)))]
        if sum(counts[d] for d in pick) > budget:
            break
        best = pick
    return best


def mark_cached(df):
    df = df.assign(buildable=[int(n in verified_widths(p)) for p, n in zip(df["patient"], df["n_channels"])])
    chosen = set()
    for split, sub in df[df["buildable"] == 1].groupby("split"):
        per_patient = {p: s.groupby("day").size().to_dict() for p, s in sub.groupby("patient")}
        for p, counts in per_patient.items():
            chosen |= {(p, d) for d in stratified_days(counts, CACHE_BUDGET[split] // len(per_patient))}
    cached = [(p, d) in chosen and b == 1 for p, d, b in zip(df["patient"], df["day"], df["buildable"])]
    return df.assign(cached=np.array(cached, int))


def check_integrity(df):
    assert (df.groupby(["patient", "day"])["split"].nunique() == 1).all(), "a day is in two splits"
    assert (df.loc[df["patient"] == HELD_OUT, "split"] == "test").all(), "the held-out patient leaked"
    last = df.groupby("patient")["day"].max()
    assert last[HELD_OUT] > last.drop(HELD_OUT).max(), "the held-out patient is not the latest"
    for p, sub in df[df["patient"] != HELD_OUT].groupby("patient"):
        by = {s: sorted(g["day"]) for s, g in sub.groupby("split")}
        assert max(by["train"]) < min(by["val"]) and max(by["val"]) < min(by["test"]), f"{p} not chronological"
    assert set(df["group"]) <= {"", "p2_late", "p3_late", "p2_final", "p3_final", "p4_unseen"}
    assert not ((df["cached"] == 1) & (df["buildable"] == 0)).any(), "a cached recording on an unverified width"
    assert not df["name"].duplicated().any(), "two recordings share a name"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", help="previous listing with patient, day, name, signal, spikes, size_gb, n_channels")
    ap.add_argument("--out", default=INFO_DIR / "splits.csv")
    a = ap.parse_args()
    df = pd.read_csv(a.corpus)[COLUMNS[:5] + ["size_gb", "n_channels"]] if a.corpus else read_corpus(DATA_PATH)
    df = mark_cached(assign_splits(df.sort_values(["patient", "day", "name"]).reset_index(drop=True)))
    check_integrity(df)
    df[COLUMNS].to_csv(a.out, index=False)
    print(df.groupby(["split", "group", "patient"]).agg(recordings=("name", "size"), days=("day", "nunique"),
                                                        cached=("cached", "sum")).to_string())
    print(f"wrote {a.out}  (split hash {split_hash()})")


if __name__ == "__main__":
    main()
