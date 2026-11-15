"""Split rules, spike mask, artifact flag and the training windows."""
import json

import numpy as np
import pandas as pd
import pytest

from utils.config import INFO_DIR
from utils.dataset import Windows, fixed_windows, sessions
from utils.dsp import TUKEY, mask_window, spike_mask
from utils.split import assign_splits, check_integrity, mark_cached


def synthetic_corpus():
    """Two recordings per day: P2 and P3 over 2022-2023, P4 (held out) later, on each array's verified width."""
    rows = []
    for p, year, n_days in (("P2", 2022, 20), ("P3", 2023, 20), ("P4", 2024, 6)):
        width = json.loads((INFO_DIR / f"electrode_to_row_{p}.json").read_text())["channel_count"]
        for d in range(n_days):
            day = f"{year}-{1 + d // 28:02d}-{1 + d % 28:02d}"
            rows += [{"patient": p, "day": day, "name": f"{p}_{day}_{k}", "signal": "", "spikes": "",
                      "size_gb": 1.0, "n_channels": width} for k in range(2)]
    return pd.DataFrame(rows)


def test_the_split_rules_hold_on_a_synthetic_corpus():
    df = mark_cached(assign_splits(synthetic_corpus()))
    check_integrity(df)
    assert set(df.loc[df["patient"] == "P4", "group"]) == {"p4_unseen"}
    for p in ("P2", "P3"):                                      # train < val < test in time, 15 % each
        sub = df[df["patient"] == p]
        assert sub.groupby("split")["day"].nunique().to_dict() == {"test": 3, "train": 14, "val": 3}
    leak = df.copy()
    leak.loc[leak.index[0], "split"] = "test"                   # one recording of a training day moved to test
    with pytest.raises(AssertionError, match="a day is in two splits"):
        check_integrity(leak)


@pytest.mark.skipif(not (INFO_DIR / "splits.csv").exists(), reason="info/splits.csv is not distributed")
def test_the_local_split_follows_the_rules():
    df = pd.read_csv(INFO_DIR / "splits.csv", keep_default_na=False)
    check_integrity(df)
    assert set(df["group"]) == {"", "p2_late", "p3_late", "p2_final", "p3_final", "p4_unseen"}
    assert set(df.loc[df["patient"] == "P4", "group"]) == {"p4_unseen"}
    assert not df["group"].str.contains("legacy").any()


def test_mask_is_a_tukey_window_and_overlaps_take_the_max():
    one = spike_mask(np.array([0]), np.array([100]), 2, 0, 300)
    assert np.array_equal(one[0, 79:140], TUKEY) and np.isclose(one.sum(), TUKEY.sum())     # [t-21, t+40) only
    two = spike_mask(np.array([0, 0]), np.array([100, 110]), 2, 0, 300)
    expected = np.zeros(300, np.float32)
    expected[79:140] = TUKEY
    expected[89:150] = np.maximum(expected[89:150], TUKEY)
    assert np.array_equal(two[0], expected) and two.max() <= 1                   # max, not sum


def test_mask_alpha_sets_the_edges(monkeypatch):
    assert np.all(mask_window(0.0) == 1)                                           # box
    assert np.allclose(mask_window(1.0), np.hanning(61))                           # Hann
    assert np.array_equal(mask_window(), TUKEY)                                    # default 0.25
    monkeypatch.setenv("MASK_ALPHA", "0")
    one = spike_mask(np.array([0]), np.array([100]), 1, 0, 300)
    assert one.sum() == 61 and np.array_equal(one, spike_mask(np.array([0]), np.array([100]), 1, 0, 300, alpha=0))


def test_windows_are_normalised_deterministic_and_clear_m_on_artifacts(fake_cache):
    tr, va = sessions("train"), sessions("val")
    assert len(tr) == 2 and len(va) == 1
    w = Windows(tr, 256, 512, seed=3)
    a, b = w[7], w[7]
    assert all(np.array_equal(a[k], b[k]) for k in a)
    assert a["x"].shape == (4, 768) and a["m"].shape == (4, 512)
    assert abs(float(np.median(a["x"]))) < 0.3
    s = va[0]
    assert len(s.art_spans) and (s.art_spans[:, 0] == 0).any()                   # the rail run is flagged
    t0 = int(s.art_spans[0, 1])
    assert (s.m(t0, 200) * s.art(t0, 200)).sum() == 0
    assert len(fixed_windows(va, 256, 512, 5)) == 5
    assert len(s.bad_t) and len(s.spike_t) == 4 * len(s.bad_t)                    # Good and Bad kept apart
    assert not set(s.bad_unit) & set(s.spike_unit)
