"""Pre-flight tests T1-T9. Each script writes experiments/tN.json."""
import json

import numpy as np

from utils.config import ROOT, git_hash, split_hash


def save(name, doc):
    out = ROOT / "experiments" / f"{name}.json"
    out.parent.mkdir(exist_ok=True)
    doc = {"split_hash": split_hash(), "git": git_hash(), **doc}
    out.write_text(json.dumps(doc, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o)))
    print(f"wrote {out}")
    return doc
