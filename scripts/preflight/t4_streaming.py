"""T4 · Causal and streamable: the causality tests, then the time per 8 ms frame.

    python -m scripts.preflight.t4_streaming
"""
import subprocess
import sys

import torch

from models.codec import Codec
from scripts.preflight import save
from utils.config import ROOT
from utils.metrics import stream_timing


def main():
    tests = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_model.py"], cwd=ROOT,
                           capture_output=True, text=True)
    print(tests.stdout[-600:])
    model = Codec()
    n = lambda m: sum(p.numel() for p in m.parameters())
    timing = {"cpu": stream_timing(model, n_frames=300, device="cpu")}
    if torch.cuda.is_available():
        timing["cuda"] = stream_timing(model, device="cuda")
    print(timing)
    save("t4", {"question": "causal and streamable", "tests_passed": tests.returncode == 0,
                "tests_tail": (tests.stdout + tests.stderr).strip().splitlines()[-1:],
                "params": {"encoder": n(model.encoder), "reconstruction_decoder": n(model.decoder),
                           "spike_decoder": n(model.spike_decoder)},
                "time_per_frame_ms": timing})


if __name__ == "__main__":
    main()
