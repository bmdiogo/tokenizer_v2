"""Settings from `.env`. A shell variable or a `KEY=VALUE` argument wins over the file.

Config dataclasses use `E("KEY", default)`: the value is read when the config is CREATED, so an
override set before that (CLI, test, job script) always reaches it.
"""
import hashlib
import os
import subprocess
import sys
from dataclasses import field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=False)


def env(key, default):
    """os.environ[key] cast to the type of `default` (default if unset or empty)."""
    value = os.getenv(key, "").split("#")[0].strip()
    if value == "":
        return default
    if isinstance(default, bool):
        return value.lower() in ("1", "true", "yes", "on")
    return type(default)(value) if default is not None else value


def E(key, default):
    """Dataclass field whose default is read from the environment at construction time."""
    return field(default_factory=lambda: env(key, default))


def path(key, default):
    p = Path(env(key, default)).expanduser()
    return p if p.is_absolute() else ROOT / p


def ints(spec):
    """'8,5,3,2' -> (8, 5, 3, 2); '' -> ()."""
    return tuple(int(v) for v in str(spec).replace(" ", "").split(",") if v)


def apply_overrides(argv=None):
    """Move every KEY=VALUE argument into os.environ; return the other arguments."""
    rest = []
    for a in sys.argv[1:] if argv is None else argv:
        key, eq, value = a.partition("=")
        if eq and key.isupper():
            os.environ[key] = value
        else:
            rest.append(a)
    return rest


def split_hash():
    p = ROOT / "info" / "splits.csv"
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12] if p.exists() else None


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


HZ = env("HZ", 30000)
DATA_PATH = path("DATA_PATH", "data")
CACHE_DIR = path("CACHE_DIR", "cache")
RUNS_DIR = path("RUNS_DIR", "runs")
INFO_DIR = ROOT / "info"
TRAIN_PATIENTS = tuple(env("TRAIN_PATIENTS", "P2 P3").split())
HELD_OUT = "P4"
VAL_GROUPS = ("p2_late", "p3_late")
TEST_GROUPS = ("p2_final", "p3_final", "p4_unseen")
