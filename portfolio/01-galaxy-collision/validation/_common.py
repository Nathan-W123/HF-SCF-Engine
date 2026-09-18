"""Shared plumbing for the validation scripts."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
VALDIR = ROOT / "validation"
RESULTS = ROOT / "results"
DATA = ROOT / "data"


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    return o


def write_result(name, payload):
    VALDIR.mkdir(parents=True, exist_ok=True)
    path = VALDIR / f"{name}.json"
    path.write_text(json.dumps(_jsonable(payload), indent=2) + "\n")
    return path


def report(name, checks, extra=None):
    """``checks``: list of (label, value, threshold_text, passed)."""
    print(f"\n=== {name} ===")
    all_ok = True
    for label, value, thr, ok in checks:
        all_ok &= bool(ok)
        flag = "PASS" if ok else "FAIL"
        print(f"  [{flag}] {label:<46s} {value:<22s} (threshold: {thr})")
    payload = {"name": name, "passed": bool(all_ok),
               "checks": [{"label": l, "value": v, "threshold": t,
                           "passed": bool(o)} for l, v, t, o in checks]}
    if extra:
        payload.update(extra)
    write_result(name, payload)
    print(f"  -> {'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'}")
    return all_ok, payload
