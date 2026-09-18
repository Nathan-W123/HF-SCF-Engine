"""Shared helpers for the validation scripts."""

from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))


def jsonable(o):
    if isinstance(o, dict):
        return {k: jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    return o


def write(name: str, payload: dict) -> str:
    path = os.path.join(HERE, f"{name}.json")
    with open(path, "w") as fh:
        json.dump(jsonable(payload), fh, indent=2, sort_keys=True)
    return path


def report(name: str, payload: dict) -> None:
    path = write(name, payload)
    checks = payload.get("checks", {})
    n_pass = sum(1 for v in checks.values() if v)
    print(f"[{name}] {n_pass}/{len(checks)} checks passed -> {path}")
    for k, v in checks.items():
        print(f"    {'PASS' if v else 'FAIL'}  {k}")


def fit_circle(pts: np.ndarray):
    """Algebraic least-squares circle fit; returns (cx, cy, R, rms_residual)."""
    x, y = pts[:, 0], pts[:, 1]
    A = np.stack([2 * x, 2 * y, np.ones_like(x)], axis=1)
    b = x ** 2 + y ** 2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0], sol[1]
    R = np.sqrt(max(sol[2] + cx ** 2 + cy ** 2, 0.0))
    r = np.hypot(x - cx, y - cy)
    return float(cx), float(cy), float(R), float(np.sqrt(np.mean((r - R) ** 2)))
