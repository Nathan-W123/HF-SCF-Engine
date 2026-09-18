"""Path bootstrap shared by the scripts in this directory."""

from __future__ import annotations

import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
MEDIA_DIR = PROJECT_ROOT / "media"
VALIDATION_DIR = PROJECT_ROOT / "validation"

for _d in (RESULTS_DIR, FIGURES_DIR, MEDIA_DIR):
    _d.mkdir(parents=True, exist_ok=True)

TRAJECTORY_FIELDS = (
    "t", "altitude", "radius", "velocity", "gamma", "psi", "lat", "lon",
    "downrange", "density", "mach", "q_dyn", "q_dot", "q_dot_dkr", "q_dot_exp315",
    "heat_load", "g_load", "wall_temperature", "drag_work", "cd", "cl",
)


def save_trajectory(result, path):
    """Write a :class:`TrajectoryResult` to a compressed ``.npz``."""
    import json

    import numpy as np

    arrays = {k: getattr(result, k) for k in TRAJECTORY_FIELDS}
    meta = {k: v for k, v in result.meta.items() if k != "raw_solution"}
    arrays["_meta_json"] = np.array(
        json.dumps({"termination": result.termination, **meta})
    )
    np.savez_compressed(path, **arrays)


def load_trajectory(path):
    """Read a ``.npz`` written by :func:`save_trajectory` into a namespace."""
    import json
    import types

    import numpy as np

    z = np.load(path, allow_pickle=False)
    ns = types.SimpleNamespace(**{k: z[k] for k in TRAJECTORY_FIELDS})
    ns.meta = json.loads(str(z["_meta_json"]))
    ns.termination = ns.meta.get("termination", "")
    return ns
