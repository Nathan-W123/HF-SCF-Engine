"""Fast smoke checks mirroring the validation scripts (the full scripts run in
``make all``; these keep the pytest suite under the time budget)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "validation"))

from common import fit_circle, jsonable   # noqa: E402


def test_fit_circle_recovers_a_known_circle():
    th = np.linspace(0, 1.7 * np.pi, 400)
    pts = np.stack([120.0 + 55.0 * np.cos(th), -30.0 + 55.0 * np.sin(th)], axis=1)
    cx, cy, R, rms = fit_circle(pts)
    assert cx == pytest.approx(120.0, abs=1e-8)
    assert cy == pytest.approx(-30.0, abs=1e-8)
    assert R == pytest.approx(55.0, abs=1e-8)
    assert rms < 1e-8


def test_jsonable_handles_numpy():
    out = jsonable({"a": np.float64(1.5), "b": np.int64(3),
                    "c": np.array([1, 2]), "d": np.bool_(True), "e": (1, 2)})
    import json
    json.dumps(out)
    assert out["c"] == [1, 2] and out["d"] is True


def test_validation_scripts_are_importable():
    import validate_avoidance, validate_dynamics, validate_regression  # noqa: F401
    import validate_tracking, validate_turbulence                      # noqa: F401
    for m in (validate_dynamics, validate_tracking, validate_avoidance,
              validate_regression, validate_turbulence):
        assert callable(m.run)
