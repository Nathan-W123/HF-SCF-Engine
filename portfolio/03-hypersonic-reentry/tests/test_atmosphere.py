"""Unit tests for the USSA76 atmosphere implementation."""

import numpy as np
import pytest

from reentry.atmosphere import (
    ExponentialAtmosphere,
    USSA76,
    USSA76_REFERENCE_TABLE,
    VacuumAtmosphere,
    geometric_altitude,
    geopotential_altitude,
)
from reentry.constants import G0, GAMMA_AIR, R0_USSA76, R_AIR

EXACT = USSA76(tabulated=False)
FAST = USSA76(tabulated=True)


@pytest.mark.parametrize("row", USSA76_REFERENCE_TABLE)
def test_layer_boundaries_match_published_table(row):
    h_km, t_ref, p_ref, rho_ref = row
    z = float(geometric_altitude(h_km * 1e3))
    p, t = EXACT._exact(z)
    rho = p / (R_AIR * t)
    assert t == pytest.approx(t_ref, rel=1e-9)
    assert p == pytest.approx(p_ref, rel=1e-6)
    assert rho == pytest.approx(rho_ref, rel=5e-5)


def test_sea_level_state():
    assert EXACT.temperature(0.0) == pytest.approx(288.15, rel=1e-12)
    assert EXACT.pressure(0.0) == pytest.approx(101325.0, rel=1e-12)
    assert EXACT.density(0.0) == pytest.approx(1.2250, rel=1e-5)
    assert EXACT.sound_speed(0.0) == pytest.approx(
        np.sqrt(GAMMA_AIR * R_AIR * 288.15), rel=1e-12
    )


def test_tabulated_matches_exact():
    z = np.random.default_rng(0).uniform(0.0, 300e3, 50_000)
    for field in ("density", "pressure", "temperature"):
        a = np.asarray(getattr(EXACT, field)(z))
        b = np.asarray(getattr(FAST, field)(z))
        assert np.max(np.abs(b - a) / np.abs(a)) < 5e-6


def test_hydrostatic_balance_inside_layers():
    """dp/dz = -rho g(z) away from lapse-rate discontinuities."""
    z = np.array([5e3, 15e3, 25e3, 40e3, 60e3, 78e3, 95e3, 130e3, 200e3])
    dz = 5.0
    p_plus = np.asarray(EXACT.pressure(z + dz))
    p_minus = np.asarray(EXACT.pressure(z - dz))
    rho = np.asarray(EXACT.density(z))
    g = G0 * (R0_USSA76 / (R0_USSA76 + z)) ** 2
    resid = np.abs((p_plus - p_minus) / (2 * dz) + rho * g) / (rho * g)
    assert np.max(resid) < 1e-6


def test_monotonic_and_positive():
    z = np.linspace(0.0, 400e3, 5000)
    rho = np.asarray(EXACT.density(z))
    p = np.asarray(EXACT.pressure(z))
    assert np.all(rho > 0) and np.all(p > 0)
    assert np.all(np.diff(rho) < 0) and np.all(np.diff(p) < 0)


def test_geopotential_round_trip():
    z = np.linspace(0.0, 86e3, 1001)
    assert np.allclose(geometric_altitude(geopotential_altitude(z)), z, atol=1e-6)


def test_86_km_seam_is_small():
    below = EXACT.density(86e3 - 1.0)
    above = EXACT.density(86e3 + 1.0)
    assert abs(above - below) / below < 1e-3


def test_exponential_and_vacuum_models():
    e = ExponentialAtmosphere(rho0=1.225, scale_height=7200.0)
    assert e.density(0.0) == pytest.approx(1.225)
    assert e.density(7200.0) == pytest.approx(1.225 / np.e)
    v = VacuumAtmosphere()
    assert float(v.density(50e3)) == 0.0
    assert float(v.pressure(50e3)) == 0.0
