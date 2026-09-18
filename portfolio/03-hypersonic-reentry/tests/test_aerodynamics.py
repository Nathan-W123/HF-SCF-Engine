"""Unit tests for the modified-Newtonian aerodynamic model.

Every assertion here is against a value that can be derived in closed form, so
the tests check the surface-integration machinery, not a stored snapshot.
"""

import numpy as np
import pytest

from reentry.aerodynamics import CP_MAX_HYPERSONIC_LIMIT, SphereCone, cp_max
from reentry.constants import GAMMA_AIR


def test_cp_max_hypersonic_limit():
    assert CP_MAX_HYPERSONIC_LIMIT == pytest.approx(1.8393710511, rel=1e-9)
    assert cp_max(1e6) == pytest.approx(CP_MAX_HYPERSONIC_LIMIT, rel=1e-6)


def test_cp_max_continuous_at_mach_one():
    lo = cp_max(1.0 - 1e-9)
    hi = cp_max(1.0 + 1e-9)
    assert lo == pytest.approx(hi, rel=1e-7)
    # Isentropic stagnation Cp at M = 1 for gamma = 1.4.
    g = GAMMA_AIR
    expect = (2.0 / g) * ((1.0 + 0.5 * (g - 1.0)) ** (g / (g - 1.0)) - 1.0)
    assert lo == pytest.approx(expect, rel=1e-9)


def test_cp_max_incompressible_limit_and_monotonicity():
    assert cp_max(1e-4) == pytest.approx(1.0, abs=1e-6)
    m = np.linspace(0.01, 50.0, 2000)
    c = cp_max(m)
    assert np.all(np.diff(c) > -1e-12)


def test_hemisphere_drag_equals_half_cp_max():
    """Newtonian drag of a hemisphere is exactly Cp_max/2 (+ cylinder = 0)."""
    hemi = SphereCone(nose_radius=1.0, base_radius=1.0, half_angle_deg=0.0)
    cd, cl = hemi.coefficients(0.0, 1e6)
    assert cd == pytest.approx(CP_MAX_HYPERSONIC_LIMIT / 2.0, rel=2e-5)
    assert cl == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("theta", [20.0, 45.0, 70.0])
def test_sharp_cone_drag(theta):
    """Newtonian drag of a sharp cone is Cp_max sin^2(theta)."""
    sc = SphereCone(nose_radius=1e-4, base_radius=1.0, half_angle_deg=theta)
    cd, _ = sc.coefficients(0.0, 1e6)
    expect = CP_MAX_HYPERSONIC_LIMIT * np.sin(np.radians(theta)) ** 2
    assert cd == pytest.approx(expect, rel=2e-4)


def test_axisymmetry_of_lift_and_drag():
    sc = SphereCone()
    a = np.radians(np.array([5.0, 12.0, 25.0]))
    cd_p, cl_p = sc.coefficients(a, 25.0)
    cd_m, cl_m = sc.coefficients(-a, 25.0)
    assert np.allclose(cd_p, cd_m, rtol=1e-9)
    assert np.allclose(cl_p, -cl_m, atol=1e-9)


def test_zero_lift_at_zero_alpha_and_drag_peaks_there():
    sc = SphereCone()
    cd0, cl0 = sc.coefficients(0.0, 25.0)
    assert abs(cl0) < 1e-12
    for a in (5.0, 15.0, 30.0):
        cd, _ = sc.coefficients(np.radians(a), 25.0)
        assert cd < cd0


def test_reference_geometry_is_self_consistent():
    sc = SphereCone()
    assert sc.area_ref == pytest.approx(np.pi * 1.30**2)
    assert np.degrees(sc.phi_t) == pytest.approx(20.0)
    d = sc.describe()
    assert 1.5 < d["cd_alpha0_mach25"] < 1.75


def test_cd_rises_through_the_supersonic_range():
    sc = SphereCone()
    cds = [sc.coefficients(0.0, m)[0] for m in (0.3, 1.0, 2.0, 5.0, 25.0)]
    assert all(cds[i] < cds[i + 1] for i in range(len(cds) - 1))


def test_over_blunt_geometry_rejected():
    with pytest.raises(ValueError):
        SphereCone(nose_radius=2.0, base_radius=1.0, half_angle_deg=70.0)
