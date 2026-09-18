"""Heating correlations, radiative-equilibrium wall temperature, Planck colour."""

import numpy as np
import pytest

from reentry.blackbody import blackbody_rgb, cie_xyz_of_temperature, make_blackbody_lut
from reentry.constants import K_SUTTON_GRAVES, SIGMA_SB
from reentry.heating import (
    heat_flux_dkr,
    heat_flux_exponent_sensitivity,
    heat_flux_sutton_graves,
    hot_wall_factor,
    radiative_equilibrium_temperature,
    DKR_REFERENCE_VELOCITY,
)


def test_sutton_graves_matches_its_definition():
    rho, v, rn = 3.0e-4, 6800.0, 0.65
    q = heat_flux_sutton_graves(rho, v, rn)
    assert q == pytest.approx(K_SUTTON_GRAVES * np.sqrt(rho / rn) * v**3, rel=1e-12)


def test_sutton_graves_scaling_laws():
    q0 = heat_flux_sutton_graves(1e-4, 7000.0, 1.0)
    assert heat_flux_sutton_graves(4e-4, 7000.0, 1.0) == pytest.approx(2 * q0, rel=1e-12)
    assert heat_flux_sutton_graves(1e-4, 14000.0, 1.0) == pytest.approx(8 * q0, rel=1e-12)
    assert heat_flux_sutton_graves(1e-4, 7000.0, 4.0) == pytest.approx(q0 / 2, rel=1e-12)


def test_exponent_sensitivity_anchors_to_sutton_graves():
    rho, rn = 2e-4, 0.65
    v = DKR_REFERENCE_VELOCITY
    assert heat_flux_exponent_sensitivity(rho, v, rn) == pytest.approx(
        heat_flux_sutton_graves(rho, v, rn), rel=1e-12
    )


def test_dkr_within_a_factor_of_sutton_graves_over_the_entry_range():
    rho = np.array([1e-5, 1e-4, 5e-4, 2e-3])
    v = np.array([7800.0, 7000.0, 6000.0, 4000.0])
    ratio = heat_flux_dkr(rho, v, 0.65) / heat_flux_sutton_graves(rho, v, 0.65)
    assert np.all(ratio > 0.8) and np.all(ratio < 1.5)


def test_radiative_equilibrium_inverts_stefan_boltzmann():
    t = 2400.0
    eps = 0.85
    q = eps * SIGMA_SB * t**4
    assert radiative_equilibrium_temperature(q, eps) == pytest.approx(t, rel=1e-12)
    assert radiative_equilibrium_temperature(0.0, eps) == 0.0


def test_hot_wall_factor_bounds():
    f = hot_wall_factor(np.array([7800.0, 3000.0, 300.0]), np.array([2400.0] * 3))
    assert np.all(f >= 0.0) and np.all(f <= 1.0)
    assert f[0] > f[1] > f[2]


def test_planck_locus_passes_near_white_at_6500k():
    xyz = cie_xyz_of_temperature(6500.0)[0]
    x, y = xyz[0] / xyz.sum(), xyz[1] / xyz.sum()
    assert abs(x - 0.3127) < 0.02 and abs(y - 0.3290) < 0.02


def test_planck_locus_colour_ordering():
    cool = blackbody_rgb(1500.0)
    warm = blackbody_rgb(3000.0)
    hot = blackbody_rgb(15000.0)
    assert cool[0] > cool[2] and cool[2] < 0.15          # deep red
    assert warm[0] >= warm[1] > warm[2]                   # orange
    assert hot[2] > hot[0]                                # blue-dominant
    assert warm[2] > cool[2]                              # bluer as T rises


def test_planck_chromaticity_is_monotonic_in_temperature():
    t = np.linspace(1000.0, 12000.0, 60)
    xyz = cie_xyz_of_temperature(t)
    x = xyz[:, 0] / xyz.sum(axis=1)
    assert np.all(np.diff(x) < 0)   # locus moves steadily toward blue


def test_blackbody_lut_shape_and_range():
    temps, rgb = make_blackbody_lut(600.0, 3600.0, 128)
    assert temps.shape == (128,) and rgb.shape == (128, 3)
    assert np.all(rgb >= 0.0) and np.all(rgb <= 1.0)
    assert np.allclose(rgb.max(axis=1), 1.0, atol=1e-9)
