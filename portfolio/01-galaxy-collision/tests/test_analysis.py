"""Structural diagnostics must recover known inputs."""
import numpy as np
import pytest

from galcol.analysis import (cyl_mass_radius, fit_scale_length, mass_radius,
                             robust_centre, rotation_curve,
                             surface_density_profile, vertical_thickness)


def _exp_disk(n=200000, rd=3.0, zd=0.3, seed=0, centre=(0.0, 0.0, 0.0)):
    rng = np.random.default_rng(seed)
    R = -rd * (np.log(rng.random(n)) + np.log(rng.random(n)))
    phi = rng.uniform(0, 2 * np.pi, n)
    z = zd * np.arctanh(rng.uniform(-1 + 1e-9, 1 - 1e-9, n))
    pos = np.stack([R * np.cos(phi), R * np.sin(phi), z], 1) + np.asarray(centre)
    return pos, np.full(n, 1.0 / n)


def test_fit_scale_length_recovers_the_input():
    pos, m = _exp_disk(rd=3.0)
    assert fit_scale_length(pos, m) == pytest.approx(3.0, rel=0.03)
    pos, m = _exp_disk(rd=4.5, seed=1)
    assert fit_scale_length(pos, m, r_fit=(4.0, 14.0)) == pytest.approx(4.5, rel=0.04)


def test_vertical_thickness_recovers_the_scale_height():
    pos, m = _exp_disk(zd=0.3)
    med, rms = vertical_thickness(pos, m)
    assert med == pytest.approx(0.3 * np.arctanh(0.5), rel=0.03)
    assert rms > med


def test_cyl_mass_radius_matches_the_analytic_half_mass_radius():
    pos, m = _exp_disk(rd=3.0)
    # exponential disk: M(<R)/M = 1-(1+x)e^-x = 0.5 at x = 1.678
    assert cyl_mass_radius(pos, m) == pytest.approx(1.678 * 3.0, rel=0.02)


def test_mass_radius_on_a_uniform_sphere():
    rng = np.random.default_rng(2)
    n = 200000
    r = 5.0 * rng.random(n) ** (1 / 3)
    ct = rng.uniform(-1, 1, n)
    st = np.sqrt(1 - ct**2)
    ph = rng.uniform(0, 2 * np.pi, n)
    pos = np.stack([r * st * np.cos(ph), r * st * np.sin(ph), r * ct], 1)
    m = np.full(n, 1.0 / n)
    assert mass_radius(pos, m) == pytest.approx(5.0 * 0.5 ** (1 / 3), rel=0.01)


def test_robust_centre_finds_an_offset_core_despite_debris():
    rng = np.random.default_rng(3)
    core = rng.normal(0, 1.0, (8000, 3)) + np.array([7.0, -3.0, 1.0])
    debris = rng.uniform(-200, 200, (2000, 3))
    c = robust_centre(np.vstack([core, debris]))
    assert np.allclose(c, [7.0, -3.0, 1.0], atol=0.4)


def test_surface_density_profile_is_exponential():
    pos, m = _exp_disk(rd=3.0)
    mid, sig = surface_density_profile(pos, m, r_max=12.0, n_bins=20)
    sel = (mid > 2) & (mid < 10)
    slope = np.polyfit(mid[sel], np.log(sig[sel]), 1)[0]
    assert -1.0 / slope == pytest.approx(3.0, rel=0.05)


def test_rotation_curve_recovers_a_flat_input():
    rng = np.random.default_rng(4)
    n = 60000
    R = rng.uniform(1.0, 15.0, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    pos = np.stack([R * np.cos(phi), R * np.sin(phi),
                    rng.normal(0, 0.2, n)], 1)
    v = 210.0
    vel = np.stack([-v * np.sin(phi), v * np.cos(phi), np.zeros(n)], 1)
    mid, vc = rotation_curve(pos, vel, np.full(n, 1.0), r_max=15.0, n_bins=10)
    ok = np.isfinite(vc) & (mid > 2)
    assert np.allclose(vc[ok], v, rtol=0.02)
