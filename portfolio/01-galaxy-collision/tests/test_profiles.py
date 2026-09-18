"""The analytic mass models must be internally consistent."""
import numpy as np
import pytest
from scipy.integrate import quad

from galcol.profiles import (ExponentialDisk, GalaxyModel, Hernquist,
                             default_galaxy, nfw_equivalent_hernquist_a)
from galcol.units import G


def test_hernquist_density_integrates_to_enclosed_mass():
    h = Hernquist(mass=3.0, a=1.7)
    for r in (0.2, 1.0, 5.0, 25.0):
        m, _ = quad(lambda x: 4.0 * np.pi * x**2 * h.density(x), 0.0, r,
                    limit=200)
        assert m == pytest.approx(h.mass_enclosed(r), rel=1e-4)


def test_hernquist_total_mass_and_potential():
    h = Hernquist(mass=2.5, a=3.0)
    assert h.mass_enclosed(1e9) == pytest.approx(2.5, rel=1e-6)
    # Phi(r) = -GM/(r+a) must give v_c^2 = G M r/(r+a)^2 by differentiation
    r = 4.0
    dr = 1e-5
    dphi = (h.potential(r + dr) - h.potential(r - dr)) / (2 * dr)
    assert r * dphi == pytest.approx(h.v_circ_sq(r), rel=1e-6)


def test_hernquist_sampling_matches_profile():
    h = Hernquist(mass=1.0, a=2.0, r_trunc=40.0)
    rng = np.random.default_rng(3)
    r = h.sample_radius(rng, 200000)
    assert r.max() <= h.r_trunc * 1.0001
    m_max = h.mass_enclosed(h.r_trunc)
    for rq in (1.0, 2.0, 6.0, 15.0):
        frac = (r < rq).mean()
        assert frac == pytest.approx(h.mass_enclosed(rq) / m_max, abs=0.004)


def test_exponential_disk_enclosed_mass():
    d = ExponentialDisk(mass=4.0, r_scale=3.0)
    for R in (1.0, 3.0, 9.0):
        m, _ = quad(lambda x: 2.0 * np.pi * x * d.sigma(x), 0.0, R, limit=200)
        assert m == pytest.approx(d.mass_enclosed_cyl(R), rel=1e-5)


def test_freeman_disk_rotation_curve_matches_numerical_integration():
    """v_c^2 from the Bessel formula vs direct integration of the thin-disk
    radial force (Binney & Tremaine eq. 2.157)."""
    d = ExponentialDisk(mass=4.0, r_scale=3.0)

    def kernel(k, R):
        from scipy.special import j0, j1
        s0 = d.mass / (2.0 * np.pi * d.r_scale**2)
        # Hankel transform of an exponential disk
        sig_k = s0 * d.r_scale**2 / (1.0 + (k * d.r_scale) ** 2) ** 1.5
        return -2.0 * np.pi * G * sig_k * j1(k * R) * k

    for R in (2.0, 5.0, 9.0):
        val, _ = quad(lambda k: kernel(k, R), 0.0, 60.0 / d.r_scale,
                      limit=400)
        vc2 = -val * R
        assert vc2 == pytest.approx(float(d.v_circ_sq(R)[0]), rel=2e-3)


def test_composite_rotation_curve_is_sane():
    m = default_galaxy()
    R = np.array([2.0, 8.0, 15.0])
    vc = m.v_circ(R)
    assert np.all(vc > 120.0) and np.all(vc < 280.0)
    # Milky-Way-like: peak within 5-12 kpc
    Rg = np.linspace(0.5, 30.0, 400)
    assert 5.0 < Rg[np.argmax(m.v_circ(Rg))] < 12.0


def test_epicyclic_frequency_keplerian_limit():
    """Far outside all the mass, kappa -> Omega (Keplerian)."""
    m = default_galaxy()
    om, ka = m.omega_kappa(np.array([4000.0]))
    assert ka[0] / om[0] == pytest.approx(1.0, abs=0.02)


def test_epicyclic_frequency_flat_rotation_limit():
    """Where v_c is flat, kappa/Omega -> sqrt(2)."""
    m = default_galaxy()
    om, ka = m.omega_kappa(np.array([9.0]))
    assert 1.25 < ka[0] / om[0] < 1.55


def test_nfw_equivalent_hernquist():
    a, mh = nfw_equivalent_hernquist_a(m_200=100.0, c=10.0, r_200=200.0)
    assert 20.0 < a < 60.0
    assert mh > 100.0                 # Hernquist total exceeds M_200
