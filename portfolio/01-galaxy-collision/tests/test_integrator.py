"""Integrator behaviour: order, symplecticity, rigid-halo option."""
import numpy as np
import pytest

from galcol.integrator import (DirectGravity, TreeGravity, angular_momentum,
                               kinetic_energy, leapfrog_kdk, linear_momentum,
                               potential_energy, timestep_criterion)
from galcol.profiles import Hernquist
from galcol.units import G


def _two_body(ecc=0.4, a=8.0, m1=1.0, m2=1.0):
    mu = G * (m1 + m2)
    period = 2 * np.pi * np.sqrt(a**3 / mu)
    r_p = a * (1 - ecc)
    v_p = np.sqrt(mu * (1 + ecc) / (a * (1 - ecc)))
    f1, f2 = m2 / (m1 + m2), m1 / (m1 + m2)
    pos = np.array([[r_p * f1, 0.0, 0.0], [-r_p * f2, 0.0, 0.0]])
    vel = np.array([[0.0, v_p * f1, 0.0], [0.0, -v_p * f2, 0.0]])
    return pos, vel, np.array([m1, m2]), np.zeros(2), period, a, mu


def test_leapfrog_is_second_order_on_a_kepler_orbit():
    errs = []
    for n in (200, 400, 800):
        pos, vel, mass, eps2, period, a, mu = _two_body()
        r0 = pos[0] - pos[1]
        leapfrog_kdk(pos, vel, mass, eps2, DirectGravity(), period / n, n)
        errs.append(np.linalg.norm((pos[0] - pos[1]) - r0) / a)
    order = np.log2(errs[0] / errs[1])
    order2 = np.log2(errs[1] / errs[2])
    assert order == pytest.approx(2.0, abs=0.25)
    assert order2 == pytest.approx(2.0, abs=0.25)


def test_two_body_returns_to_start_after_one_period():
    pos, vel, mass, eps2, period, a, mu = _two_body(ecc=0.3)
    r0 = (pos[0] - pos[1]).copy()
    leapfrog_kdk(pos, vel, mass, eps2, DirectGravity(), period / 4000, 4000)
    assert np.linalg.norm((pos[0] - pos[1]) - r0) / a < 1e-4


def test_leapfrog_conserves_momentum_and_angular_momentum():
    rng = np.random.default_rng(6)
    n = 150
    pos = np.ascontiguousarray(rng.normal(0, 3, (n, 3)))
    vel = np.ascontiguousarray(rng.normal(0, 40, (n, 3)))
    mass = rng.uniform(0.01, 0.05, n)
    eps2 = np.full(n, 0.1)
    vel -= (mass[:, None] * vel).sum(0) / mass.sum()
    L0 = angular_momentum(mass, pos, vel)
    leapfrog_kdk(pos, vel, mass, eps2, DirectGravity(), 0.002, 200)
    assert np.max(np.abs(linear_momentum(mass, vel))) < 1e-10 * mass.sum() * 40
    L1 = angular_momentum(mass, pos, vel)
    assert np.linalg.norm(L1 - L0) / np.linalg.norm(L0) < 1e-9


def test_energy_error_is_bounded_not_growing():
    """Over many orbits the leapfrog energy error must oscillate, so the
    error late in the run is no worse than early in the run."""
    pos, vel, mass, eps2, period, a, mu = _two_body(ecc=0.5)
    g = DirectGravity()
    dt = period / 600          # 10 full orbits in 6000 steps
    rel = []

    class CB:
        wants_potential = True

        def __call__(self, step, t, p, v, acc, pot):
            if pot is not None:
                rel.append(kinetic_energy(mass, v) + potential_energy(mass, pot))

    leapfrog_kdk(pos, vel, mass, eps2, g, dt, 6000, callback=CB())
    e = np.asarray(rel)
    r = np.abs((e - e[0]) / abs(e[0]))
    assert r.max() < 1e-3
    early = r[: len(r) // 4].max()
    late = r[-len(r) // 4:].max()
    assert late < 3.0 * early + 1e-9         # no secular growth


def test_rigid_halo_matches_analytic_acceleration():
    halo = Hernquist(mass=40.0, a=15.0)
    g = DirectGravity(rigid_halo=halo)
    pos = np.array([[10.0, 0.0, 0.0], [0.0, -25.0, 0.0]])
    mass = np.zeros(2)                      # isolate the rigid term
    eps2 = np.full(2, 0.1)
    acc, pot = g(pos, mass, eps2, want_potential=True)
    assert acc[0, 0] == pytest.approx(-halo.accel_mag(10.0), rel=1e-10)
    assert acc[1, 1] == pytest.approx(halo.accel_mag(25.0), rel=1e-10)
    assert pot[0] == pytest.approx(halo.potential(10.0), rel=1e-10)


def test_tree_and_direct_gravity_give_the_same_orbit():
    rng = np.random.default_rng(8)
    n = 800
    pos = np.ascontiguousarray(rng.normal(0, 5, (n, 3)))
    vel = np.ascontiguousarray(rng.normal(0, 30, (n, 3)))
    mass = np.full(n, 0.01)
    eps2 = np.full(n, 0.25)
    p1, v1 = pos.copy(), vel.copy()
    p2, v2 = pos.copy(), vel.copy()
    leapfrog_kdk(p1, v1, mass, eps2, DirectGravity(), 0.002, 60)
    leapfrog_kdk(p2, v2, mass, eps2, TreeGravity(theta=0.3), 0.002, 60)
    scale = np.linalg.norm(p1 - pos, axis=1).mean()
    assert np.median(np.linalg.norm(p1 - p2, axis=1)) < 0.05 * scale


def test_timestep_criterion_shrinks_with_acceleration():
    acc = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 100.0]])
    eps = np.array([0.25, 0.25])
    dt = timestep_criterion(acc, eps)
    assert dt[0] > dt[1]
    assert dt[1] == pytest.approx(np.sqrt(2 * 0.025 * 0.25 / 100.0), rel=1e-12)
