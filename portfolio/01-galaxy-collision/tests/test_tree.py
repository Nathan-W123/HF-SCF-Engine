"""Barnes-Hut correctness: it must agree with the exact sum."""
import numpy as np
import pytest

from galcol.tree import (accel_direct, accel_direct_subset, accel_tree,
                         build_tree)
from galcol.units import G


def _cluster(n, seed=0, sep=30.0):
    rng = np.random.default_rng(seed)
    a = rng.normal(0.0, 4.0, (n // 2, 3))
    b = rng.normal(0.0, 6.0, (n - n // 2, 3)) + np.array([sep, 3.0, -2.0])
    pos = np.ascontiguousarray(np.vstack([a, b]))
    mass = rng.uniform(0.5, 1.5, n) / n
    eps2 = np.full(n, 0.09)
    return pos, mass, eps2


def test_tree_matches_direct_and_improves_with_theta():
    pos, mass, eps2 = _cluster(3000, seed=1)
    exact, pot_e = accel_direct(pos, mass, eps2, want_potential=True)
    norm = np.linalg.norm(exact, axis=1)
    prev = None
    for theta in (1.0, 0.7, 0.5, 0.3):
        acc, pot = accel_tree(pos, mass, eps2, theta=theta,
                              want_potential=True)
        err = np.median(np.linalg.norm(acc - exact, axis=1) / norm)
        assert err < 2e-2
        if prev is not None:
            assert err < prev            # smaller theta -> smaller error
        prev = err
    assert np.allclose(pot, pot_e, rtol=5e-3)


def test_group_and_per_particle_walks_agree():
    pos, mass, eps2 = _cluster(2000, seed=5)
    a_grp, _ = accel_tree(pos, mass, eps2, theta=0.5, groups=True)
    a_pp, _ = accel_tree(pos, mass, eps2, theta=0.5, groups=False)
    exact, _ = accel_direct(pos, mass, eps2)
    n = np.linalg.norm(exact, axis=1)
    assert np.median(np.linalg.norm(a_grp - exact, axis=1) / n) < 1e-2
    assert np.median(np.linalg.norm(a_pp - exact, axis=1) / n) < 1e-2


def test_direct_sum_conserves_momentum_exactly():
    """Softening is symmetric (eps_i^2 + eps_j^2), so sum(m a) must vanish."""
    rng = np.random.default_rng(2)
    n = 400
    pos = np.ascontiguousarray(rng.normal(0, 5, (n, 3)))
    mass = rng.uniform(0.2, 2.0, n)
    eps2 = rng.uniform(0.01, 0.5, n)      # deliberately unequal softenings
    acc, _ = accel_direct(pos, mass, eps2)
    net = (mass[:, None] * acc).sum(axis=0)
    scale = np.abs(mass[:, None] * acc).sum()
    assert np.max(np.abs(net)) / scale < 1e-12


def test_two_body_force_is_exact_newton():
    pos = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    mass = np.array([2.0, 3.0])
    eps2 = np.zeros(2)
    acc, pot = accel_direct(pos, mass, eps2, want_potential=True)
    assert acc[0, 0] == pytest.approx(G * 3.0 / 100.0, rel=1e-12)
    assert acc[1, 0] == pytest.approx(-G * 2.0 / 100.0, rel=1e-12)
    assert pot[0] == pytest.approx(-G * 3.0 / 10.0, rel=1e-12)


def test_plummer_softening_removes_the_singularity():
    pos = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    mass = np.array([1.0, 1.0])
    eps2 = np.array([0.25, 0.25])
    acc, pot = accel_direct(pos, mass, eps2, want_potential=True)
    assert np.all(np.isfinite(acc)) and np.all(np.isfinite(pot))
    assert np.allclose(acc, 0.0)
    assert pot[0] == pytest.approx(-G / np.sqrt(0.5), rel=1e-12)


def test_tree_handles_degenerate_configurations():
    # all particles identical, then all collinear: the build must terminate
    for pos in (np.zeros((200, 3)),
                np.ascontiguousarray(np.stack(
                    [np.linspace(0, 1, 200), np.zeros(200), np.zeros(200)], 1))):
        pos = np.ascontiguousarray(pos, dtype=float)
        mass = np.full(200, 0.01)
        eps2 = np.full(200, 0.04)
        acc, _ = accel_tree(pos, mass, eps2, theta=0.6)
        assert np.all(np.isfinite(acc))


def test_tree_node_masses_sum_to_total():
    pos, mass, eps2 = _cluster(1500, seed=9)
    t = build_tree(pos, mass, eps2, leaf_size=12)
    assert t.mass[0] == pytest.approx(mass.sum(), rel=1e-12)
    com = np.array([t.mx[0], t.my[0], t.mz[0]])
    assert np.allclose(com, (mass[:, None] * pos).sum(0) / mass.sum(), atol=1e-9)


def test_direct_subset_matches_full_direct():
    pos, mass, eps2 = _cluster(600, seed=4)
    full, _ = accel_direct(pos, mass, eps2)
    idx = np.array([0, 7, 123, 599], dtype=np.int64)
    sub = accel_direct_subset(pos, mass, eps2, idx)
    assert np.allclose(sub, full[idx], rtol=1e-12, atol=1e-12)
