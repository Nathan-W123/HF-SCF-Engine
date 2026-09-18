import numpy as np
import pytest

from swarmsim.avoidance import (CollisionAvoidance, HEAD_ON_TOL,
                                orca_halfspace, project_to_kinematic_set,
                                _right_of_way_normal)
from swarmsim.config import AvoidanceParams, VehicleParams
from swarmsim.qp import max_margin_velocity, solve_qp

AP = AvoidanceParams()
VP = VehicleParams()
R_COMB = 2.0 * AP.orca_radius


def _closest_approach(p_rel, v_rel, tau, n=600):
    t = np.linspace(0.0, tau, n)
    return float(np.linalg.norm(p_rel[None, :] - v_rel[None, :] * t[:, None],
                                axis=1).min())


def test_orca_reciprocity():
    """u computed by A must be exactly the negative of u computed by B."""
    rng = np.random.default_rng(0)
    for _ in range(300):
        p = rng.normal(size=3) * 200.0
        if np.linalg.norm(p) < 5.0:
            continue
        vA = rng.normal(size=3) * 15.0
        vB = rng.normal(size=3) * 15.0
        nA, uA = orca_halfspace(p, vA - vB, R_COMB, AP.tau_horizon, AP.tau_escape)
        nB, uB = orca_halfspace(-p, vB - vA, R_COMB, AP.tau_horizon, AP.tau_escape)
        assert np.linalg.norm(uA + uB) < 1e-9
        assert np.linalg.norm(nA + nB) < 1e-9


def test_reciprocal_split_resolves_conflicts():
    """Each side taking half of u must clear the velocity obstacle."""
    rng = np.random.default_rng(1)
    n_conflict = 0
    for _ in range(1500):
        # Non-overlapping pairs only: once two vehicles are already inside the
        # combined radius no velocity can restore separation instantly, and the
        # law switches to the bounded recovery branch instead.
        d = rng.uniform(1.05 * R_COMB, 6.0 * R_COMB)
        u = rng.normal(size=3)
        p = d * u / np.linalg.norm(u)
        vA = rng.normal(size=3) * 18.0
        vB = rng.normal(size=3) * 18.0
        v_rel = vA - vB
        if _closest_approach(p, v_rel, AP.tau_horizon) >= R_COMB:
            continue
        n_conflict += 1
        _, uA = orca_halfspace(p, v_rel, R_COMB, AP.tau_horizon, AP.tau_escape)
        _, uB = orca_halfspace(-p, -v_rel, R_COMB, AP.tau_horizon, AP.tau_escape)
        new_rel = (vA + 0.5 * uA) - (vB + 0.5 * uB)
        assert _closest_approach(p, new_rel, AP.tau_horizon) >= R_COMB - 1e-3
    assert n_conflict > 40


def test_head_on_degeneracy_uses_right_of_way():
    """Exactly head-on: the normal must be a well-defined horizontal turn-right."""
    # Close enough that the pair really is in conflict within the horizon,
    # otherwise the cut-off-cap branch (correctly) returns a permissive plane.
    p = np.array([300.0, 0.0, 0.0])
    v_rel = np.array([50.0, 0.0, 0.0])
    n, u = orca_halfspace(p, v_rel, R_COMB, AP.tau_horizon, AP.tau_escape)
    assert np.isfinite(n).all() and np.isfinite(u).all()
    assert np.linalg.norm(u) > 1.0            # a real, non-degenerate escape
    assert abs(n[2]) < 1e-12                  # horizontal line of sight stays flat
    assert np.allclose(n, _right_of_way_normal(p, R_COMB))
    # the escaped relative velocity lands exactly on the obstacle boundary
    v_new = v_rel + u
    assert (np.linalg.norm(np.cross(p, v_new)) / np.linalg.norm(v_new)
            == pytest.approx(R_COMB, abs=1e-6))
    # and it is antisymmetric, so reciprocity still holds
    n2, u2 = orca_halfspace(-p, -v_rel, R_COMB, AP.tau_horizon, AP.tau_escape)
    assert np.linalg.norm(u + u2) < 1e-12


def test_head_on_escape_clears_the_obstacle():
    p = np.array([300.0, 0.0, 0.0])
    vA = np.array([25.0, 0.0, 0.0])
    vB = np.array([-25.0, 0.0, 0.0])
    _, uA = orca_halfspace(p, vA - vB, R_COMB, AP.tau_horizon, AP.tau_escape)
    _, uB = orca_halfspace(-p, vB - vA, R_COMB, AP.tau_horizon, AP.tau_escape)
    new_rel = (vA + 0.5 * uA) - (vB + 0.5 * uB)
    assert _closest_approach(p, new_rel, AP.tau_horizon) >= R_COMB - 1e-3


def test_no_conflict_leaves_preferred_velocity_untouched():
    ca = CollisionAvoidance(AP, VP)
    pos = np.array([[0.0, 0.0, 800.0], [5000.0, 5000.0, 800.0]])
    vel = np.array([[25.0, 0.0, 0.0], [-25.0, 0.0, 0.0]])
    v_new, diag = ca.filter(pos, vel, vel.copy(), 0.1)
    assert np.allclose(v_new, vel)
    assert diag.deflection.max() < 1e-9


def test_qp_matches_scipy_on_random_feasible_problems():
    scipy_opt = pytest.importorskip("scipy.optimize")
    rng = np.random.default_rng(2)
    n_ok = 0
    for _ in range(80):
        m = int(rng.integers(1, 9))
        A = rng.normal(size=(m, 3))
        A /= np.linalg.norm(A, axis=1, keepdims=True)
        v_pref = rng.normal(size=3) * 10.0
        v_feas = rng.normal(size=3) * 10.0
        b = A @ v_feas - rng.uniform(0.1, 5.0, m)
        v, ok, _ = solve_qp(v_pref, A, b, sweeps=500, tol=1e-13)
        assert ok
        res = scipy_opt.minimize(
            lambda x: 0.5 * np.sum((x - v_pref) ** 2), v_feas,
            jac=lambda x: x - v_pref, method="SLSQP",
            constraints=[{"type": "ineq", "fun": lambda x, A=A, b=b: A @ x - b,
                          "jac": lambda x, A=A: A}],
            options={"maxiter": 500, "ftol": 1e-14})
        if not res.success:
            continue
        n_ok += 1
        assert np.linalg.norm(v - res.x) < 1e-5
    assert n_ok > 25   # SLSQP itself fails on a minority of random problems


def test_qp_returns_preferred_when_unconstrained():
    v = np.array([1.0, 2.0, 3.0])
    out, ok, viol = solve_qp(v, None, None)
    assert ok and np.allclose(out, v) and viol == 0.0


def test_qp_reports_infeasibility():
    # two opposing half-spaces that cannot both hold
    A = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    b = np.array([10.0, 10.0])
    _, ok, viol = solve_qp(np.zeros(3), A, b, sweeps=60)
    assert not ok and viol > 0.0


def test_max_margin_fallback_picks_best_of_candidates():
    A = np.array([[1.0, 0.0, 0.0]])
    b = np.array([5.0])
    cands = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [-10.0, 0.0, 0.0]])
    v, margin = max_margin_velocity(A, b, np.zeros(3), cands)
    assert np.allclose(v, [10.0, 0.0, 0.0])
    assert margin == pytest.approx(5.0)


def test_kinematic_projection_respects_limits():
    rng = np.random.default_rng(4)
    N = 60
    s = np.zeros((N, 7))
    s[:, 3] = VP.V_nom
    s[:, 4] = rng.uniform(-np.pi, np.pi, N)
    wind = np.zeros((N, 3))
    v_des = rng.normal(size=(N, 3)) * 40.0
    v_air = project_to_kinematic_set(s, v_des, wind, VP, horizon=2.0,
                                     return_ground=False)
    sp = np.linalg.norm(v_air, axis=1)
    assert np.all(sp <= VP.V_max + 1e-9) and np.all(sp >= VP.V_min - 1e-9)
    gam = np.arcsin(np.clip(v_air[:, 2] / sp, -1, 1))
    assert np.all(np.abs(gam) <= VP.gamma_max + 1e-9)
    psi_new = np.arctan2(v_air[:, 1], v_air[:, 0])
    dpsi = np.abs((psi_new - s[:, 4] + np.pi) % (2 * np.pi) - np.pi)
    dpsi_max = 9.80665 * np.tan(VP.phi_limit) / VP.V_nom * 2.0
    assert np.all(dpsi <= dpsi_max + 1e-9)


def test_noncooperative_vehicle_is_not_deflected():
    ca = CollisionAvoidance(AP, VP)
    pos = np.array([[0.0, 0.0, 800.0], [300.0, 0.0, 800.0]])
    vel = np.array([[25.0, 0.0, 0.0], [-25.0, 0.0, 0.0]])
    coop = np.array([True, False])
    v_new, diag = ca.filter(pos, vel, vel.copy(), 0.1, cooperative=coop)
    assert np.allclose(v_new[1], vel[1])          # the rogue keeps its command
    assert diag.deflection[0] > 0.5               # the cooperative one gives way


def test_cooperative_vehicle_takes_full_burden_for_rogue():
    """Deflection against a rogue must exceed that against a cooperative peer."""
    ca = CollisionAvoidance(AP, VP)
    pos = np.array([[0.0, 0.0, 800.0], [400.0, 0.0, 800.0]])
    vel = np.array([[25.0, 0.0, 0.0], [-25.0, 0.0, 0.0]])
    _, d_coop = ca.filter(pos, vel, vel.copy(), 0.1,
                          cooperative=np.array([True, True]))
    _, d_rogue = ca.filter(pos, vel, vel.copy(), 0.1,
                           cooperative=np.array([True, False]))
    # The ORCA share doubles (0.5 -> 1.0); the hard barrier row is common to
    # both cases, so the ratio is bounded well below 2.
    assert d_rogue.deflection[0] > d_coop.deflection[0] * 1.25
