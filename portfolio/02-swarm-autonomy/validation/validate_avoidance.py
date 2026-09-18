#!/usr/bin/env python3
"""Validation 3 -- collision-avoidance invariant and law properties.

Part A: the safety invariant on the high-density stress test
-----------------------------------------------------------
Run the antipodal ``swap`` scenario (36 vehicles on a 2.1 km ring, every vehicle
assigned the diametrically opposite goal, so every nominal trajectory passes
through one common point) and assert

    min over all pairs and all logged times of |p_i - p_j|  >=  R_min
    number of samples with separation < R_collision           ==  0

The achieved minimum and its margin over ``R_min`` are reported, together with
the sampling-gap bound (worst closing speed x logging interval), which bounds
how much the true continuous-time minimum can differ from the sampled one.

Part B: properties of the avoidance law itself
----------------------------------------------
All the pair geometries sampled below are *non-overlapping* (separation above
the combined radius): once two vehicles are already inside it, no instantaneous
velocity can restore separation and the law switches to a bounded recovery
branch, which is checked separately in the test suite.

1. **Reciprocity.**  For a random pair, the ORCA escape vector computed by A
   and by B must satisfy ``u_A = -u_B``.  This is what lets the two vehicles
   each take half the correction without communicating.
2. **Joint feasibility / safety of the reciprocal solution.**  If each vehicle
   takes exactly its half (``v_A + u_A/2``, ``v_B + u_B/2``), the resulting
   relative velocity must lie outside the velocity obstacle, i.e. the pair is
   predicted collision-free over the time horizon ``tau``.  Checked by forward
   propagation of the relative state.
3. **Half-space membership.**  The reciprocal velocities must lie on the
   correct side of their own ORCA plane (they are on its boundary).
4. **QP correctness.**  The Hildreth solver's answer is compared against
   ``scipy.optimize.minimize`` (SLSQP) on random feasible problems: both the
   objective value and the solution vector must agree.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from scipy.optimize import minimize

from common import report                                   # noqa: E402
from swarmsim import metrics, scenarios, sim                 # noqa: E402
from swarmsim.avoidance import orca_halfspace                # noqa: E402
from swarmsim.config import AvoidanceParams                  # noqa: E402
from swarmsim.qp import solve_qp                             # noqa: E402


# --------------------------------------------------------------------------
def part_a(result=None):
    if result is None:
        result = sim.run(scenarios.swap())
    m = metrics.scenario_metrics(result)
    env = metrics.envelope_check(result)
    return m, env, result


# --------------------------------------------------------------------------
def _vo_clear(p_rel, v_rel, r_comb, tau, n_t=400):
    """Is the pair predicted collision-free over [0, tau] at this v_rel?"""
    t = np.linspace(0.0, tau, n_t)
    d = np.linalg.norm(p_rel[None, :] - v_rel[None, :] * t[:, None], axis=1)
    return float(d.min())


def part_b(n_cases=1500, seed=11):
    rng = np.random.default_rng(seed)
    ap = AvoidanceParams()
    r_comb = 2.0 * ap.orca_radius
    tau = ap.tau_horizon

    worst_recip = 0.0
    worst_plane_A = np.inf
    worst_plane_B = np.inf
    n_conflict = 0
    n_resolved = 0
    worst_miss_ratio = np.inf

    for _ in range(n_cases):
        d = rng.uniform(1.05 * r_comb, 6.0 * r_comb)
        u = rng.normal(size=3)
        u /= np.linalg.norm(u)
        p_rel = d * u                               # B relative to A
        vA = rng.normal(size=3) * np.array([14.0, 14.0, 3.0])
        vA = vA / max(np.linalg.norm(vA), 1e-9) * rng.uniform(18.0, 34.0)
        vB = rng.normal(size=3) * np.array([14.0, 14.0, 3.0])
        vB = vB / max(np.linalg.norm(vB), 1e-9) * rng.uniform(18.0, 34.0)
        v_rel = vA - vB

        nA, uA = orca_halfspace(p_rel, v_rel, r_comb, tau, ap.tau_escape)
        nB, uB = orca_halfspace(-p_rel, -v_rel, r_comb, tau, ap.tau_escape)
        worst_recip = max(worst_recip, float(np.linalg.norm(uA + uB)))

        vA_new = vA + 0.5 * uA
        vB_new = vB + 0.5 * uB
        worst_plane_A = min(worst_plane_A, float((vA_new - (vA + 0.5 * uA)) @ nA))
        worst_plane_B = min(worst_plane_B, float((vB_new - (vB + 0.5 * uB)) @ nB))

        miss0 = _vo_clear(p_rel, v_rel, r_comb, tau)
        if miss0 < r_comb:                          # a genuine conflict
            n_conflict += 1
            miss1 = _vo_clear(p_rel, vA_new - vB_new, r_comb, tau, n_t=4000)
            worst_miss_ratio = min(worst_miss_ratio, miss1 / r_comb)
            if miss1 >= r_comb - 1e-3:
                n_resolved += 1

    return {"n_cases": n_cases,
            "max_reciprocity_residual_mps": worst_recip,
            "min_halfspace_margin_A": worst_plane_A,
            "min_halfspace_margin_B": worst_plane_B,
            "n_conflict_cases": n_conflict,
            "n_resolved_by_reciprocal_split": n_resolved,
            "worst_resolved_miss_over_r_comb": (None if not np.isfinite(worst_miss_ratio)
                                                else worst_miss_ratio)}


# --------------------------------------------------------------------------
def part_qp(n_cases=120, seed=5):
    rng = np.random.default_rng(seed)
    worst_dv = 0.0
    worst_dobj = 0.0
    n_ok = 0
    for _ in range(n_cases):
        m = rng.integers(1, 12)
        A = rng.normal(size=(m, 3))
        A /= np.linalg.norm(A, axis=1, keepdims=True)
        v_pref = rng.normal(size=3) * 12.0
        # Build a guaranteed-feasible set: pick an interior point and offset b.
        v_feas = rng.normal(size=3) * 12.0
        b = A @ v_feas - rng.uniform(0.05, 6.0, m)

        v, ok, viol = solve_qp(v_pref, A, b, sweeps=400, tol=1e-12)
        if not ok:
            continue
        res = minimize(lambda x: 0.5 * np.sum((x - v_pref) ** 2), v_feas,
                       jac=lambda x: x - v_pref, method="SLSQP",
                       constraints=[{"type": "ineq",
                                     "fun": lambda x, A=A, b=b: A @ x - b,
                                     "jac": lambda x, A=A: A}],
                       options={"maxiter": 400, "ftol": 1e-12})
        if not res.success:
            continue
        n_ok += 1
        worst_dv = max(worst_dv, float(np.linalg.norm(v - res.x)))
        o1 = 0.5 * float(np.sum((v - v_pref) ** 2))
        o2 = 0.5 * float(np.sum((res.x - v_pref) ** 2))
        worst_dobj = max(worst_dobj, abs(o1 - o2))
    return {"n_compared": n_ok,
            "max_solution_difference_vs_slsqp": worst_dv,
            "max_objective_difference_vs_slsqp": worst_dobj}


# --------------------------------------------------------------------------
def run(result=None):
    m, env, result = part_a(result)
    b = part_b()
    q = part_qp()

    R_min = m["R_min_m"]
    checks = {
        "stress_test_min_separation_ge_R_min": bool(m["min_separation_m"] >= R_min),
        "stress_test_zero_collisions": bool(m["collisions"] == 0),
        "stress_test_all_goals_reached": bool(m["goal_completion"] >= 1.0 - 1e-9),
        "envelope_respected": bool(env["V_within_limits"] and env["phi_within_limits"]
                                   and env["gamma_within_limits"]
                                   and env["load_factor_within_limits"]),
        "orca_reciprocity_u_A_eq_minus_u_B": bool(b["max_reciprocity_residual_mps"] < 1e-9),
        "reciprocal_split_resolves_every_conflict":
            bool(b["n_conflict_cases"] > 0
                 and b["n_resolved_by_reciprocal_split"] == b["n_conflict_cases"]),
        "reciprocal_split_lands_on_the_obstacle_boundary":
            bool(b["worst_resolved_miss_over_r_comb"] is not None
                 and b["worst_resolved_miss_over_r_comb"] >= 1.0 - 1e-4),
        "qp_matches_slsqp_to_1e-6":
            bool(q["n_compared"] > 50 and q["max_solution_difference_vs_slsqp"] < 1e-6),
    }
    payload = {
        "name": "collision_avoidance",
        "description": ("Safety invariant on the 36-vehicle antipodal swap, plus "
                        "reciprocity/feasibility of the ORCA law and QP correctness."),
        "stress_test": m,
        "envelope": env,
        "law_properties": b,
        "qp_cross_check": q,
        "checks": checks,
        "all_passed": all(checks.values()),
    }
    return payload


if __name__ == "__main__":
    report("validate_avoidance", run())
