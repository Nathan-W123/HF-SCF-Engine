"""Scenario metrics.

Definitions
-----------
min_separation
    ``min over time, over all pairs i != j`` of the 3-D Euclidean distance.
    Computed on the logged samples (20 Hz by default), so it is an upper bound
    on the true continuous-time minimum; the logging rate is fast compared with
    the closing rates involved (worst case ~70 m/s closure => 3.5 m of travel
    between samples), and the residual is reported as ``sampling_gap_m``.
separation_violations
    Number of (sample, pair) events with distance < ``R_min``.
violating_pairs
    Number of distinct unordered pairs that ever violate.
collisions
    Same, with distance < ``R_collision``.
path_efficiency
    ``flown 3-D length / straight-line distance from the start to the goal
    ball``.  Length is accumulated only while the vehicle is en route, i.e.
    until it first comes within ``goal_radius`` of its goal, and the normaliser
    is ``|goal - start| - goal_radius`` so that the ratio is >= 1 for any
    admissible path.  The excess is the cost of routing around no-fly zones plus
    the cost of deconfliction.  The reported mean is over vehicles that reached
    their goal; ``path_efficiency_all_vehicles_mean`` includes the rest.
goal_completion
    Fraction of vehicles whose horizontal distance to their goal fell below the
    guidance goal radius at some point in the run.
time_to_completion
    Time at which the last vehicle reached its goal (nan if not all did).
control_effort
    ``integral |phi| dt`` per vehicle [rad s] over the en-route phase -- a
    bank-activity measure comparable across scenarios.
"""

from __future__ import annotations

import numpy as np


def pairwise_min_distance(states: np.ndarray, active: np.ndarray | None = None):
    """Per-sample minimum pairwise distance and the full distance history.

    Returns ``(dmin_t, dmin_pair_idx)`` where ``dmin_t`` has shape ``(T,)``.
    """
    pos = states[:, :, :3]
    T, N = pos.shape[0], pos.shape[1]
    iu, ju = np.triu_indices(N, k=1)
    d = np.linalg.norm(pos[:, iu, :] - pos[:, ju, :], axis=2)   # (T, P)
    return d, iu, ju


def scenario_metrics(result, R_min=None, R_collision=None) -> dict:
    spec = result.spec
    R_min = spec.avoid.R_min if R_min is None else R_min
    R_collision = spec.avoid.R_collision if R_collision is None else R_collision

    d, iu, ju = pairwise_min_distance(result.states)
    dmin_t = d.min(axis=1)
    k_min = int(np.argmin(dmin_t))
    p_min = int(np.argmin(d[k_min]))

    viol_mask = d < R_min
    coll_mask = d < R_collision
    viol_pairs = np.where(viol_mask.any(axis=0))[0]
    coll_pairs = np.where(coll_mask.any(axis=0))[0]

    # Path length is accumulated until the vehicle enters the goal ball, so the
    # correct normaliser is the straight-line distance to that ball, which by
    # the triangle inequality makes the ratio >= 1 for any admissible path.
    r_goal = spec.guidance.goal_radius
    denom = np.maximum(result.direct_length - r_goal, 1e-9)
    eff = result.flown_length / denom
    reached = ~np.isnan(result.reach_time)
    eff_reached = eff[reached] if reached.any() else eff

    # Sampling-gap bound: worst closing speed x logging interval.
    vg = result.v_ground
    speeds = np.linalg.norm(vg, axis=2)
    dt_log = float(result.t[1] - result.t[0]) if len(result.t) > 1 else 0.0
    gap = float(2.0 * speeds.max() * dt_log)

    out = {
        "scenario": result.name,
        "n_vehicles": int(result.states.shape[1]),
        "duration_s": float(result.t[-1]),
        "R_min_m": float(R_min),
        "R_collision_m": float(R_collision),
        "min_separation_m": float(dmin_t.min()),
        "min_separation_time_s": float(result.t[k_min]),
        "min_separation_pair": [int(iu[p_min]), int(ju[p_min])],
        "separation_margin_m": float(dmin_t.min() - R_min),
        "separation_violation_samples": int(viol_mask.sum()),
        "violating_pairs": int(len(viol_pairs)),
        "collisions": int(coll_mask.sum()),
        "colliding_pairs": int(len(coll_pairs)),
        "sampling_gap_m": gap,
        "path_efficiency_mean": float(eff_reached.mean()),
        "path_efficiency_max": float(eff_reached.max()),
        "path_efficiency_all_vehicles_mean": float(eff.mean()),
        "plan_length_over_direct_mean": float(
            (result.plan_length / np.maximum(result.direct_length, 1e-9)).mean()),
        "goal_completion": float(reached.mean()),
        "n_reached": int(reached.sum()),
        "time_to_completion_s": (float(np.nanmax(result.reach_time))
                                 if reached.all() else None),
        "control_effort_mean_rad_s": float(result.control_effort.mean()),
        "control_effort_max_rad_s": float(result.control_effort.max()),
        "mean_deflection_mps": float(result.deflection.mean()),
        "max_deflection_mps": float(result.deflection.max()),
        "qp_infeasible_events": int(result.n_qp_infeasible),
        "qp_fallback_events": int(result.n_fallback),
        "plan_failures": int(result.plan_failures),
    }
    return out


def envelope_check(result) -> dict:
    """Verify that the flown states never leave the vehicle envelope."""
    vp = result.spec.vehicle
    s = result.states
    V = s[:, :, 3]
    gam = s[:, :, 5]
    phi = s[:, :, 6]
    n_load = 1.0 / np.cos(phi)
    tol = 1e-9
    return {
        "V_min_flown": float(V.min()),
        "V_max_flown": float(V.max()),
        "V_within_limits": bool(V.min() >= vp.V_min - tol and V.max() <= vp.V_max + tol),
        "abs_gamma_max_deg": float(np.rad2deg(np.abs(gam).max())),
        "gamma_within_limits": bool(np.abs(gam).max() <= vp.gamma_max + tol),
        "abs_phi_max_deg": float(np.rad2deg(np.abs(phi).max())),
        "phi_within_limits": bool(np.abs(phi).max() <= vp.phi_limit + tol),
        "load_factor_max": float(n_load.max()),
        "load_factor_within_limits": bool(n_load.max() <= vp.load_factor_max + 1e-6),
    }


def zone_incursions(result) -> dict:
    """Count logged samples inside any no-fly cylinder."""
    zones = result.spec.zones
    if not zones:
        return {"zone_incursion_samples": 0, "min_zone_clearance_m": None}
    pos = result.states[:, :, :3].reshape(-1, 3)
    worst = np.inf
    count = 0
    for z in zones:
        horiz = np.hypot(pos[:, 0] - z.x, pos[:, 1] - z.y)
        inside_band = (pos[:, 2] > z.z_low) & (pos[:, 2] < z.z_high)
        clearance = horiz - z.radius
        worst = min(worst, float(clearance[inside_band].min()) if inside_band.any()
                    else worst)
        count += int(((clearance < 0) & inside_band).sum())
    return {"zone_incursion_samples": count,
            "min_zone_clearance_m": float(worst)}
