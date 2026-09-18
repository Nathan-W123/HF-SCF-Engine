#!/usr/bin/env python3
"""Validation 2 -- L1 guidance tracking convergence.

A single vehicle, avoidance disabled, no wind, released from a range of initial
lateral offsets and asked to capture

  * a straight reference path, and
  * a circular reference path of radius 600 m (~5.4 x the minimum turn radius).

For each release the script records the full cross-track-error history against
distance flown and extracts

  settling distance  first along-path distance after which |xte| stays below
                     5 % of the initial offset (and below 5 m) for good;
  steady-state error mean |xte| over the last 600 m of the run;
  overshoot          max |xte| on the far side of the path after first capture.

Thresholds: every release must settle, with steady-state ``|xte|`` below 0.01 m
on the straight path and 0.05 m on the circle, and no case may diverge.  The
circular threshold can be that tight because the L1 reference point is computed
as the *exact* intersection of the L1 circle with the reference circle, which
makes the on-path tangential state an exact equilibrium of the law rather than
leaving the standing curvature offset a carrot-chasing implementation would.
"""

from __future__ import annotations

import numpy as np

from common import report                                    # noqa: E402
from swarmsim.config import GuidanceParams, VehicleParams     # noqa: E402
from swarmsim import dynamics as dy                           # noqa: E402
from swarmsim.guidance import (CirclePath, PolylinePath,      # noqa: E402
                               l1_distance, preferred_velocity)


def fly(path, offset, vp, gp, T=260.0, dt=0.02, h_ref=800.0, start_psi=None):
    """Release one vehicle at a lateral offset and track the path."""
    if isinstance(path, CirclePath):
        p0 = path.c + np.array([path.R + offset, 0.0])
        psi0 = np.pi / 2 * path.dir if start_psi is None else start_psi
    else:
        p0 = np.array([0.0, offset])
        psi0 = 0.0 if start_psi is None else start_psi

    s = np.array([[p0[0], p0[1], h_ref, vp.V_nom, psi0, 0.0, 0.0]])
    wind = np.zeros((1, 3))
    xte_hist, dist_hist = [], []
    dist = 0.0
    prev = s[0, :2].copy()
    n = int(T / dt)
    ctrl_every = 5
    cmd = np.array([[vp.V_nom, 0.0, 0.0]])
    for k in range(n):
        if k % ctrl_every == 0:
            vg = dy.ground_velocity(s, wind)[0]
            v_pref, L1 = preferred_velocity(s[0, :3], vg, path, h_ref, vp.V_nom, gp)
            cmd = dy.commands_from_velocity(s, v_pref[None, :] - wind, vp,
                                            np.array([L1]), dt * ctrl_every)
        s = dy.rk4_step(s, cmd, dt, vp, wind=wind)
        dist += float(np.linalg.norm(s[0, :2] - prev))
        prev = s[0, :2].copy()
        if k % 5 == 0:
            _, _, xte = path.project(s[0, :2])
            xte_hist.append(xte)
            dist_hist.append(dist)
    return np.array(dist_hist), np.array(xte_hist)


def analyse(dist, xte, offset, band_frac=0.05, band_floor=5.0):
    band = max(band_frac * abs(offset), band_floor)
    inside = np.abs(xte) <= band
    settle = np.nan
    if inside.any():
        # last index where it is outside the band; settling is just after that
        outside = np.where(~inside)[0]
        j = outside[-1] + 1 if len(outside) else 0
        if j < len(dist):
            settle = float(dist[j])
    tail = dist >= dist[-1] - 600.0
    ss = float(np.mean(np.abs(xte[tail])))
    # overshoot: extreme excursion of opposite sign after the first crossing
    sgn = np.sign(offset) if offset != 0 else 1.0
    after = xte * sgn
    cross = np.where(after <= 0)[0]
    over = float(np.max(-after[cross[0]:])) if len(cross) else 0.0
    return {"settling_distance_m": settle,
            "settling_band_m": band,
            "steady_state_xte_m": ss,
            "overshoot_m": over,
            "max_abs_xte_m": float(np.abs(xte).max())}


def run():
    vp = VehicleParams()
    gp = GuidanceParams()
    offsets = [-400.0, -200.0, -60.0, 60.0, 200.0, 400.0]

    straight = PolylinePath(np.array([[0.0, 0.0], [14000.0, 0.0]]))
    circle = CirclePath((0.0, 0.0), 600.0, direction=+1)

    out = {"straight": [], "circle": []}
    curves = {"straight": [], "circle": []}
    for off in offsets:
        d, x = fly(straight, off, vp, gp, T=300.0)
        a = analyse(d, x, off)
        a["offset_m"] = off
        out["straight"].append(a)
        curves["straight"].append((off, d[::6].tolist(), x[::6].tolist()))

        d, x = fly(circle, off, vp, gp, T=300.0)
        a = analyse(d, x, off)
        a["offset_m"] = off
        out["circle"].append(a)
        curves["circle"].append((off, d[::6].tolist(), x[::6].tolist()))

    st_ss = max(r["steady_state_xte_m"] for r in out["straight"])
    ci_ss = max(r["steady_state_xte_m"] for r in out["circle"])
    st_settled = all(np.isfinite(r["settling_distance_m"]) for r in out["straight"])
    ci_settled = all(np.isfinite(r["settling_distance_m"]) for r in out["circle"])
    st_set = max(r["settling_distance_m"] for r in out["straight"]) if st_settled else np.inf
    ci_set = max(r["settling_distance_m"] for r in out["circle"]) if ci_settled else np.inf

    checks = {
        "straight_all_releases_settle": bool(st_settled),
        "circle_all_releases_settle": bool(ci_settled),
        "straight_steady_state_xte_lt_0.01m": bool(st_ss < 0.01),
        "circle_steady_state_xte_lt_0.05m": bool(ci_ss < 0.05),
        "no_divergence": bool(all(r["max_abs_xte_m"] < 1.2 * max(abs(o) for o in offsets)
                                  + 60.0 for r in out["straight"] + out["circle"])),
    }
    payload = {
        "name": "tracking_convergence",
        "description": "L1 guidance capture of straight and circular paths.",
        "L1_distance_at_cruise_m": l1_distance(vp.V_nom, gp),
        "circle_radius_m": circle.R,
        "min_turn_radius_m": vp.turn_radius_min,
        "straight": out["straight"],
        "circle": out["circle"],
        "worst_straight_settling_distance_m": float(st_set),
        "worst_circle_settling_distance_m": float(ci_set),
        "worst_straight_steady_state_xte_m": float(st_ss),
        "worst_circle_steady_state_xte_m": float(ci_ss),
        "checks": checks,
        "all_passed": all(checks.values()),
    }
    np.savez_compressed(
        __file__.replace("validate_tracking.py", "tracking_curves.npz"),
        **{f"{kind}_{i}": np.array([c[1], c[2]])
           for kind, lst in curves.items() for i, c in enumerate(lst)},
        offsets=np.array(offsets))
    return payload


if __name__ == "__main__":
    report("validate_tracking", run())
