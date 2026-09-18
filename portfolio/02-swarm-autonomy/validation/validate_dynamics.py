#!/usr/bin/env python3
"""Validation 1 -- vehicle dynamics.

Three independent checks:

A. **Steady coordinated-turn radius.**  Hold ``phi_cmd = phi``, ``gamma = 0``,
   ``V_cmd = V`` and fly for several turn periods with no wind.  Fit a circle to
   the flown ground track and compare with the analytic ``R = V^2/(g tan phi)``.
   Threshold: relative error < 1e-6 for every (V, phi) in the grid.

B. **Envelope limits.**  Drive the model with commands well outside the
   envelope (bank, airspeed, flight-path angle) and confirm that the flown
   states stay inside the limits, and that the commanded *rates* never exceed
   the roll-rate / flight-path-rate / acceleration limits.

C. **Integrator convergence order.**  Integrate a smooth, non-saturating
   manoeuvre (so the right-hand side is C-infinity and the classical order
   applies) with a sequence of halved timesteps against a reference computed at
   ``dt/64``, and fit the slope of ``log(error)`` vs ``log(dt)``.  RK4 should
   give 4.  Threshold: measured slope in [3.7, 4.3].
"""

from __future__ import annotations

import numpy as np

from common import fit_circle, report            # noqa: E402
from swarmsim.config import G0, VehicleParams     # noqa: E402
from swarmsim import dynamics as dy               # noqa: E402


def check_turn_radius():
    vp = VehicleParams()
    rows = []
    worst = 0.0
    for V in (18.0, 22.0, 25.0, 30.0, 34.0):
        for phi_deg in (10.0, 20.0, 30.0, 40.0, 45.0):
            phi = np.deg2rad(phi_deg)
            if phi > vp.phi_limit + 1e-12:
                continue
            R_an = dy.turn_radius(V, phi)
            period = 2 * np.pi * R_an / V
            dt = 0.01
            n = int(round(3.0 * period / dt))
            s = np.array([[0.0, 0.0, 800.0, V, 0.0, 0.0, phi]])
            cmd = np.array([[V, phi, 0.0]])
            pts = np.empty((n, 2))
            for k in range(n):
                s = dy.rk4_step(s, cmd, dt, vp, wind=np.zeros((1, 3)))
                pts[k] = s[0, :2]
            _, _, R_fit, rms = fit_circle(pts)
            rel = abs(R_fit - R_an) / R_an
            worst = max(worst, rel)
            rows.append({"V_mps": V, "phi_deg": phi_deg,
                         "R_analytic_m": R_an, "R_measured_m": R_fit,
                         "rel_error": rel, "circle_fit_rms_m": rms})
    return rows, worst


def check_limits():
    vp = VehicleParams()
    dt = 0.02
    n = int(60.0 / dt)
    rng = np.random.default_rng(0)
    N = 8
    s = np.zeros((N, 7))
    s[:, 2] = 900.0
    s[:, 3] = vp.V_nom
    s[:, 4] = rng.uniform(-np.pi, np.pi, N)
    prev = s.copy()
    max_rate = np.zeros(3)      # |dV/dt|, |dgamma/dt|, |dphi/dt|
    for k in range(n):
        # deliberately illegal commands, switching sign every 4 s
        sgn = 1.0 if (k * dt) % 8.0 < 4.0 else -1.0
        cmd = np.tile(np.array([[60.0 * sgn if sgn > 0 else 2.0,
                                 sgn * np.deg2rad(85.0),
                                 sgn * np.deg2rad(40.0)]]), (N, 1))
        s = dy.rk4_step(s, cmd, dt, vp, wind=np.zeros((N, 3)))
        rate = np.abs(s[:, [3, 5, 6]] - prev[:, [3, 5, 6]]) / dt
        max_rate = np.maximum(max_rate, rate.max(axis=0))
        prev = s.copy()
    return {
        "V_min_flown": float(s[:, 3].min()),
        "abs_phi_max_deg": float(np.rad2deg(np.abs(s[:, 6]).max())),
        "max_dV_dt": float(max_rate[0]),
        "max_dgamma_dt_deg_s": float(np.rad2deg(max_rate[1])),
        "max_dphi_dt_deg_s": float(np.rad2deg(max_rate[2])),
        "limit_accel_max": vp.accel_max,
        "limit_gamma_dot_deg_s": float(np.rad2deg(vp.gamma_dot_max)),
        "limit_p_max_deg_s": float(np.rad2deg(vp.p_max)),
        "limit_phi_deg": float(np.rad2deg(vp.phi_limit)),
    }


def _integrate(dt, T, vp, s0, cmd):
    s = s0.copy()
    n = int(round(T / dt))
    for _ in range(n):
        s = dy.rk4_step(s, cmd, dt, vp, wind=np.array([[3.0, -2.0, 0.5]]),
                        enforce_limits=False)
    return s


def check_convergence():
    """Order test on a smooth trajectory with no active saturation."""
    vp = VehicleParams()
    # Start banked and climbing, command a nearby attitude so the first-order
    # lags never hit their rate limits and no state limit is reached.
    s0 = np.array([[0.0, 0.0, 800.0, 25.0, 0.4, np.deg2rad(2.0),
                    np.deg2rad(12.0)]])
    cmd = np.array([[26.5, np.deg2rad(22.0), np.deg2rad(5.0)]])
    T = 12.0
    dt_ref = T / 24576.0
    ref = _integrate(dt_ref, T, vp, s0, cmd)[0]

    dts, errs = [], []
    for m in (384, 768, 1536, 3072):
        dt = T / m
        s = _integrate(dt, T, vp, s0, cmd)[0]
        e = float(np.linalg.norm(s[:3] - ref[:3]))
        dts.append(dt)
        errs.append(e)
    dts = np.array(dts)
    errs = np.array(errs)
    slope = float(np.polyfit(np.log(dts), np.log(errs), 1)[0])
    ratios = (errs[:-1] / errs[1:]).tolist()
    return {"dt_s": dts.tolist(), "position_error_m": errs.tolist(),
            "error_ratio_per_halving": ratios,
            "measured_order": slope,
            "reference_dt_s": dt_ref}


def run():
    rows, worst = check_turn_radius()
    lim = check_limits()
    conv = check_convergence()
    vp = VehicleParams()

    checks = {
        "turn_radius_matches_analytic_rel_err_lt_1e-6": worst < 1e-6,
        "bank_within_limit": lim["abs_phi_max_deg"] <= np.rad2deg(vp.phi_limit) + 1e-6,
        "airspeed_within_limit": lim["V_min_flown"] >= vp.V_min - 1e-6,
        "roll_rate_within_limit":
            lim["max_dphi_dt_deg_s"] <= np.rad2deg(vp.p_max) * 1.02,
        "gamma_rate_within_limit":
            lim["max_dgamma_dt_deg_s"] <= np.rad2deg(vp.gamma_dot_max) * 1.02,
        "accel_within_limit": lim["max_dV_dt"] <= vp.accel_max * 1.02,
        "rk4_order_between_3.7_and_4.3":
            3.7 <= conv["measured_order"] <= 4.3,
    }
    payload = {
        "name": "vehicle_dynamics",
        "description": "Coordinated-turn radius, envelope limits, RK4 order.",
        "turn_radius_cases": rows,
        "turn_radius_worst_rel_error": worst,
        "limits": lim,
        "convergence": conv,
        "vehicle_params": vp.to_dict(),
        "checks": checks,
        "all_passed": all(checks.values()),
    }
    return payload


if __name__ == "__main__":
    report("validate_dynamics", run())
