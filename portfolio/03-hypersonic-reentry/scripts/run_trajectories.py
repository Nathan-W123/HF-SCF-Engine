#!/usr/bin/env python3
"""Run the production trajectories and write them to ``results/``.

Three products:

1. ``traj_ballistic.npz`` -- the nominal non-lifting entry (alpha = 0).
2. ``traj_lifting.npz``   -- the same entry state flown at the trim angle of
   attack with the lift vector up (bank = 0).
3. ``corridor.json``      -- a sweep over entry flight-path angle showing the
   shallow (skip-out) and steep (high-g) ends of the entry corridor.

All three feed the figures, the hero render and ``results/summary.json``.
"""

from __future__ import annotations

import json

import numpy as np

from _bootstrap import RESULTS_DIR, save_trajectory  # noqa: E402

from reentry import Vehicle, simulate  # noqa: E402
from reentry.atmosphere import USSA76  # noqa: E402
from reentry.trajectory import hot_wall_temperature  # noqa: E402

ENTRY_ALTITUDE = 120.0e3
ENTRY_VELOCITY = 7800.0
NOMINAL_GAMMA = -5.5
TERMINAL_ALTITUDE = 10.0e3
ALPHA_TRIM_DEG = -20.0  # sign convention: negative alpha => lift vector up


def build_cases():
    atm = USSA76()
    ballistic = Vehicle(alpha_trim_rad=0.0, name="70deg sphere-cone, ballistic")
    lifting = Vehicle(
        alpha_trim_rad=np.radians(ALPHA_TRIM_DEG),
        bank_rad=0.0,
        name=f"70deg sphere-cone, lifting (alpha={ALPHA_TRIM_DEG:g} deg, bank=0)",
    )
    return atm, ballistic, lifting


def _entry(atm, vehicle, gamma_deg, velocity):
    return simulate(
        vehicle=vehicle,
        atmosphere=atm,
        altitude0=ENTRY_ALTITUDE,
        velocity0=velocity,
        gamma0_deg=float(gamma_deg),
        terminal_altitude=TERMINAL_ALTITUDE,
        exit_altitude=ENTRY_ALTITUDE,
        rtol=1e-9,
        atol=1e-9,
        n_output=3001,
        t_max=6000.0,
    )


def find_skip_boundary(atm, vehicle, velocity, g_skip, g_capture, n_iter=18):
    """Bisect the entry flight-path angle that separates skip-out from capture."""
    lo, hi = g_skip, g_capture  # lo skips out, hi is captured (hi is steeper)
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        if _entry(atm, vehicle, mid, velocity).termination == "skip_out":
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def run_corridor(atm, vehicle, gammas, velocity=None):
    velocity = ENTRY_VELOCITY if velocity is None else velocity
    rows = []
    for g in gammas:
        res = simulate(
            vehicle=vehicle,
            atmosphere=atm,
            altitude0=ENTRY_ALTITUDE,
            velocity0=velocity,
            gamma0_deg=float(g),
            terminal_altitude=TERMINAL_ALTITUDE,
            exit_altitude=ENTRY_ALTITUDE,
            rtol=1e-9,
            atol=1e-9,
            n_output=3001,
            t_max=6000.0,
        )
        s = res.summary()
        rows.append(
            {
                "gamma0_deg": float(g),
                "termination": res.termination,
                "peak_g_load": s["peak_g_load"],
                "peak_heat_flux_W_cm2": s["peak_heat_flux_W_cm2"],
                "total_heat_load_J_cm2": s["total_heat_load_J_cm2"],
                "peak_dynamic_pressure_kPa": s["peak_dynamic_pressure_kPa"],
                "downrange_km": s["downrange_km"],
                "duration_s": s["duration_s"],
                "peak_heat_flux_altitude_km": s["peak_heat_flux_altitude_km"],
                "peak_g_altitude_km": s["peak_g_altitude_km"],
                "min_altitude_km": float(np.min(res.altitude) / 1e3),
                "peak_wall_temperature_K": s["peak_wall_temperature_K"],
                "exit_velocity_m_s": float(res.velocity[-1]),
                "entry_velocity_m_s": velocity,
            }
        )
    return rows


def main() -> int:
    atm, ballistic, lifting = build_cases()

    results = {}
    for tag, veh in (("ballistic", ballistic), ("lifting", lifting)):
        res = simulate(
            vehicle=veh,
            atmosphere=atm,
            altitude0=ENTRY_ALTITUDE,
            velocity0=ENTRY_VELOCITY,
            gamma0_deg=NOMINAL_GAMMA,
            terminal_altitude=TERMINAL_ALTITUDE,
            rtol=1e-10,
            atol=1e-10,
            n_output=6001,
        )
        save_trajectory(res, RESULTS_DIR / f"traj_{tag}.npz")
        s = res.summary()
        tw_hot = hot_wall_temperature(res)
        s["peak_wall_temperature_hot_wall_corrected_K"] = float(np.max(tw_hot))
        s["heat_load_dkr_over_sutton_graves"] = (
            s["total_heat_load_dkr_J_cm2"] / s["total_heat_load_J_cm2"]
        )
        s["peak_q_dkr_W_cm2"] = float(np.max(res.q_dot_dkr) / 1e4)
        s["peak_q_exp315_W_cm2"] = float(np.max(res.q_dot_exp315) / 1e4)
        s["peak_q_dkr_over_sutton_graves"] = (
            s["peak_q_dkr_W_cm2"] / s["peak_heat_flux_W_cm2"]
        )
        s["vehicle"] = veh.describe()
        results[tag] = s
        print(
            f"[{tag:9s}] peak q = {s['peak_heat_flux_W_cm2']:7.1f} W/cm^2 at "
            f"{s['peak_heat_flux_altitude_km']:5.1f} km, peak g = "
            f"{s['peak_g_load']:5.2f}, load = {s['total_heat_load_J_cm2']:7.0f} J/cm^2, "
            f"downrange = {s['downrange_km']:6.0f} km, T_w,max = "
            f"{s['peak_wall_temperature_K']:6.0f} K"
        )

    # --- corridor sweep 1: LEO entry speed (sub-circular) ---
    gammas = np.round(np.arange(-1.0, -12.01, -0.25), 3)
    leo = run_corridor(atm, ballistic, gammas, ENTRY_VELOCITY)
    leo_landed = [r for r in leo if r["termination"] == "terminal_altitude"]
    over10g = [r["gamma0_deg"] for r in leo_landed if r["peak_g_load"] > 10.0]
    leo_skip = [r["gamma0_deg"] for r in leo if r["termination"] == "skip_out"]

    # --- corridor sweep 2: super-circular (lunar-return-like) entry speed ---
    v_super = 11000.0
    gammas_s = np.round(np.arange(-1.0, -12.01, -0.25), 3)
    sup = run_corridor(atm, ballistic, gammas_s, v_super)
    sup_skip = [r["gamma0_deg"] for r in sup if r["termination"] == "skip_out"]
    sup_land = [r["gamma0_deg"] for r in sup if r["termination"] == "terminal_altitude"]
    boundary = None
    if sup_skip and sup_land:
        boundary = find_skip_boundary(atm, ballistic, v_super, max(sup_skip),
                                      min(sup_land))
    over10g_s = [r["gamma0_deg"] for r in sup
                 if r["termination"] == "terminal_altitude" and r["peak_g_load"] > 10.0]

    corridor_json = {
        "entry_altitude_km": ENTRY_ALTITUDE / 1e3,
        "vehicle": ballistic.describe(),
        "leo": {
            "entry_velocity_m_s": ENTRY_VELOCITY,
            "circular_speed_at_entry_m_s": float(
                np.sqrt(3.986004418e14 / (6371.0e3 + ENTRY_ALTITUDE))
            ),
            "skip_out_gammas_deg": leo_skip,
            "shallowest_gamma_exceeding_10g_deg": max(over10g) if over10g else None,
            "rows": leo,
        },
        "super_circular": {
            "entry_velocity_m_s": v_super,
            "skip_out_boundary_gamma_deg": boundary,
            "shallowest_gamma_exceeding_10g_deg": max(over10g_s) if over10g_s else None,
            "rows": sup,
        },
    }
    (RESULTS_DIR / "corridor.json").write_text(json.dumps(corridor_json, indent=2))
    print(
        f"[corridor ] LEO {ENTRY_VELOCITY:.0f} m/s: {len(leo)} cases, no skip-out"
        f" ({len(leo_skip)} skips); >10 g for gamma0 <= "
        f"{max(over10g) if over10g else 'n/a'} deg"
    )
    print(
        f"[corridor ] super-circular {v_super:.0f} m/s: skip-out boundary at "
        f"gamma0 = {boundary:.4f} deg" if boundary is not None else
        "[corridor ] super-circular: no boundary found"
    )

    (RESULTS_DIR / "cases.json").write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
