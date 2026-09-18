#!/usr/bin/env python3
"""Validation 2 -- Allen-Eggers closed-form ballistic entry.

The Allen-Eggers solution (NACA Report 1381, 1958) is the classical closed form
for ballistic entry into an exponential atmosphere, and it is the strongest
quantitative benchmark available to a reentry simulator because the peak
deceleration

    a_max = V_e^2 |sin gamma_e| / (2 e H)

depends on *nothing* but entry speed, entry flight-path angle and the
atmospheric scale height -- not on mass, area or drag coefficient.  The altitude
at which it occurs,

    z(a_max) = H ln( rho_0 H / (beta |sin gamma_e|) ),    beta = m/(C_D A),

does depend on the ballistic coefficient but not on the entry speed.

Anchoring.  Allen-Eggers is written for a vehicle arriving from *infinity* with
speed ``V_e``.  The simulator starts at a finite 120 km, where the exponential
atmosphere has already removed a fraction ``K exp(-z_0/H) ~ 3e-6`` of the speed.
The analytic solution is therefore anchored by inverting the closed form at the
simulator's initial state, ``V_e = V_0 exp(+K exp(-z_0/H))``, so that the two
solutions share the same initial condition exactly.  This removes a 3e-6 offset
that is a finite-entry-altitude artefact, not a physics difference.

Three configurations isolate the three Allen-Eggers assumptions:

**A. Exact limit.**  Exponential atmosphere, constant ``C_D``, no lift, gravity
off, no curvature term, flight-path angle frozen.  Here the simulator's ODE *is*
the Allen-Eggers ODE, so agreement must be at integrator precision.  This
validates the closed-form implementation and the integrator together.

**B. Constant-gamma assumption released.**  Gravity still off, but the spherical
geometry, the ``V^2/r`` curvature term and a freely evolving flight-path angle
are restored.  The only remaining difference from Allen-Eggers is that gamma is
no longer constant, so the error must vanish identically as the entry becomes
vertical (where there is nothing to turn) and grow as the entry becomes shallow.

**C. Full physics.**  Inverse-square gravity restored as well.  The residual
error no longer vanishes at vertical entry: gravity does work on the vehicle
between the entry interface and the peak-deceleration altitude, which
Allen-Eggers ignores.  The test asserts that the residual is *quantitatively
explained* by that work, ``2 g_bar (z_0 - z_peak) / V_e^2``.
"""

from __future__ import annotations

import numpy as np

from _bootstrap import check, finish, main  # noqa: E402

from reentry import Vehicle, simulate  # noqa: E402
from reentry.analytic import allen_eggers_deceleration, allen_eggers_peak  # noqa: E402
from reentry.atmosphere import ExponentialAtmosphere  # noqa: E402
from reentry.constants import G0, MU_EARTH, R_EARTH  # noqa: E402
from reentry.utils import refine_peak, rel_error  # noqa: E402

RHO0 = 1.2250
SCALE_HEIGHT = 7200.0
CD_CONST = 1.50
V_ENTRY = 7000.0
ALT_ENTRY = 120.0e3
GAMMAS = [-5.0, -10.0, -20.0, -30.0, -45.0, -60.0, -90.0]


def _anchored_ve(gamma_rad, beta):
    """Allen-Eggers ``V_e`` (speed at infinity) matching ``V_ENTRY`` at ``ALT_ENTRY``."""
    k = RHO0 * SCALE_HEIGHT / (2.0 * beta * abs(np.sin(gamma_rad)))
    return V_ENTRY * np.exp(k * np.exp(-ALT_ENTRY / SCALE_HEIGHT))


def _simulate_case(gamma_deg, *, gravity, idealised, n_output=20001):
    atm = ExponentialAtmosphere(rho0=RHO0, scale_height=SCALE_HEIGHT)
    return simulate(
        vehicle=Vehicle(),
        atmosphere=atm,
        altitude0=ALT_ENTRY,
        velocity0=V_ENTRY,
        gamma0_deg=gamma_deg,
        rtol=1e-11,
        atol=1e-11,
        terminal_altitude=10.0e3,
        exit_altitude=1.0e9,
        n_output=n_output,
        constant_cd=CD_CONST,
        include_gravity=gravity,
        curvature=not idealised,
        freeze_gamma=idealised,
    )


def _peak(res):
    return refine_peak(res.altitude, res.g_load * G0)


def run() -> dict:
    veh = Vehicle()
    beta = veh.mass / (CD_CONST * veh.area_ref)
    checks = []

    # ------------------------------------------------------------------ A
    gamma_a = -30.0
    ga = np.radians(gamma_a)
    ve_a = _anchored_ve(ga, beta)
    res_a = _simulate_case(gamma_a, gravity=False, idealised=True)
    z_sim, a_sim, _ = _peak(res_a)
    a_ref, z_ref, v_ref = allen_eggers_peak(ve_a, ga, beta, RHO0, SCALE_HEIGHT)
    # Cubic interpolation of V(z) -- linear interpolation on the ~200 m output
    # grid would itself contribute a 2e-5 error and mask the true agreement.
    from scipy.interpolate import CubicSpline

    v_of_z = CubicSpline(res_a.altitude[::-1], res_a.velocity[::-1])
    v_sim_at_peak = float(v_of_z(z_ref))
    a_an_profile = allen_eggers_deceleration(
        res_a.altitude, ve_a, ga, beta, RHO0, SCALE_HEIGHT
    )
    profile_dev = float(np.max(np.abs(res_a.g_load * G0 - a_an_profile)) / a_ref)

    checks.append(
        check("A. exact limit: peak deceleration rel. error",
              abs(rel_error(a_sim, a_ref)), 1e-6)
    )
    checks.append(
        check("A. exact limit: peak-altitude rel. error",
              abs(rel_error(z_sim, z_ref)), 1e-6)
    )
    checks.append(
        check("A. exact limit: V at analytic peak rel. error",
              abs(rel_error(v_sim_at_peak, v_ref)), 1e-6)
    )
    checks.append(
        check("A. exact limit: max |a_sim - a_AE| / a_max over profile",
              profile_dev, 1e-8)
    )

    # --------------------------------------------------------------- B, C
    sweep = []
    for gd in GAMMAS:
        g = np.radians(gd)
        ve = _anchored_ve(g, beta)
        a_ref_g, z_ref_g, _ = allen_eggers_peak(ve, g, beta, RHO0, SCALE_HEIGHT)
        row = {
            "gamma_deg": gd,
            "ve_anchored_m_s": float(ve),
            "a_max_analytic_m_s2": a_ref_g,
            "a_max_analytic_g": a_ref_g / G0,
            "z_peak_analytic_km": z_ref_g / 1e3,
        }
        for tag, gravity in (("B", False), ("C", True)):
            res = _simulate_case(gd, gravity=gravity, idealised=False)
            zp, ap, _ = _peak(res)
            row[f"{tag}_a_max_sim_m_s2"] = ap
            row[f"{tag}_a_max_sim_g"] = ap / G0
            row[f"{tag}_a_max_rel_error"] = rel_error(ap, a_ref_g)
            row[f"{tag}_z_peak_sim_km"] = zp / 1e3
            row[f"{tag}_z_peak_error_km"] = (zp - z_ref_g) / 1e3
            row[f"{tag}_termination"] = res.termination
        # Predicted gravity contribution: work done between entry and peak.
        g_bar = MU_EARTH / (R_EARTH + 0.5 * (ALT_ENTRY + z_ref_g)) ** 2
        row["gravity_work_prediction"] = float(
            2.0 * g_bar * (ALT_ENTRY - z_ref_g) / ve**2
        )
        row["C_residual_after_gravity_prediction"] = (
            row["C_a_max_rel_error"] - row["gravity_work_prediction"]
        )
        sweep.append(row)

    by = {r["gamma_deg"]: r for r in sweep}

    # B: the constant-gamma assumption is exact for vertical entry.
    checks.append(
        check("B. gravity off, gamma=-90 deg: |a_max| rel. error",
              abs(by[-90.0]["B_a_max_rel_error"]), 1e-6)
    )
    checks.append(
        check("B. gravity off, gamma=-45 deg: |a_max| rel. error",
              abs(by[-45.0]["B_a_max_rel_error"]), 0.02)
    )
    b_errs = [abs(by[g]["B_a_max_rel_error"]) for g in (-10.0, -20.0, -30.0, -45.0, -60.0, -90.0)]
    b_monotone = all(b_errs[i] > b_errs[i + 1] for i in range(len(b_errs) - 1))
    checks.append(
        check("B. |error| decreases monotonically with steepness",
              1.0 if b_monotone else 0.0, 0.5, comparison=">")
    )

    # C: full physics -- bounded error, and the residual is explained by gravity.
    steep = [r for r in sweep if r["gamma_deg"] <= -30.0]
    checks.append(
        check("C. full physics, gamma <= -30 deg: worst |a_max| rel. error",
              max(abs(r["C_a_max_rel_error"]) for r in steep), 0.06)
    )
    checks.append(
        check("C. full physics, gamma <= -30 deg: worst |z_peak| error",
              max(abs(r["C_z_peak_error_km"]) for r in steep), 0.5, units="km")
    )
    checks.append(
        check("C. gamma <= -45 deg: |error - gravity-work prediction|",
              max(abs(r["C_residual_after_gravity_prediction"])
                  for r in sweep if r["gamma_deg"] <= -45.0),
              0.012)
    )

    data = {
        "configuration": {
            "rho0_kg_m3": RHO0,
            "scale_height_m": SCALE_HEIGHT,
            "constant_cd": CD_CONST,
            "ballistic_coefficient_kg_m2": beta,
            "entry_velocity_m_s": V_ENTRY,
            "entry_altitude_km": ALT_ENTRY / 1e3,
            "terminal_altitude_km": 10.0,
        },
        "exact_limit": {
            "gamma_deg": gamma_a,
            "a_max_sim_m_s2": a_sim,
            "a_max_sim_g": a_sim / G0,
            "a_max_analytic_m_s2": a_ref,
            "a_max_analytic_g": a_ref / G0,
            "a_max_rel_error": rel_error(a_sim, a_ref),
            "z_peak_sim_km": z_sim / 1e3,
            "z_peak_analytic_km": z_ref / 1e3,
            "z_peak_rel_error": rel_error(z_sim, z_ref),
            "v_at_peak_sim_m_s": v_sim_at_peak,
            "v_at_peak_analytic_m_s": v_ref,
            "max_profile_deviation_over_amax": profile_dev,
        },
        "sweep": sweep,
    }
    return finish(
        "Validation 2: Allen-Eggers closed-form ballistic entry benchmark",
        __doc__,
        checks,
        data,
    )


if __name__ == "__main__":
    raise SystemExit(main(run, "val_02_allen_eggers"))
