#!/usr/bin/env python3
"""Validation 1 -- vacuum orbital behaviour.

With the atmosphere switched off the 3-DOF flight equations must reduce to pure
Keplerian motion.  Three independent things are checked:

A. **Circular orbit, non-rotating Earth.**  Specific orbital energy
   ``E = V^2/2 - mu/r`` and specific angular momentum ``|r x v|`` must be
   conserved, and the radius must not drift.
B. **Period.**  Propagating for exactly one Keplerian period
   ``T = 2 pi sqrt(a^3/mu)`` must return the vehicle to its starting point; the
   closure error is reported as a distance.
C. **Rotating Earth.**  With ``omega = OMEGA_EARTH`` the equations acquire
   Coriolis and centrifugal terms.  An inclined, eccentric orbit is used so that
   no term is accidentally zero, and three things are required: the *inertial*
   energy and angular momentum must be conserved, the orbit-plane normal must
   not drift, and -- the decisive test -- the whole trajectory must agree with a
   completely independent inertial Cartesian two-body propagation started from
   the same converted initial state.  The Cartesian propagator shares no code
   with the flight equations, so any sign error in a Coriolis or centrifugal
   term shows up immediately as a position divergence.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from _bootstrap import check, finish, main  # noqa: E402

from reentry import simulate, Vehicle  # noqa: E402
from reentry.atmosphere import VacuumAtmosphere  # noqa: E402
from reentry.constants import MU_EARTH, OMEGA_EARTH, R_EARTH  # noqa: E402
from reentry.dynamics import specific_orbital_elements, state_to_cartesian  # noqa: E402


def _drift(x):
    x = np.asarray(x, dtype=float)
    return float(np.max(np.abs(x - x[0])) / abs(x[0]))


def run() -> dict:
    vac = VacuumAtmosphere()
    veh = Vehicle()
    checks = []
    data = {}

    # ---------------- A/B: circular orbit, non-rotating ----------------
    alt0 = 400.0e3
    r0 = R_EARTH + alt0
    v_circ = float(np.sqrt(MU_EARTH / r0))
    period = float(2.0 * np.pi * np.sqrt(r0**3 / MU_EARTH))

    res = simulate(
        vehicle=veh,
        atmosphere=vac,
        altitude0=alt0,
        velocity0=v_circ,
        gamma0_deg=0.0,
        psi0_deg=90.0,
        lat0_deg=0.0,
        t_max=3.0 * period,
        rtol=1e-12,
        atol=1e-12,
        terminal_altitude=-R_EARTH,
        exit_altitude=1.0e10,
        n_output=6001,
    )
    pos, vel = res.cartesian(omega=0.0)
    energy, hvec, sma = specific_orbital_elements(pos, vel)
    hmag = np.linalg.norm(hvec, axis=-1)

    checks.append(check("A. circular: |dE/E| drift over 3 periods", _drift(energy), 1e-11))
    checks.append(check("A. circular: |dh/h| drift over 3 periods", _drift(hmag), 1e-11))
    checks.append(
        check("A. circular: radius drift |dr|", float(np.max(np.abs(res.radius - r0))),
              1e-3, units="m")
    )
    checks.append(
        check(
            "A. circular: |a - r0| semi-major axis error",
            float(np.max(np.abs(sma - r0))),
            1e-3,
            units="m",
        )
    )

    # Period closure: propagate exactly one Keplerian period.
    res_p = simulate(
        vehicle=veh, atmosphere=vac, altitude0=alt0, velocity0=v_circ, gamma0_deg=0.0,
        psi0_deg=90.0, lat0_deg=0.0, t_max=period, rtol=1e-12, atol=1e-12,
        terminal_altitude=-R_EARTH, exit_altitude=1.0e10, n_output=3,
    )
    p0, _ = state_to_cartesian(res_p.radius[0], res_p.lon[0], res_p.lat[0],
                              res_p.velocity[0], res_p.gamma[0], res_p.psi[0])
    p1, _ = state_to_cartesian(res_p.radius[-1], res_p.lon[-1], res_p.lat[-1],
                               res_p.velocity[-1], res_p.gamma[-1], res_p.psi[-1])
    closure = float(np.linalg.norm(p1 - p0))
    checks.append(check("B. period closure error after 2*pi*sqrt(a^3/mu)", closure, 1.0,
                        units="m"))
    checks.append(
        check("B. period closure / circumference", closure / (2 * np.pi * r0), 1e-8)
    )

    # ---------------- C: rotating Earth, inclined eccentric orbit -------
    alt_c = 500.0e3
    r_c = R_EARTH + alt_c
    v_rel = 7600.0
    res_r = simulate(
        vehicle=veh, atmosphere=vac, altitude0=alt_c, velocity0=v_rel,
        gamma0_deg=6.0, psi0_deg=40.0, lat0_deg=20.0, lon0_deg=-30.0,
        omega=OMEGA_EARTH, t_max=3000.0, rtol=1e-12, atol=1e-12,
        terminal_altitude=-R_EARTH, exit_altitude=1.0e10, n_output=4001,
    )
    pos_r, vel_r = res_r.cartesian(omega=OMEGA_EARTH)
    e_r, h_r, a_r = specific_orbital_elements(pos_r, vel_r)
    hmag_r = np.linalg.norm(h_r, axis=-1)
    # Direction of the angular-momentum vector must be fixed too (orbit plane).
    hhat = h_r / hmag_r[:, None]
    plane_drift = float(np.max(np.linalg.norm(hhat - hhat[0], axis=-1)))

    checks.append(check("C. rotating: inertial |dE/E| drift", _drift(e_r), 1e-10))
    checks.append(check("C. rotating: inertial |dh/h| drift", _drift(hmag_r), 1e-10))
    checks.append(check("C. rotating: orbit-plane normal drift", plane_drift, 1e-9))

    # Independent inertial Cartesian two-body propagation from the same state.
    def two_body(t, y):
        rv = y[:3]
        rn = np.linalg.norm(rv)
        return np.concatenate([y[3:], -MU_EARTH * rv / rn**3])

    sol_cart = solve_ivp(
        two_body,
        (0.0, float(res_r.t[-1])),
        np.concatenate([pos_r[0], vel_r[0]]),
        method="DOP853",
        rtol=1e-13,
        atol=1e-9,
        t_eval=res_r.t,
    )
    pos_err = float(np.max(np.linalg.norm(sol_cart.y[:3].T - pos_r, axis=1)))
    vel_err = float(np.max(np.linalg.norm(sol_cart.y[3:].T - vel_r, axis=1)))
    checks.append(
        check("C. rotating: max |dr| vs inertial Cartesian 2-body", pos_err, 1.0e-2,
              units="m")
    )
    checks.append(
        check("C. rotating: max |dv| vs inertial Cartesian 2-body", vel_err, 1.0e-5,
              units="m/s")
    )

    data = {
        "circular_orbit": {
            "altitude_km": alt0 / 1e3,
            "v_circular_m_s": v_circ,
            "kepler_period_s": period,
            "period_closure_error_m": closure,
            "energy_drift_rel": _drift(energy),
            "angular_momentum_drift_rel": _drift(hmag),
            "specific_energy_J_per_kg": float(energy[0]),
        },
        "rotating_orbit": {
            "omega_rad_s": OMEGA_EARTH,
            "altitude0_km": alt_c / 1e3,
            "relative_speed_m_s": v_rel,
            "gamma0_deg": 6.0,
            "inclination_deg": float(np.degrees(np.arccos(abs(hhat[0, 2])))),
            "semi_major_axis_km": float(a_r[0] / 1e3),
            "inertial_energy_drift_rel": _drift(e_r),
            "inertial_h_drift_rel": _drift(hmag_r),
            "orbit_plane_normal_drift": plane_drift,
            "max_position_error_vs_cartesian_2body_m": pos_err,
            "max_velocity_error_vs_cartesian_2body_m_s": vel_err,
            "propagation_time_s": float(res_r.t[-1]),
        },
    }

    return finish(
        "Validation 1: vacuum orbital energy, angular momentum and period",
        __doc__,
        checks,
        data,
    )


if __name__ == "__main__":
    raise SystemExit(main(run, "val_01_orbit_energy"))
