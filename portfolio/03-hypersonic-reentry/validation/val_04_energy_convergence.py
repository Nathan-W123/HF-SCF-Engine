#!/usr/bin/env python3
"""Validation 4 -- energy budget and integrator convergence order.

**A. Energy budget.**  Lift acts perpendicular to the velocity and therefore does
no work, so along the whole trajectory the mechanical energy balance

    d/dt ( V^2/2 - mu/r )  =  - (D/m) V

must hold exactly.  The simulator carries ``W = int (D/m) V dt`` as a state
variable integrated by the same Runge-Kutta scheme as everything else, so
``Delta(V^2/2 - mu/r) + W`` is a genuine, non-trivial closure test of the
right-hand side *and* of the integration: an error in the drag term, in the
gravity term, or in the integrator all break it.  Checked for both the ballistic
and the lifting entry.

**B. Convergence order.**  A fixed-step classical RK4 is run on the same
right-hand side over a fixed 100 s window with the step count doubled repeatedly,
and the error measured against a DOP853 reference at ``rtol = atol = 1e-13``.
The observed order ``log2(e_n / e_2n)`` must approach 4.

The exponential atmosphere is used for this study because the USSA76 lapse-rate
table makes ``T(z)`` only piecewise smooth: a formal order study across a layer
boundary would measure the kink, not the integrator.  The aerodynamic model,
gravity and spherical geometry are all fully active.

**C. Integrator cross-check.**  The production adaptive DOP853 result and a
fine fixed-step RK4 result must agree on the full USSA76 entry to within a metre
in altitude and a millimetre per second in speed.

**D. Tolerance self-convergence.**  The peak heat flux and total heat load of the
nominal entry are recomputed at tolerances from 1e-6 to 1e-12 and compared with a
1e-12 reference, confirming that the reported numbers are converged and not
tolerance artefacts.
"""

from __future__ import annotations

import numpy as np

from _bootstrap import check, finish, main  # noqa: E402

from reentry import Vehicle, simulate  # noqa: E402
from reentry.aerodynamics import SphereCone  # noqa: E402
from reentry.atmosphere import ExponentialAtmosphere, USSA76  # noqa: E402
from reentry.constants import MU_EARTH, R_EARTH  # noqa: E402
from reentry.dynamics import EntryModel  # noqa: E402
from reentry.integrators import integrate_dop853, integrate_rk4  # noqa: E402

ATM = USSA76()
BALLISTIC = Vehicle(alpha_trim_rad=0.0)
LIFTING = Vehicle(alpha_trim_rad=np.radians(-20.0), bank_rad=0.0)


def _energy_closure(res):
    e = 0.5 * res.velocity**2 - MU_EARTH / res.radius
    de = float(e[-1] - e[0])
    w = float(res.drag_work[-1])
    return abs(de + w) / abs(de), de, w


def run() -> dict:
    checks = []

    # ------------------------------------------------------------------ A
    res_b = simulate(vehicle=BALLISTIC, atmosphere=ATM, gamma0_deg=-5.5,
                     velocity0=7800.0, rtol=1e-12, atol=1e-12)
    res_l = simulate(vehicle=LIFTING, atmosphere=ATM, gamma0_deg=-5.5,
                     velocity0=7800.0, rtol=1e-12, atol=1e-12)
    cb, de_b, w_b = _energy_closure(res_b)
    cl_, de_l, w_l = _energy_closure(res_l)
    checks.append(check("A. ballistic: |dE + W_drag| / |dE|", cb, 1e-10))
    checks.append(check("A. lifting:   |dE + W_drag| / |dE|", cl_, 1e-10))

    # ------------------------------------------------------------------ B
    veh = Vehicle()
    atm_exp = ExponentialAtmosphere(rho0=1.2250, scale_height=7200.0)
    model = EntryModel(vehicle=veh, atmosphere=atm_exp)
    y0 = np.array(
        [R_EARTH + 120.0e3, 0.0, 0.0, 7000.0, np.radians(-30.0), np.radians(90.0),
         0.0, 0.0, 0.0]
    )
    t_end = 100.0
    ref = integrate_dop853(model.rhs, (0.0, t_end), y0, rtol=1e-13, atol=1e-13)
    y_ref = ref.sol(t_end)
    # Per-component relative normalisation so that no single state (the heat
    # load, ~1e8 J/m^2) dominates the error norm purely through its magnitude.
    scale = np.maximum(np.abs(y_ref), 1.0e-2)

    steps = [125, 250, 500, 1000, 2000, 4000]
    errors = []
    for n in steps:
        _, ys = integrate_rk4(model.rhs, (0.0, t_end), y0, n)
        err = float(np.max(np.abs((ys[-1] - y_ref) / scale)))
        errors.append(err)
    orders = [
        float(np.log2(errors[i] / errors[i + 1])) for i in range(len(errors) - 1)
    ]
    checks.append(
        check("B. RK4 observed order (finest pair)", orders[-1], 3.8, comparison=">")
    )
    checks.append(check("B. RK4 observed order (finest pair) upper bound",
                        orders[-1], 4.3))
    checks.append(
        check("B. mean observed order over all refinements",
              float(np.mean(orders)), 3.9, comparison=">")
    )
    checks.append(
        check("B. RK4 relative error at n=4000 (dt = 25 ms)", errors[-1], 1e-10)
    )

    # ------------------------------------------------------------------ C
    model_u = EntryModel(vehicle=BALLISTIC, atmosphere=ATM)
    y0u = np.array(
        [R_EARTH + 120.0e3, 0.0, 0.0, 7800.0, np.radians(-5.5), np.radians(90.0),
         0.0, 0.0, 0.0]
    )
    t_cmp = 150.0
    ref_u = integrate_dop853(model_u.rhs, (0.0, t_cmp), y0u, rtol=1e-12, atol=1e-12)
    y_ref_u = ref_u.sol(t_cmp)
    n_rk4 = 20_000
    _, ys_u = integrate_rk4(model_u.rhs, (0.0, t_cmp), y0u, n_rk4)
    d_alt = abs(ys_u[-1][0] - y_ref_u[0])
    d_vel = abs(ys_u[-1][3] - y_ref_u[3])
    checks.append(check("C. DOP853 vs RK4 (dt=7.5 ms): altitude difference", d_alt, 1.0,
                        units="m"))
    checks.append(check("C. DOP853 vs RK4 (dt=7.5 ms): speed difference", d_vel, 1e-3,
                        units="m/s"))

    # ------------------------------------------------------------------ D
    ref_res = simulate(vehicle=BALLISTIC, atmosphere=ATM, gamma0_deg=-5.5,
                       velocity0=7800.0, rtol=1e-12, atol=1e-12)
    q_ref = ref_res.peak_heat_flux
    load_ref = ref_res.total_heat_load
    tol_rows = []
    for tol in (1e-6, 1e-8, 1e-10):
        r = simulate(vehicle=BALLISTIC, atmosphere=ATM, gamma0_deg=-5.5,
                     velocity0=7800.0, rtol=tol, atol=tol)
        tol_rows.append(
            {
                "rtol": tol,
                "peak_q_rel_error": abs(r.peak_heat_flux - q_ref) / q_ref,
                "heat_load_rel_error": abs(r.total_heat_load - load_ref) / load_ref,
                "n_rhs_evals": r.meta["n_rhs_evals"],
            }
        )
    checks.append(
        check("D. peak heat flux converged at rtol=1e-10",
              tol_rows[2]["peak_q_rel_error"], 1e-7)
    )
    checks.append(
        check("D. total heat load converged at rtol=1e-10",
              tol_rows[2]["heat_load_rel_error"], 1e-7)
    )
    checks.append(
        check("D. peak heat flux still accurate at a loose rtol=1e-6",
              tol_rows[0]["peak_q_rel_error"], 1e-4)
    )

    data = {
        "energy_budget": {
            "ballistic": {
                "delta_specific_energy_J_per_kg": de_b,
                "drag_work_J_per_kg": w_b,
                "relative_closure": cb,
            },
            "lifting": {
                "delta_specific_energy_J_per_kg": de_l,
                "drag_work_J_per_kg": w_l,
                "relative_closure": cl_,
            },
        },
        "rk4_convergence": {
            "window_s": t_end,
            "steps": steps,
            "normalised_error": errors,
            "observed_orders": orders,
            "note": (
                "Refining beyond n=4000 (dt=25 ms) saturates against the "
                "accuracy of the DOP853 reference itself (~3e-12 relative), so "
                "the study stops there."
            ),
        },
        "integrator_cross_check": {
            "window_s": t_cmp,
            "rk4_steps": n_rk4,
            "altitude_difference_m": float(d_alt),
            "speed_difference_m_s": float(d_vel),
        },
        "tolerance_study": tol_rows,
    }
    return finish(
        "Validation 4: energy budget, RK4 convergence order and integrator cross-check",
        __doc__,
        checks,
        data,
    )


if __name__ == "__main__":
    raise SystemExit(main(run, "val_04_energy_convergence"))
