"""Validation 1 -- two-body Kepler orbit and second-order convergence.

The same kick-drift-kick leapfrog and the same (unsoftened) direct force
kernel used by the production run are applied to an isolated two-body problem
whose exact solution is known.  Two things are checked:

  1. the orbit is accurate in absolute terms, and
  2. halving dt reduces the error by ~4x, i.e. the *measured* convergence
     order is 2, which is the defining property of the leapfrog scheme.

The error metric is the maximum over one full orbital period of the separation
between the integrated and the analytic relative position vector, normalised
by the semi-major axis.
"""
from __future__ import annotations

import numpy as np

from _common import report
from galcol.integrator import DirectGravity, leapfrog_kdk
from galcol.units import G, TIME_UNIT_GYR

A = 10.0            # semi-major axis [kpc]
ECC = 0.5
M1 = 1.0            # 1e10 Msun
M2 = 1.0


def kepler_exact(t, a, ecc, mu):
    """Relative position on a Kepler ellipse, pericentre at t=0."""
    n = np.sqrt(mu / a ** 3)
    M = n * np.asarray(t)
    E = M.copy()
    for _ in range(80):                       # Newton on Kepler's equation
        f = E - ecc * np.sin(E) - M
        E = E - f / (1.0 - ecc * np.cos(E))
    x = a * (np.cos(E) - ecc)
    y = a * np.sqrt(1.0 - ecc ** 2) * np.sin(E)
    return np.stack([x, y, np.zeros_like(x)], axis=-1)


def run(n_steps):
    mu = G * (M1 + M2)
    period = 2.0 * np.pi * np.sqrt(A ** 3 / mu)
    r_p = A * (1.0 - ECC)
    v_p = np.sqrt(mu * (1.0 + ECC) / (A * (1.0 - ECC)))

    mass = np.array([M1, M2])
    # place the pair about a stationary centre of mass
    f1 = M2 / (M1 + M2)
    f2 = M1 / (M1 + M2)
    pos = np.array([[r_p * f1, 0.0, 0.0], [-r_p * f2, 0.0, 0.0]])
    vel = np.array([[0.0, v_p * f1, 0.0], [0.0, -v_p * f2, 0.0]])
    eps2 = np.zeros(2)                        # unsoftened: exact Newtonian

    dt = period / n_steps
    rec_t, rec_r = [], []

    class CB:
        wants_potential = False

        def __call__(self, step, t, p, v, a, pot):
            rec_t.append(t)
            rec_r.append(p[0] - p[1])

    leapfrog_kdk(pos, vel, mass, eps2, DirectGravity(), dt, n_steps, callback=CB())
    rec_t = np.asarray(rec_t)
    rec_r = np.asarray(rec_r)
    exact = kepler_exact(rec_t, A, ECC, mu)
    err = np.linalg.norm(rec_r - exact, axis=1) / A
    return dict(n_steps=n_steps, dt=dt,
                dt_myr=dt * TIME_UNIT_GYR * 1000.0,
                period=period, period_gyr=period * TIME_UNIT_GYR,
                max_err=float(err.max()), final_err=float(err[-1]),
                rms_err=float(np.sqrt(np.mean(err ** 2))))


def main():
    ladder = [400, 800, 1600, 3200, 6400]
    runs = [run(n) for n in ladder]
    dts = np.array([r["dt"] for r in runs])
    errs = np.array([r["max_err"] for r in runs])
    slope, _ = np.polyfit(np.log(dts), np.log(errs), 1)
    # pairwise ratios: each halving of dt should reduce the error by ~4
    ratios = [float(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]

    # The physically meaningful row is the one whose timestep matches the
    # production run (dt = 0.978 Myr); the threshold is "better than 1% of the
    # semi-major axis after a complete orbit".
    prod_dt_myr = 0.978
    k_prod = int(np.argmin([abs(r["dt_myr"] - prod_dt_myr) for r in runs]))
    prod = runs[k_prod]

    checks = [
        (f"max |r_sim - r_exact| / a over 1 orbit at dt={prod['dt_myr']:.2f} Myr "
         f"(production dt)", f"{prod['max_err']:.3e}", "< 1e-2",
         prod["max_err"] < 1e-2),
        ("measured convergence order (slope of log err vs log dt)",
         f"{slope:.4f}", "2.00 +/- 0.15", abs(slope - 2.0) < 0.15),
        ("error-reduction factor per dt halving (min over ladder)",
         f"{min(ratios):.3f}", "> 3.6", min(ratios) > 3.6),
        ("max |r_sim - r_exact| / a at the finest dt "
         f"({runs[-1]['dt_myr']:.3f} Myr)", f"{errs[-1]:.3e}", "< 1e-3",
         errs[-1] < 1e-3),
    ]
    extra = {
        "semi_major_axis_kpc": A, "eccentricity": ECC,
        "masses_1e10_msun": [M1, M2],
        "period_gyr": runs[0]["period_gyr"],
        "convergence_slope": float(slope),
        "production_dt_row": prod,
        "error_ratios": ratios,
        "ladder": runs,
    }
    ok, _ = report("kepler_two_body", checks, extra)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
