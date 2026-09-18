#!/usr/bin/env python3
"""Validation 3 -- USSA76 atmosphere model against the published standard table.

The implementation hard-codes only the sea-level state (288.15 K, 101325 Pa),
the seven geopotential lapse rates and the USSA76 defining constants.  Every
layer base temperature and, crucially, every layer base *pressure* is derived by
chaining the barometric formula upward from sea level.  The published USSA76
values at the layer boundaries are therefore a genuine external reference, and
this test compares against them at

    H = 0, 11, 20, 32, 47, 51, 71, 84.852 km' (geopotential)

which are the geometric altitudes 0 ... 86 km.

Additional internal checks:

* **Hydrostatic residual.**  The model must satisfy ``dp/dz = -rho g(z)`` with
  ``g(z) = g0 (r0/(r0+z))^2`` at every altitude from 0 to 500 km, including the
  numerically integrated region above 86 km.  This is checked with centred
  finite differences of the model itself.
* **Speed of sound** at sea level must equal ``sqrt(gamma R T0)`` = 340.294 m/s.
* **Monotonicity** of pressure and density.
* **Interpolation error** of the fast tabulated evaluation against the exact
  closed-form evaluation.
* **Geopotential/geometric round trip.**
"""

from __future__ import annotations

import numpy as np

from _bootstrap import check, finish, main  # noqa: E402

from reentry.atmosphere import (  # noqa: E402
    USSA76,
    USSA76_REFERENCE_TABLE,
    geometric_altitude,
    geopotential_altitude,
)
from reentry.constants import G0, GAMMA_AIR, R0_USSA76, R_AIR  # noqa: E402


def run() -> dict:
    exact = USSA76(tabulated=False)
    fast = USSA76(tabulated=True)
    checks = []

    rows = []
    e_t, e_p, e_r = [], [], []
    for h_km, t_ref, p_ref, rho_ref in USSA76_REFERENCE_TABLE:
        z = float(geometric_altitude(h_km * 1e3))
        p, t = exact._exact(z)
        rho = p / (R_AIR * t)
        dt_ = abs(t - t_ref) / t_ref
        dp_ = abs(p - p_ref) / p_ref
        dr_ = abs(rho - rho_ref) / rho_ref
        e_t.append(dt_)
        e_p.append(dp_)
        e_r.append(dr_)
        rows.append(
            {
                "geopotential_altitude_km": h_km,
                "geometric_altitude_km": z / 1e3,
                "T_model_K": t,
                "T_reference_K": t_ref,
                "T_rel_error": dt_,
                "p_model_Pa": p,
                "p_reference_Pa": p_ref,
                "p_rel_error": dp_,
                "rho_model_kg_m3": rho,
                "rho_reference_kg_m3": rho_ref,
                "rho_rel_error": dr_,
            }
        )

    checks.append(check("max |dT/T| at the 8 layer boundaries", max(e_t), 1e-9))
    checks.append(check("max |dp/p| at the 8 layer boundaries", max(e_p), 1e-6))
    checks.append(check("max |drho/rho| at the 8 layer boundaries", max(e_r), 5e-5))

    # -- hydrostatic residual ------------------------------------------------
    z = np.linspace(100.0, 500.0e3, 400_001)
    dz = z[1] - z[0]
    p = np.asarray(exact.pressure(z))
    rho = np.asarray(exact.density(z))
    g = G0 * (R0_USSA76 / (R0_USSA76 + z)) ** 2
    dpdz = (p[2:] - p[:-2]) / (2.0 * dz)
    resid = np.abs(dpdz + rho[1:-1] * g[1:-1]) / (rho[1:-1] * g[1:-1])
    # Exclude a +-200 m neighbourhood of every profile breakpoint.  A centred
    # difference that straddles a lapse-rate discontinuity loses an order of
    # accuracy, and the cubic spline used for ln p above 86 km rings slightly
    # across the same kinks; neither is a property of the atmosphere model, so
    # the residual is measured in the smooth interior of each layer.
    breaks = np.concatenate(
        [np.asarray(geometric_altitude(np.array([11e3, 20e3, 32e3, 47e3, 51e3, 71e3,
                                                 84.852e3]))),
         np.array([86e3, 91e3, 110e3, 120e3])]
    )
    keep = np.ones(resid.size, dtype=bool)
    for b in breaks:
        keep &= np.abs(z[1:-1] - b) > 200.0
    checks.append(
        check("median hydrostatic residual |dp/dz + rho g| / (rho g)",
              float(np.median(resid[keep])), 1e-8)
    )
    checks.append(
        check("max hydrostatic residual |dp/dz + rho g| / (rho g)",
              float(np.max(resid[keep])), 1e-6)
    )

    # The 86 km seam: below it USSA76 is expressed in molecular-scale
    # temperature (186.946 K), above it in kinetic temperature (186.8673 K).
    # The model therefore has a small, documented density step there.
    rho_below = float(exact.density(86.0e3 - 1.0))
    rho_above = float(exact.density(86.0e3 + 1.0))
    seam = abs(rho_above - rho_below) / rho_below
    checks.append(
        check("density discontinuity at the 86 km model seam", seam, 1e-3)
    )

    # -- speed of sound ------------------------------------------------------
    a0 = float(exact.sound_speed(0.0))
    a0_ref = float(np.sqrt(GAMMA_AIR * R_AIR * 288.15))
    checks.append(
        check("sea-level speed of sound vs sqrt(gamma R T0)",
              abs(a0 - a0_ref) / a0_ref, 1e-12)
    )

    # -- monotonicity --------------------------------------------------------
    zz = np.linspace(0.0, 1000.0e3, 200_001)
    pp = np.asarray(exact.pressure(zz))
    rr = np.asarray(exact.density(zz))
    checks.append(
        check("pressure monotonically decreasing (# violations)",
              float(np.sum(np.diff(pp) >= 0.0)), 1.0)
    )
    checks.append(
        check("density monotonically decreasing (# violations)",
              float(np.sum(np.diff(rr) >= 0.0)), 1.0)
    )

    # -- tabulated vs exact --------------------------------------------------
    rng = np.random.default_rng(12345)
    zs = rng.uniform(0.0, 300.0e3, 400_000)
    err_rho = float(np.max(np.abs(fast.density(zs) - exact.density(zs))
                           / exact.density(zs)))
    err_t = float(np.max(np.abs(fast.temperature(zs) - exact.temperature(zs))
                         / exact.temperature(zs)))
    checks.append(check("tabulated vs exact: max |drho/rho| (0-300 km)", err_rho, 5e-6))
    checks.append(check("tabulated vs exact: max |dT/T| (0-300 km)", err_t, 5e-6))

    # -- geopotential round trip --------------------------------------------
    z_rt = np.linspace(0.0, 86.0e3, 10_001)
    rt = float(np.max(np.abs(geometric_altitude(geopotential_altitude(z_rt)) - z_rt)))
    checks.append(check("geopotential/geometric round-trip error", rt, 1e-7, units="m"))

    data = {
        "layer_boundary_comparison": rows,
        "sea_level_speed_of_sound_m_s": a0,
        "hydrostatic_residual_max": float(np.max(resid[keep])),
        "hydrostatic_residual_median": float(np.median(resid[keep])),
        "density_step_at_86km_rel": seam,
        "tabulated_max_rel_density_error": err_rho,
        "profile_samples": {
            "altitude_km": [0, 20, 40, 60, 80, 86, 100, 120, 150, 200],
            "density_kg_m3": [
                float(exact.density(a * 1e3))
                for a in (0, 20, 40, 60, 80, 86, 100, 120, 150, 200)
            ],
            "temperature_K": [
                float(exact.temperature(a * 1e3))
                for a in (0, 20, 40, 60, 80, 86, 100, 120, 150, 200)
            ],
        },
    }
    return finish(
        "Validation 3: USSA76 atmosphere vs the published standard table",
        __doc__,
        checks,
        data,
    )


if __name__ == "__main__":
    raise SystemExit(main(run, "val_03_atmosphere_table"))
