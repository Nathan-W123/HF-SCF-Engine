"""Validations 3 and 4 -- energy and angular-momentum conservation.

Part A reads the diagnostics written during the production collision run and
reports

    max |dE| / |E_0|    and    max |dL| / |L_0|

over the full 2.2 Gyr, together with the best-fit *secular* drift rate.  For a
symplectic integrator the energy error must be bounded and oscillatory, not
growing: the leapfrog conserves a shadow Hamiltonian that differs from the
true one by O(dt^2), so the error wobbles at the orbital frequency around a
fixed offset instead of accumulating.  The test is therefore not only "is the
error small" but "does a linear trend explain it" -- it should not.

Part B makes that claim concrete by integrating the *same* small self-
gravitating galaxy twice with two different second-order schemes:

    * the production kick-drift-kick leapfrog (symplectic), and
    * explicit midpoint / RK2 (same formal order, not symplectic).

Both have O(dt^2) local truncation error, so any difference in the long-term
energy behaviour is due to symplecticity alone.
"""
from __future__ import annotations

import json

import numpy as np

from _common import DATA, report
from galcol.integrator import (DirectGravity, angular_momentum,
                               kinetic_energy, leapfrog_kdk,
                               potential_energy)
from galcol.ics import Resolution, make_isolated
from galcol.production import MODEL
from galcol.profiles import default_galaxy
from galcol.simulate import angmom_error, energy_error
from galcol.units import TIME_UNIT_GYR

TOL_ENERGY = 5.0e-3
TOL_ANGMOM = 1.0e-2
TOL_SECULAR_FRACTION = 1.0     # linear trend must not account for the whole error


# --------------------------------------------------------------------------
def _rk2_energy_trace(pos, vel, mass, eps2, gravity, dt, n_steps, sample=4):
    """Explicit-midpoint (RK2) integration; returns (t, E) samples."""
    pos = pos.copy(); vel = vel.copy()
    ts, es = [], []
    for step in range(n_steps + 1):
        if step % sample == 0:
            a, pot = gravity(pos, mass, eps2, want_potential=True)
            es.append(kinetic_energy(mass, vel) + potential_energy(mass, pot))
            ts.append(step * dt)
        a, _ = gravity(pos, mass, eps2)
        pm = pos + 0.5 * dt * vel
        vm = vel + 0.5 * dt * a
        am, _ = gravity(pm, mass, eps2)
        pos = pos + dt * vm
        vel = vel + dt * am
    return np.asarray(ts), np.asarray(es)


def _kdk_energy_trace(pos, vel, mass, eps2, gravity, dt, n_steps, sample=4):
    pos = pos.copy(); vel = vel.copy()
    ts, es = [], []

    class CB:
        wants_potential = True

        def __call__(self, step, t, p, v, a, pot):
            if step % sample == 0 and pot is not None:
                ts.append(t)
                es.append(kinetic_energy(mass, v) + potential_energy(mass, pot))

    leapfrog_kdk(pos, vel, mass, eps2, gravity, dt, n_steps, callback=CB())
    return np.asarray(ts), np.asarray(es)


def _drift_stats(t, e):
    e0 = e[0]
    rel = (e - e0) / abs(e0)
    slope = float(np.polyfit(t, rel, 1)[0]) if t.size > 2 else 0.0
    span = float(t[-1] - t[0])
    return {
        "max_abs_rel": float(np.abs(rel).max()),
        "drift_per_gyr": slope / TIME_UNIT_GYR,
        "total_linear_drift": abs(slope * span),
        "secular_fraction": (abs(slope * span) / max(np.abs(rel).max(), 1e-300)),
    }


def symplectic_comparison(t_end_gyr=1.0, dt=0.002, seed=11):
    """Part B: leapfrog vs RK2 on the same small live-halo galaxy."""
    model = default_galaxy()
    res = Resolution(n_disk=800, n_bulge=250, n_halo=1000,
                     eps_disk=0.5, eps_bulge=0.5, eps_halo=1.2)
    p = make_isolated(model, res, seed=seed, live_halo=True)
    n_steps = int(round((t_end_gyr / TIME_UNIT_GYR) / dt))
    g = DirectGravity()
    t1, e1 = _kdk_energy_trace(p.pos, p.vel, p.mass, p.eps2, g, dt, n_steps)
    t2, e2 = _rk2_energy_trace(p.pos, p.vel, p.mass, p.eps2, g, dt, n_steps)
    return {
        "n_particles": int(p.n), "dt_code": dt, "t_end_gyr": t_end_gyr,
        "n_steps": n_steps,
        "leapfrog": _drift_stats(t1, e1),
        "rk2": _drift_stats(t2, e2),
        "trace": {"t_gyr": (t1 * TIME_UNIT_GYR).tolist(),
                  "leapfrog_rel": ((e1 - e1[0]) / abs(e1[0])).tolist(),
                  "rk2_rel": ((e2 - e2[0]) / abs(e2[0])).tolist()},
    }


def main(diag_path=None):
    diag_path = diag_path or (DATA / "collision" / "diagnostics.npz")
    d = np.load(diag_path, allow_pickle=True)
    diag = {k: d[k] for k in d.files if k not in ("meta", "step_times")}
    meta = json.loads(str(d["meta"]))

    ee = energy_error(diag)
    le = angmom_error(diag)
    st = _drift_stats(diag["t"], diag["etot"])

    # Linear momentum: the initial conditions are exactly at rest, and a
    # Barnes-Hut force is only approximately antisymmetric, so this measures
    # how much spurious momentum the tree injects.
    P = np.stack([diag["px"], diag["py"], diag["pz"]], axis=1)
    m_total = 2.0 * (MODEL.disk.mass
                     + MODEL.bulge.mass_enclosed(MODEL.bulge.r_trunc)
                     + MODEL.halo.mass_enclosed(MODEL.halo.r_trunc))
    com_speed = np.linalg.norm(P, axis=1) / m_total
    com_drift_kpc = float(np.trapezoid(com_speed, diag["t"]))
    mom = {"max_com_speed_km_s": float(com_speed.max()),
           "final_com_speed_km_s": float(com_speed[-1]),
           "com_displacement_kpc": com_drift_kpc,
           "fraction_of_internal_speed": float(com_speed.max() / 220.0)}

    print("  running symplectic-vs-RK2 comparison ...", flush=True)
    cmp = symplectic_comparison()
    ratio = cmp["rk2"]["max_abs_rel"] / max(cmp["leapfrog"]["max_abs_rel"], 1e-300)

    checks = [
        (f"max |dE| / |E_0| over the {meta['t_end_gyr']} Gyr collision run",
         f"{ee['max_rel_error']:.3e}", f"< {TOL_ENERGY:.0e}",
         ee["max_rel_error"] < TOL_ENERGY),
        ("energy error is bounded, not secular "
         "(linear trend / peak error)",
         f"{st['secular_fraction']:.3f}", f"< {TOL_SECULAR_FRACTION:.1f}",
         st["secular_fraction"] < TOL_SECULAR_FRACTION),
        ("max |dL| / |L_0| over the collision run",
         f"{le['max_rel_error']:.3e}", f"< {TOL_ANGMOM:.0e}",
         le["max_rel_error"] < TOL_ANGMOM),
        ("control: RK2 (same order, non-symplectic) energy drift "
         "exceeds leapfrog's",
         f"{ratio:.1f}x", "> 3x", ratio > 3.0),
        ("spurious centre-of-mass drift from tree force asymmetry",
         f"{mom['max_com_speed_km_s']:.3f} km/s", "< 1 km/s",
         mom["max_com_speed_km_s"] < 1.0),
    ]
    extra = {
        "run_meta": meta,
        "energy": ee, "energy_drift": st, "angular_momentum": le,
        "secular_drift_per_gyr": st["drift_per_gyr"],
        "symplectic_comparison": cmp,
        "linear_momentum": mom,
    }
    ok, _ = report("conservation", checks, extra)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
