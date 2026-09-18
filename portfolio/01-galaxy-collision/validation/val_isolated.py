"""Validation 2 -- isolated-galaxy stability.

The collision result is only meaningful if each galaxy is in equilibrium
*before* the encounter: a disk that puffs up or spreads on its own would
manufacture tidal-looking structure out of nothing.  One galaxy is therefore
evolved alone -- same integrator, softening, opening angle, timestep and live
halo as the production run -- and its structure is tracked in time.

Three structural measures:

  * cylindrical half-mass radius of the disk       (radial spreading)
  * exponential scale length fitted over 3-10.5 kpc (profile shape)
  * median |z| of disk particles at R = 2-10 kpc    (vertical heating)

Vertical thickening is the most sensitive -- it is the classic symptom of
two-body relaxation against the more massive halo particles -- so it carries
the loosest tolerance.  All three tolerances are declared below, before the
run, and the measured numbers are reported whatever they turn out to be.
"""
from __future__ import annotations

import numpy as np

from _common import DATA, report
from galcol.analysis import (cyl_mass_radius, fit_scale_length, robust_centre,
                             vertical_thickness)
from galcol.ics import Resolution, make_isolated
from galcol.profiles import default_galaxy
from galcol.simulate import (SimConfig, angmom_error, energy_error,
                             run_simulation)
from galcol.units import PTYPE_DISK, TIME_UNIT_GYR

TOL_HALF_MASS = 0.10      # max fractional change in disk half-mass radius
TOL_SCALE_LEN = 0.15      # max fractional change in fitted scale length
TOL_THICKNESS = 0.60      # max fractional growth in median |z|
TOL_ENERGY = 5.0e-3       # max |dE|/|E0|


def main(t_end_gyr=1.2, res=None, theta=0.8, dt=0.001, n_probe=40):
    model = default_galaxy()
    res = res or Resolution()
    p = make_isolated(model, res, live_halo=True)
    disk = np.flatnonzero(p.ptype == PTYPE_DISK)
    m_disk = p.mass[disk]

    track = {k: [] for k in ("t_gyr", "r_half", "r_scale", "z_med", "z_rms")}

    def probe(step, t, pos, vel):
        dpos = pos[disk]
        c = robust_centre(dpos)
        track["t_gyr"].append(float(t * TIME_UNIT_GYR))
        track["r_half"].append(cyl_mass_radius(dpos, m_disk, centre=c))
        track["r_scale"].append(fit_scale_length(dpos, m_disk, centre=c))
        zm, zr = vertical_thickness(dpos, m_disk, centre=c)
        track["z_med"].append(zm)
        track["z_rms"].append(zr)

    cfg = SimConfig(dt=dt, t_end_gyr=t_end_gyr, theta=theta, leaf_size=12,
                    diag_every=10, out_dir=str(DATA / "isolated"),
                    snap_dt_gyr=0.1, label="isolated")
    diag, meta = run_simulation(p, cfg, progress=True, store_snapshots=True,
                                probe=probe,
                                probe_every=max(1, cfg.n_steps // n_probe))

    trk = {k: np.asarray(v) for k, v in track.items()}
    d_half = float(trk["r_half"][-1] / trk["r_half"][0] - 1.0)
    d_scale = float(trk["r_scale"][-1] / trk["r_scale"][0] - 1.0)
    d_thick = float(trk["z_med"][-1] / trk["z_med"][0] - 1.0)
    # worst excursion at any probed time, not just the endpoint
    w_half = float(np.max(np.abs(trk["r_half"] / trk["r_half"][0] - 1.0)))
    w_scale = float(np.max(np.abs(trk["r_scale"] / trk["r_scale"][0] - 1.0)))
    w_thick = float(np.max(trk["z_med"] / trk["z_med"][0] - 1.0))

    ee = energy_error(diag)
    le = angmom_error(diag)

    checks = [
        (f"disk half-mass radius change over {t_end_gyr} Gyr "
         f"({trk['r_half'][0]:.2f} -> {trk['r_half'][-1]:.2f} kpc)",
         f"{100 * d_half:+.2f}%", f"|.| < {100 * TOL_HALF_MASS:.0f}%",
         abs(d_half) < TOL_HALF_MASS),
        (f"fitted scale length change "
         f"({trk['r_scale'][0]:.2f} -> {trk['r_scale'][-1]:.2f} kpc)",
         f"{100 * d_scale:+.2f}%", f"|.| < {100 * TOL_SCALE_LEN:.0f}%",
         abs(d_scale) < TOL_SCALE_LEN),
        (f"disk median |z| growth "
         f"({trk['z_med'][0]:.3f} -> {trk['z_med'][-1]:.3f} kpc)",
         f"{100 * d_thick:+.2f}%", f"< {100 * TOL_THICKNESS:.0f}%",
         d_thick < TOL_THICKNESS),
        ("max |dE| / |E0| over the isolated run",
         f"{ee['max_rel_error']:.3e}", f"< {TOL_ENERGY:.0e}",
         ee["max_rel_error"] < TOL_ENERGY),
    ]
    extra = {
        "t_end_gyr": t_end_gyr,
        "n_particles": meta["n_particles"],
        "resolution": {"n_disk": res.n_disk, "n_bulge": res.n_bulge,
                       "n_halo": res.n_halo},
        "softening_kpc": meta["softening_kpc"],
        "final_change": {"r_half": d_half, "r_scale": d_scale, "z_med": d_thick},
        "worst_change": {"r_half": w_half, "r_scale": w_scale, "z_med": w_thick},
        "track": {k: v.tolist() for k, v in trk.items()},
        "energy": ee, "angular_momentum": le,
        "meta": meta,
    }
    ok, _ = report("isolated_galaxy_stability", checks, extra)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
