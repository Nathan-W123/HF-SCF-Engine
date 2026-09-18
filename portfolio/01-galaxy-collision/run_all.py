#!/usr/bin/env python3
"""One-command full reproduction of the galaxy-collision project.

    python3 run_all.py              # everything, from scratch (~20 min, 4 cores)
    python3 run_all.py --quick      # same pipeline at reduced N (~3 min)
    python3 run_all.py --only simulate,hero
    python3 run_all.py --skip simulate

Stages, in order:

    kepler        two-body accuracy + measured convergence order
    tree          Barnes-Hut force error vs exact O(N^2), vs theta
    simulate      the production collision run (writes data/collision/)
    isolated      isolated-galaxy stability over 1.2 Gyr
    conservation  energy + angular momentum from the production diagnostics
    morphology    encounter/merger metrics measured from the snapshots
    figures       the six technical figures
    hero          media/hero.png and media/hero.mp4
    summary       results/summary.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "validation"))

RESULTS = ROOT / "results"
VALDIR = ROOT / "validation"
DATA = ROOT / "data"

STAGES = ["kepler", "tree", "simulate", "isolated", "conservation",
          "morphology", "figures", "hero", "summary"]


def banner(name):
    print("\n" + "=" * 74)
    print(f"  {name}")
    print("=" * 74, flush=True)


# --------------------------------------------------------------------------
def stage_kepler(quick):
    import val_kepler
    return val_kepler.main()


def stage_tree(quick):
    import val_tree_accuracy
    from galcol.ics import Resolution
    if quick:
        return val_tree_accuracy.main(
            res=Resolution(n_disk=2500, n_bulge=700, n_halo=2800),
            n_sinks=256)
    return val_tree_accuracy.main()


def stage_simulate(quick):
    from galcol.production import QUICK_SIM, SIM, build_initial_conditions
    from galcol.simulate import angmom_error, energy_error, run_simulation
    cfg = QUICK_SIM if quick else SIM
    p = build_initial_conditions(quick=quick)
    print(f"  N = {p.n:,} particles, {cfg.n_steps:,} steps of "
          f"{cfg.dt * 0.9777922216807892 * 1000:.3f} Myr "
          f"to t = {cfg.t_end_gyr} Gyr", flush=True)
    diag, meta = run_simulation(p, cfg, progress=True, store_snapshots=True)
    print(f"  wall clock {meta['wall_clock_s'] / 60:.2f} min, "
          f"{1e3 * meta['wall_per_step_s']:.0f} ms/step, "
          f"{meta['n_snapshots']} snapshots", flush=True)
    print(f"  max |dE|/|E0| = {energy_error(diag)['max_rel_error']:.3e}, "
          f"max |dL|/|L0| = {angmom_error(diag)['max_rel_error']:.3e}", flush=True)
    return 0


def stage_isolated(quick):
    import val_isolated
    from galcol.ics import Resolution
    if quick:
        return val_isolated.main(
            t_end_gyr=0.35,
            res=Resolution(n_disk=2500, n_bulge=700, n_halo=2800,
                           eps_disk=0.4, eps_bulge=0.45, eps_halo=1.0),
            dt=0.003, n_probe=20)
    return val_isolated.main(t_end_gyr=1.2, dt=0.0015)


def stage_conservation(quick):
    import val_conservation
    return val_conservation.main()


def stage_morphology(quick):
    """Measure what the encounter actually did, from the snapshots."""
    from galcol.analysis import mass_radius, robust_centre
    from galcol.units import TIME_UNIT_GYR

    run = DATA / "collision"
    d = np.load(run / "diagnostics.npz", allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    t = d["t"] * TIME_UNIT_GYR
    sep = d["sep"]

    # first pericentre = first local minimum of the separation
    i_peri = int(np.argmin(sep[: max(2, len(sep) // 3)]))
    # merger = the first time the separation drops below 2 kpc and stays there
    merged = sep < 2.0
    t_merge = None
    for i in range(len(merged)):
        if merged[i] and merged[i:].all():
            t_merge = float(t[i])
            break

    hdr = np.load(run / "header.npz", allow_pickle=True)
    ptype = hdr["ptype"][hdr["star_idx"]]
    mass = hdr["mass"][hdr["star_idx"]].astype(np.float64)
    files = sorted(run.glob("snap_*.npz"))
    times = np.array([float(np.load(f)["t_gyr"]) for f in files])

    tail = []
    for f in files[:: max(1, len(files) // 60)]:
        pos = np.load(f)["star_pos"].astype(np.float64)
        c = robust_centre(pos, ptype == 1)
        r = np.linalg.norm(pos - c, axis=1)
        tail.append(float(np.percentile(r, 99.5)))
    tail = np.asarray(tail)
    t_tail = times[:: max(1, len(files) // 60)][: tail.size]

    def shape(pos, m, c, r_max):
        d = pos - c
        r = np.linalg.norm(d, axis=1)
        s = r < r_max
        w = m[s]
        I = (w[:, None, None] * (d[s][:, :, None] * d[s][:, None, :])).sum(0) / w.sum()
        ev = np.sort(np.sqrt(np.maximum(np.linalg.eigvalsh(I), 0)))[::-1]
        return float(ev[2] / ev[0]), float(ev[1] / ev[0])

    pos0 = np.load(files[0])["star_pos"].astype(np.float64)
    m0 = (hdr["gid"][hdr["star_idx"]] == 0)
    c0 = robust_centre(pos0[m0], ptype[m0] == 1)
    rh0 = mass_radius(pos0[m0], mass[m0], centre=c0)
    ca0, ba0 = shape(pos0[m0], mass[m0], c0, rh0 * 2.0)

    posN = np.load(files[-1])["star_pos"].astype(np.float64)
    cN = robust_centre(posN, ptype == 1)
    rN = np.linalg.norm(posN - cN, axis=1)
    rhN = mass_radius(posN, mass, centre=cN)
    caN, baN = shape(posN, mass, cN, rhN * 2.0)
    debris = float(mass[rN > 30.0].sum() / mass.sum())

    out = {
        "t_first_pericentre_gyr": float(t[i_peri]),
        "r_first_pericentre_kpc": float(sep[i_peri]),
        "requested_kepler_pericentre_kpc": 12.0,
        "t_merge_gyr": t_merge,
        "max_tail_extent_kpc": float(tail.max()),
        "t_max_tail_gyr": float(t_tail[int(np.argmax(tail))]),
        "tail_extent_kpc_series": tail.tolist(),
        "tail_time_gyr_series": t_tail.tolist(),
        "initial_progenitor": {"stellar_half_mass_radius_kpc": rh0,
                               "axis_ratio_c_over_a": ca0,
                               "axis_ratio_b_over_a": ba0},
        "final_remnant": {"stellar_half_mass_radius_kpc": rhN,
                          "axis_ratio_c_over_a": caN,
                          "axis_ratio_b_over_a": baN,
                          "stellar_mass_fraction_beyond_30kpc": debris},
        "separation_gyr": t.tolist(),
        "separation_kpc": sep.tolist(),
        "run_meta": meta,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "morphology.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"  first pericentre: {out['r_first_pericentre_kpc']:.1f} kpc at "
          f"{out['t_first_pericentre_gyr']:.3f} Gyr")
    print(f"  merger (sep < 2 kpc for good): "
          f"{'%.3f Gyr' % t_merge if t_merge else 'not reached'}")
    print(f"  longest tidal tail: {out['max_tail_extent_kpc']:.1f} kpc at "
          f"{out['t_max_tail_gyr']:.2f} Gyr")
    print(f"  progenitor disk c/a = {ca0:.3f}  ->  remnant c/a = {caN:.3f}")
    print(f"  stellar mass beyond 30 kpc at the end: {100 * debris:.1f}%")
    return 0


def stage_figures(quick):
    return subprocess.call([sys.executable,
                            str(ROOT / "figures" / "make_figures.py")])


def stage_hero(quick):
    cmd = [sys.executable, str(ROOT / "render_hero.py")]
    if quick:
        cmd += ["--frames", "90"]
    return subprocess.call(cmd)


def stage_summary(quick):
    from galcol.production import ENCOUNTER, MODEL, RESOLUTION, SIM
    RESULTS.mkdir(parents=True, exist_ok=True)
    val = {}
    for p in sorted(VALDIR.glob("*.json")):
        val[p.stem] = json.loads(p.read_text())
    morph = {}
    mp = RESULTS / "morphology.json"
    if mp.exists():
        morph = json.loads(mp.read_text())

    run_meta = morph.get("run_meta", {})
    cons = val.get("conservation", {})
    tree = val.get("tree_force_accuracy", {})
    kep = val.get("kepler_two_body", {})
    iso = val.get("isolated_galaxy_stability", {})
    prod_row = None
    if tree:
        prod_row = min(tree["rows"],
                       key=lambda r: abs(r["theta"] - tree["production_theta"]))

    summary = {
        "project": "Gravitational N-body galaxy collision",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "quick_mode": bool(quick),
        "model": MODEL.as_dict(),
        "resolution": {"n_disk_per_galaxy": RESOLUTION.n_disk,
                       "n_bulge_per_galaxy": RESOLUTION.n_bulge,
                       "n_halo_per_galaxy": RESOLUTION.n_halo,
                       "n_total": run_meta.get("n_particles"),
                       "softening_kpc": run_meta.get("softening_kpc")},
        "encounter": {"r_pericentre_kpc": ENCOUNTER.r_peri,
                      "eccentricity": ENCOUNTER.ecc,
                      "initial_separation_kpc": ENCOUNTER.r_start,
                      "inclination_1_deg": ENCOUNTER.inc1,
                      "inclination_2_deg": ENCOUNTER.inc2},
        "integration": {"scheme": "kick-drift-kick leapfrog (symplectic)",
                        "dt_myr": run_meta.get("dt_myr", SIM.dt * 977.79),
                        "t_end_gyr": run_meta.get("t_end_gyr", SIM.t_end_gyr),
                        "n_steps": run_meta.get("n_steps"),
                        "theta": run_meta.get("theta"),
                        "dt_criterion_median_myr": run_meta.get("dt_criterion_median_myr"),
                        "dt_criterion_p01_myr": run_meta.get("dt_criterion_p01_myr")},
        "performance": {"wall_clock_min": (run_meta.get("wall_clock_s", 0) / 60.0),
                        "ms_per_step": 1e3 * run_meta.get("wall_per_step_s", 0),
                        "us_per_step_per_particle":
                            run_meta.get("wall_per_step_per_particle_us"),
                        "barnes_hut_speedup_vs_direct":
                            (prod_row or {}).get("speedup_vs_direct")},
        "validation": {
            "kepler": {
                "passed": kep.get("passed"),
                "measured_convergence_order": kep.get("convergence_slope"),
                "max_position_error_over_a_at_production_dt":
                    (kep.get("production_dt_row") or {}).get("max_err"),
                "max_position_error_over_a_finest_dt":
                    (kep.get("ladder") or [{}])[-1].get("max_err"),
            },
            "isolated_galaxy": {
                "passed": iso.get("passed"),
                "t_end_gyr": iso.get("t_end_gyr"),
                "final_change": iso.get("final_change"),
            },
            "energy": {
                "passed": cons.get("passed"),
                "max_rel_error": (cons.get("energy") or {}).get("max_rel_error"),
                "secular_drift_per_gyr": cons.get("secular_drift_per_gyr"),
                "secular_fraction_of_peak":
                    (cons.get("energy_drift") or {}).get("secular_fraction"),
                "rk2_control_drift_ratio":
                    ((cons.get("symplectic_comparison") or {}).get("rk2") or {}).get("max_abs_rel"),
            },
            "angular_momentum": {
                "max_rel_error":
                    (cons.get("angular_momentum") or {}).get("max_rel_error"),
            },
            "barnes_hut_force_accuracy": {
                "passed": tree.get("passed"),
                "production_theta": tree.get("production_theta"),
                "median_rel_error": (prod_row or {}).get("median_rel_err"),
                "p99_rel_error": (prod_row or {}).get("p99_rel_err"),
            },
        },
        "results": {k: v for k, v in morph.items()
                    if k not in ("separation_gyr", "separation_kpc",
                                 "tail_extent_kpc_series",
                                 "tail_time_gyr_series", "run_meta")},
        "all_validations_passed": all(
            v.get("passed", True) for v in val.values()),
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2,
                                                     default=float) + "\n")
    print(json.dumps(summary["validation"], indent=2, default=float))
    print(f"\n  wrote {RESULTS / 'summary.json'}")
    print(f"  all validations passed: {summary['all_validations_passed']}")
    return 0 if summary["all_validations_passed"] else 1


HANDLERS = {n: globals()[f"stage_{n}"] for n in STAGES}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="reduced N and shorter integration")
    ap.add_argument("--only", default=None, help="comma-separated stage list")
    ap.add_argument("--skip", default="", help="comma-separated stage list")
    args = ap.parse_args()

    stages = args.only.split(",") if args.only else list(STAGES)
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    stages = [s.strip() for s in stages if s.strip() and s.strip() not in skip]
    for s in stages:
        if s not in HANDLERS:
            raise SystemExit(f"unknown stage {s!r}; known: {', '.join(STAGES)}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    failures = []
    timings = {}
    for s in stages:
        banner(f"stage: {s}")
        t0 = time.perf_counter()
        rc = HANDLERS[s](args.quick)
        timings[s] = time.perf_counter() - t0
        print(f"  [{s}] {timings[s]:.1f} s, exit {rc}", flush=True)
        if rc:
            failures.append(s)
    total = time.perf_counter() - t_start

    banner("pipeline complete")
    for s, dt in timings.items():
        print(f"  {s:<14s} {dt / 60:6.2f} min")
    print(f"  {'TOTAL':<14s} {total / 60:6.2f} min")
    if failures:
        print(f"\n  stages reporting failure: {', '.join(failures)}")
        return 1
    print("\n  all stages OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
