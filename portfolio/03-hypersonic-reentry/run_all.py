#!/usr/bin/env python3
"""One-command full reproduction.

    python3 run_all.py            # trajectories -> validation -> figures -> hero
    python3 run_all.py --quick    # same pipeline, short video (smoke test)
    python3 run_all.py --skip-hero

Stages
------
1. ``scripts/run_trajectories.py``   nominal ballistic + lifting entries and the
   two entry-corridor sweeps  ->  ``results/``
2. ``validation/run_all_validation.py``  the four numbered validations, each with
   numeric pass/fail thresholds  ->  ``validation/``  (a failure aborts the run)
3. ``scripts/make_figures.py``       the six technical figures  ->  ``figures/``
4. ``scripts/make_hero.py``          ``media/hero.png`` and ``media/hero.mp4``
5. this script collects every number into ``results/summary.json``.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent
SRC = ROOT / "src"
RESULTS = ROOT / "results"
VALIDATION = ROOT / "validation"
FIGURES = ROOT / "figures"
MEDIA = ROOT / "media"

sys.path.insert(0, str(SRC))


def _run(script: pathlib.Path, args=(), label=""):
    print(f"\n=== {label or script.name} " + "=" * max(4, 66 - len(label)))
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, str(script.name), *args], cwd=str(script.parent)
    )
    dt = time.time() - t0
    if proc.returncode != 0:
        raise SystemExit(f"FAILED: {script} (exit {proc.returncode})")
    print(f"--- {script.name} finished in {dt:.1f} s")
    return dt


def _library_versions():
    import importlib

    out = {"python": sys.version.split()[0]}
    for mod in ("numpy", "scipy", "matplotlib", "numba", "imageio", "imageio_ffmpeg",
                "PIL", "tqdm"):
        try:
            out[mod] = importlib.import_module(mod).__version__
        except Exception:  # pragma: no cover
            out[mod] = "unavailable"
    return out


def write_summary(timings: dict, quick: bool, hero: bool) -> dict:
    import reentry
    from reentry.aerodynamics import CP_MAX_HYPERSONIC_LIMIT
    from reentry.constants import (
        K_SUTTON_GRAVES, MU_EARTH, OMEGA_EARTH, R_EARTH, SIGMA_SB,
    )

    cases = json.loads((RESULTS / "cases.json").read_text())
    corridor = json.loads((RESULTS / "corridor.json").read_text())
    report = json.loads((VALIDATION / "validation_report.json").read_text())

    val = {}
    for rep in report["reports"]:
        val[rep["name"]] = {
            "passed": rep["passed"],
            "checks": [
                {k: c[k] for k in ("name", "value", "threshold", "comparison",
                                   "units", "passed")}
                for c in rep["checks"]
            ],
            "data": rep["data"],
        }

    ae = report["reports"][1]["data"]
    v1 = report["reports"][0]["data"]
    v4 = report["reports"][3]["data"]

    summary = {
        "project": "03-hypersonic-reentry",
        "title": "Hypersonic atmospheric reentry simulator",
        "version": reentry.__version__,
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(
            timespec="seconds"),
        "environment": _library_versions(),
        "reproduction": {
            "command": "make all",
            "quick_mode": quick,
            "hero_rendered": hero,
            "stage_seconds": timings,
            "total_seconds": round(sum(timings.values()), 1),
        },
        "model": {
            "gravity": "central inverse-square, mu = 3.986004418e14 m^3/s^2, "
                       "spherical Earth R = 6371.0 km (WGS-84 mean radius R1)",
            "earth_rotation_included_in_reported_results": False,
            "earth_rotation_available": True,
            "omega_earth_rad_s": OMEGA_EARTH,
            "mu_m3_s2": MU_EARTH,
            "r_earth_m": R_EARTH,
            "atmosphere": "USSA76 from the geopotential layer table to 86 km; "
                          "USSA76 kinetic-temperature profile with hydrostatic "
                          "integration at constant mean molecular weight above",
            "aerodynamics": "modified Newtonian surface integration over a 70 deg "
                            "sphere-cone, Cp_max from the Rayleigh pitot relation "
                            "(supersonic) / isentropic stagnation (subsonic)",
            "cp_max_hypersonic_limit": CP_MAX_HYPERSONIC_LIMIT,
            "heating_primary": "Sutton-Graves, k = %.6g SI" % K_SUTTON_GRAVES,
            "heating_cross_check": "Detra-Kemp-Riddell form (V^3.15) -- leading "
                                   "coefficient unverified offline, see README",
            "wall_temperature": "radiative equilibrium, eps*sigma*T^4 = qdot, "
                                "eps = 0.85, sigma = %.9g" % SIGMA_SB,
            "integrator": "scipy DOP853, rtol = atol = 1e-10, terminal-altitude "
                          "and skip-out events; fixed-step RK4 cross-check",
            "degrees_of_freedom": 3,
        },
        "cases": cases,
        "entry_corridor": {
            "leo": {k: corridor["leo"][k] for k in
                    ("entry_velocity_m_s", "circular_speed_at_entry_m_s",
                     "skip_out_gammas_deg", "shallowest_gamma_exceeding_10g_deg")},
            "super_circular": {k: corridor["super_circular"][k] for k in
                               ("entry_velocity_m_s", "skip_out_boundary_gamma_deg",
                                "shallowest_gamma_exceeding_10g_deg")},
            "n_cases": len(corridor["leo"]["rows"]) + len(
                corridor["super_circular"]["rows"]),
        },
        "validation": {
            "all_passed": report["all_passed"],
            "n_checks": report["n_checks"],
            "n_failed": report["n_failed"],
            "reports": val,
        },
        "headline": {
            "peak_stagnation_heat_flux_W_cm2":
                cases["ballistic"]["peak_heat_flux_W_cm2"],
            "peak_heat_flux_altitude_km":
                cases["ballistic"]["peak_heat_flux_altitude_km"],
            "peak_wall_temperature_K": cases["ballistic"]["peak_wall_temperature_K"],
            "peak_deceleration_g": cases["ballistic"]["peak_g_load"],
            "total_heat_load_J_cm2": cases["ballistic"]["total_heat_load_J_cm2"],
            "lifting_peak_deceleration_g": cases["lifting"]["peak_g_load"],
            "lifting_downrange_km": cases["lifting"]["downrange_km"],
            "allen_eggers_exact_limit_rel_error":
                ae["exact_limit"]["a_max_rel_error"],
            "allen_eggers_profile_agreement":
                ae["exact_limit"]["max_profile_deviation_over_amax"],
            "orbit_energy_drift": v1["circular_orbit"]["energy_drift_rel"],
            "kepler_period_closure_m": v1["circular_orbit"]["period_closure_error_m"],
            "energy_budget_closure_ballistic":
                v4["energy_budget"]["ballistic"]["relative_closure"],
            "rk4_observed_order": v4["rk4_convergence"]["observed_orders"][-1],
        },
        "artifacts": {},
    }

    for d in (RESULTS, VALIDATION, FIGURES, MEDIA):
        for f in sorted(d.glob("*")):
            if f.is_file() and f.suffix in {".png", ".mp4", ".json", ".npz", ".txt"}:
                summary["artifacts"][str(f.relative_to(ROOT))] = f.stat().st_size

    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true",
                    help="short hero video (smoke test of the whole pipeline)")
    ap.add_argument("--skip-hero", action="store_true")
    args = ap.parse_args()

    t_start = time.time()
    timings = {}
    timings["trajectories"] = _run(ROOT / "scripts" / "run_trajectories.py",
                                   label="1/4 trajectories")
    timings["validation"] = _run(ROOT / "validation" / "run_all_validation.py",
                                 label="2/4 validation")
    timings["figures"] = _run(ROOT / "scripts" / "make_figures.py",
                              label="3/4 figures")
    if not args.skip_hero:
        timings["hero"] = _run(ROOT / "scripts" / "make_hero.py",
                               ["--quick"] if args.quick else [],
                               label="4/4 hero render")

    summary = write_summary(timings, args.quick, not args.skip_hero)
    total = time.time() - t_start
    print("\n" + "=" * 72)
    print(f"results/summary.json written   ({total / 60:.1f} min total)")
    h = summary["headline"]
    print(f"  peak heat flux      {h['peak_stagnation_heat_flux_W_cm2']:8.1f} W/cm^2"
          f"  at {h['peak_heat_flux_altitude_km']:.1f} km")
    print(f"  peak wall temp      {h['peak_wall_temperature_K']:8.0f} K")
    print(f"  peak deceleration   {h['peak_deceleration_g']:8.2f} g   "
          f"(lifting: {h['lifting_peak_deceleration_g']:.2f} g)")
    print(f"  total heat load     {h['total_heat_load_J_cm2']:8.0f} J/cm^2")
    v = summary["validation"]
    print(f"  validation          {v['n_checks'] - v['n_failed']}/{v['n_checks']} "
          f"checks passed ({'ALL PASS' if v['all_passed'] else 'FAILURES'})")
    return 0 if summary["validation"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
