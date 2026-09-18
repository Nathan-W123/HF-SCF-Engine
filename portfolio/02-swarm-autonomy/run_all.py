#!/usr/bin/env python3
"""One-command full reproduction.

    python3 run_all.py            # scenarios -> sweep -> validation -> figures -> hero
    python3 run_all.py --quick    # short scenarios, small sweep, still image only

Writes:
    results/<scenario>.pkl     pickled SwarmResult for each scenario
    results/sweep.json         robustness sweep over wind level x seed
    results/summary.json       machine-readable key numbers
    validation/*.json          validation outputs with pass/fail
    figures/*.png              supporting technical figures
    media/hero.png, hero.mp4   the hero visual
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
sys.path.insert(0, os.path.join(HERE, "validation"))

from common import jsonable as jsonable_early                 # noqa: E402
from swarmsim import __version__, metrics, scenarios, sim     # noqa: E402
from swarmsim.config import DrydenParams, WindParams          # noqa: E402

RES = os.path.join(HERE, "results")
VAL = os.path.join(HERE, "validation")

SCENARIOS = ("transit", "swap", "gust", "failure")
SWEEP_SIGMA = (0.0, 1.0, 2.0, 3.0)      # multiples of the nominal Dryden sigmas
SWEEP_SEEDS = (0, 1, 2, 3, 4)


# --------------------------------------------------------------------------
def _run_scenario(name, quick=False):
    t_max = 150.0 if quick else None
    n = 18 if quick else None
    fn = scenarios.ALL[name]
    kw = {}
    if t_max is not None:
        kw["t_max"] = t_max
    if n is not None:
        kw["n"] = n
    spec = fn(**kw)
    t0 = time.time()
    res = sim.run(spec)
    wall = time.time() - t0
    with open(os.path.join(RES, f"{name}.pkl"), "wb") as fh:
        pickle.dump(res, fh, protocol=4)
    m = metrics.scenario_metrics(res)
    m.update(metrics.envelope_check(res))
    m.update(metrics.zone_incursions(res))
    m["wall_time_s"] = wall
    m["wind"] = spec.wind.to_dict()
    return name, m


def _sweep_job(args):
    scale, seed, quick = args
    n = 18 if quick else 28
    # The transit corridor is ~7.8 km along the diagonal at 25 m/s, so the
    # horizon has to leave room for the crossing plus a turbulence allowance.
    t_max = 200.0 if quick else 400.0
    w = WindParams(mean=(-6.0 * min(scale, 1.5), 3.5 * min(scale, 1.5), 0.0),
                   dryden=DrydenParams().scaled(scale),
                   turbulence_on=scale > 0.0)
    spec = scenarios.transit(n=n, seed=seed, t_max=t_max, wind=w,
                             name=f"sweep_s{scale}_{seed}", log_every=4)
    res = sim.run(spec)
    m = metrics.scenario_metrics(res)
    m.update(metrics.envelope_check(res))
    m.update(metrics.zone_incursions(res))
    m["sigma_scale"] = scale
    m["seed"] = seed
    m["sigma_w_mps"] = DrydenParams().scaled(scale).sigma_w
    m["mean_wind_mps"] = float(np.linalg.norm(w.mean))
    return m


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-hero", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    os.makedirs(RES, exist_ok=True)
    os.makedirs(VAL, exist_ok=True)
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "media"), exist_ok=True)

    t_start = time.time()
    summary = {"package_version": __version__, "quick": args.quick}

    # ---------------- scenarios ----------------------------------------
    print("== scenarios ==")
    with Pool(min(args.workers, len(SCENARIOS))) as pool:
        out = pool.starmap(_run_scenario, [(n, args.quick) for n in SCENARIOS])
    scen = dict(out)
    for name, m in scen.items():
        print(f"  {name:8s} minsep {m['min_separation_m']:7.1f} m  "
              f"viol {m['separation_violation_samples']:4d}  "
              f"coll {m['collisions']}  goal {m['goal_completion']:.2f}  "
              f"eff {m['path_efficiency_mean']:.3f}  [{m['wall_time_s']:.0f} s]")
    summary["scenarios"] = scen

    # ---------------- robustness sweep ---------------------------------
    print("== robustness sweep ==")
    seeds = SWEEP_SEEDS[:2] if args.quick else SWEEP_SEEDS
    jobs = [(s, sd, args.quick) for s in SWEEP_SIGMA for sd in seeds]
    with Pool(args.workers) as pool:
        rows = pool.map(_sweep_job, jobs)
    sweep = {
        "n_runs": len(rows),
        "n_seeds": len(seeds),
        "sigma_scales": list(SWEEP_SIGMA),
        "sigma_w_mps": {str(s): DrydenParams().scaled(s).sigma_w for s in SWEEP_SIGMA},
        "R_min_m": rows[0]["R_min_m"],
        "runs": rows,
    }
    agg = {
        "min_separation_over_all_runs_m": float(min(r["min_separation_m"] for r in rows)),
        "total_collisions": int(sum(r["collisions"] for r in rows)),
        "total_separation_violation_samples":
            int(sum(r["separation_violation_samples"] for r in rows)),
        "worst_goal_completion": float(min(r["goal_completion"] for r in rows)),
        "mean_goal_completion": float(np.mean([r["goal_completion"] for r in rows])),
        "mean_path_efficiency": float(np.mean([r["path_efficiency_mean"] for r in rows])),
        "total_zone_incursion_samples":
            int(sum(r["zone_incursion_samples"] for r in rows)),
    }
    # per-turbulence-level breakdown, so the headline table needs no re-derivation
    by_level = {}
    for L in SWEEP_SIGMA:
        rs = [r for r in rows if r["sigma_scale"] == L]
        ms = sorted(r["min_separation_m"] for r in rs)
        by_level[str(L)] = {
            "sigma_w_mps": DrydenParams().scaled(L).sigma_w,
            "mean_wind_mps": rs[0]["mean_wind_mps"],
            "min_separation_worst_m": float(ms[0]),
            "min_separation_median_m": float(ms[len(ms) // 2]),
            "separation_violation_samples": int(
                sum(r["separation_violation_samples"] for r in rs)),
            "collisions": int(sum(r["collisions"] for r in rs)),
            "goal_completion_mean": float(
                np.mean([r["goal_completion"] for r in rs])),
            "path_efficiency_mean": float(
                np.mean([r["path_efficiency_mean"] for r in rs])),
            "zone_incursion_samples": int(
                sum(r["zone_incursion_samples"] for r in rs)),
        }
    sweep["by_level"] = by_level
    sweep["aggregate"] = agg
    with open(os.path.join(RES, "sweep.json"), "w") as fh:
        json.dump(jsonable_early(sweep), fh, indent=2)
    print(f"  {len(rows)} runs; worst min separation "
          f"{agg['min_separation_over_all_runs_m']:.1f} m; "
          f"collisions {agg['total_collisions']}; "
          f"mean completion {agg['mean_goal_completion']:.3f}")
    summary["sweep"] = agg

    # ---------------- validation ---------------------------------------
    print("== validation ==")
    import validate_dynamics
    import validate_tracking
    import validate_avoidance
    import validate_regression
    import validate_turbulence
    from common import jsonable, report

    with open(os.path.join(RES, "swap.pkl"), "rb") as fh:
        swap_result = pickle.load(fh)

    vres = {}
    for mod, name, kw in (
            (validate_dynamics, "validate_dynamics", {}),
            (validate_tracking, "validate_tracking", {}),
            (validate_avoidance, "validate_avoidance", {"result": swap_result}),
            (validate_regression, "validate_regression", {}),
            (validate_turbulence, "validate_turbulence", {})):
        payload = mod.run(**kw)
        report(name, payload)
        vres[payload["name"]] = {"all_passed": payload["all_passed"],
                                 "checks": payload["checks"]}
    summary["validation"] = vres
    summary["validation_all_passed"] = all(v["all_passed"] for v in vres.values())

    # ---------------- figures ------------------------------------------
    print("== figures ==")
    subprocess.run([sys.executable, os.path.join(HERE, "make_figures.py")],
                   check=True)

    # ---------------- hero ---------------------------------------------
    if not args.skip_hero:
        print("== hero ==")
        cmd = [sys.executable, os.path.join(HERE, "make_hero.py")]
        if args.quick:
            cmd.append("--still-only")
        subprocess.run(cmd, check=True)

    summary["total_wall_time_s"] = time.time() - t_start
    # numpy scalars (np.bool_ in particular) are not JSON-serialisable
    with open(os.path.join(RES, "summary.json"), "w") as fh:
        json.dump(jsonable(summary), fh, indent=2, sort_keys=True)
    print(f"\nsummary -> {os.path.join(RES, 'summary.json')}   "
          f"total {summary['total_wall_time_s'] / 60:.1f} min")
    if not summary["validation_all_passed"]:
        print("!! one or more validation checks FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
