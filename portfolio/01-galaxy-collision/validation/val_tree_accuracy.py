"""Validation 5 -- Barnes-Hut force accuracy versus exact O(N^2) summation.

A tree code is only as good as its opening angle.  This check takes the actual
production particle configuration, computes the *exact* softened acceleration
on a random subsample of sinks by direct summation over all N sources, and
compares it with the Barnes-Hut result as a function of theta.  It also times
both, which is where the speed-up claim in the README comes from.
"""
from __future__ import annotations

import time

import numpy as np

from _common import report
from galcol.ics import Encounter, Resolution, make_collision
from galcol.production import ENCOUNTER, RESOLUTION, SIM
from galcol.profiles import default_galaxy
from galcol.tree import accel_direct_subset, accel_tree, build_tree

N_SINKS = 1024
THETAS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
PRODUCTION_THETA = SIM.theta

# Two error metrics are reported.
#
#  * |da| / |a_exact| -- the intuitive per-particle relative error.  Its upper
#    percentiles are dominated by the handful of particles that sit where the
#    forces of the two galaxies nearly cancel, so |a_exact| -> 0 and the ratio
#    diverges for reasons that have nothing to do with the tree.
#  * |da| / a_rms, with a_rms the root-mean-square acceleration of the sink
#    sample -- the metric used in the tree-code literature (Barnes & Hut 1986;
#    Springel 2005) precisely because it is not sensitive to that cancellation.
#
# The pass/fail criteria use |da|/|a|, the metric normally quoted for a tree
# code, with thresholds set at the standard "sub-percent median" bar.  The
# a_rms numbers are recorded alongside so the shape of the tail is visible:
# because absolute errors are largest where the acceleration is largest (the
# galaxy centres) while relative errors are largest where it is smallest (the
# saddle between the two galaxies), neither normalisation is clean on its own.


def main(res=None, enc=None, n_sinks=N_SINKS):
    model = default_galaxy()
    res = res or RESOLUTION
    enc = enc or ENCOUNTER
    p = make_collision(model, res, enc)
    pos, mass, eps2 = p.pos, p.mass, p.eps2
    n = p.n

    rng = np.random.default_rng(7)
    idx = np.sort(rng.choice(n, size=min(n_sinks, n), replace=False)).astype(np.int64)

    accel_direct_subset(pos[:64], mass[:64], eps2[:64], np.arange(4, dtype=np.int64))
    t0 = time.perf_counter()
    exact = accel_direct_subset(pos, mass, eps2, idx)
    t_direct_subset = time.perf_counter() - t0
    # cost of a FULL direct step, extrapolated from the measured subset cost
    t_direct_full = t_direct_subset * n / idx.size
    norm = np.linalg.norm(exact, axis=1)
    a_rms = float(np.sqrt(np.mean((exact * exact).sum(axis=1))))

    rows = []
    accel_tree(pos[:512], mass[:512], eps2[:512], theta=0.7)   # warm the JIT
    for th in THETAS:
        tree = build_tree(pos, mass, eps2, leaf_size=12)
        acc, _ = accel_tree(pos, mass, eps2, theta=th, tree=tree)
        t0 = time.perf_counter()
        reps = 3
        for _ in range(reps):
            acc, _ = accel_tree(pos, mass, eps2, theta=th, tree=tree)
        t_tree = (time.perf_counter() - t0) / reps
        dabs = np.linalg.norm(acc[idx] - exact, axis=1)
        err = dabs / norm
        errn = dabs / a_rms
        rows.append({
            "theta": float(th),
            "median_rel_err": float(np.median(err)),
            "p90_rel_err": float(np.percentile(err, 90)),
            "p99_rel_err": float(np.percentile(err, 99)),
            "max_rel_err": float(err.max()),
            "median_err_over_arms": float(np.median(errn)),
            "p99_err_over_arms": float(np.percentile(errn, 99)),
            "max_err_over_arms": float(errn.max()),
            "tree_walk_s": float(t_tree),
            "speedup_vs_direct": float(t_direct_full / t_tree),
        })
        print(f"  theta={th:.2f}  med|da|/|a|={rows[-1]['median_rel_err']:.3e}  "
              f"p99|da|/|a|={rows[-1]['p99_rel_err']:.3e}  "
              f"p99|da|/a_rms={rows[-1]['p99_err_over_arms']:.3e}  "
              f"walk={t_tree:.3f}s  speedup={rows[-1]['speedup_vs_direct']:.1f}x")

    prod = min(rows, key=lambda r: abs(r["theta"] - PRODUCTION_THETA))
    monotone = all(rows[i]["median_rel_err"] <= rows[i + 1]["median_rel_err"] * 1.05
                   for i in range(len(rows) - 1))

    checks = [
        (f"median relative force error |da|/|a| at theta={PRODUCTION_THETA}",
         f"{prod['median_rel_err']:.3e}", "< 1.0e-2", prod["median_rel_err"] < 1.0e-2),
        (f"99th-percentile relative force error at theta={PRODUCTION_THETA}",
         f"{prod['p99_rel_err']:.3e}", "< 5.0e-2", prod["p99_rel_err"] < 5.0e-2),
        ("median error decreases monotonically as theta decreases",
         f"{monotone}", "True", monotone),
        (f"Barnes-Hut speed-up over direct O(N^2) at theta={PRODUCTION_THETA}",
         f"{prod['speedup_vs_direct']:.1f}x", "> 20x", prod["speedup_vs_direct"] > 20.0),
    ]
    extra = {
        "n_particles": int(n),
        "n_sinks": int(idx.size),
        "a_rms_sink_sample": a_rms,
        "leaf_size": 12,
        "direct_full_step_s_extrapolated": float(t_direct_full),
        "production_theta": PRODUCTION_THETA,
        "rows": rows,
    }
    ok, _ = report("tree_force_accuracy", checks, extra)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
