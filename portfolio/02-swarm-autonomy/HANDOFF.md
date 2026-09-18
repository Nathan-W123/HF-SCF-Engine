# HANDOFF — 02-swarm-autonomy

## Status

**Complete.** `make all` runs end to end from a clean tree and exits 0;
`pytest` passes; all five validation scripts execute with numeric thresholds and
write JSON; six supporting figures and both hero assets are produced entirely
from real simulation output.

| item | state |
|---|---|
| `make all` from scratch | clean, **18.5 min** wall on 4 cores (measured with another job competing for the same cores) |
| `pytest` | **97 test cases** (78 test functions, 8 files), **22 s** |
| validation | **33 checks, all pass**, outputs in `validation/*.json` |
| figures | 6 PNGs in `figures/` |
| hero | `media/hero.png` 1920×1080, `media/hero.mp4` 1920×1080 / 30 fps / 22.0 s / 9.3 MB |
| machine-readable results | `results/summary.json`, `results/sweep.json` |

## Headline result

**36 fixed-wing aircraft, every nominal route through one common point, 69.0 m
of separation held against a 60 m requirement, zero collisions, all 36 goals
reached — at a cost of 0.6 % in path length.**

The high-density antipodal exchange is the stress test the safety invariant is
asserted on, and it is also the run the hero image and video are rendered from.

## Validation status

All 33 checks pass. The numbers that matter:

| check | threshold | measured |
|---|---|---|
| steady turn radius vs `V²/(g tan φ)`, 25 cases | rel. err < 1e-6 | **4.5e-13** |
| RK4 convergence order | 3.7 – 4.3 | **4.035** (error ratios 16.6 / 16.3 / 16.3 per halving) |
| envelope + rate limits under saturating commands | inside limits | V ∈ [18.00, 34.00] m/s, \|φ\| ≤ 45.00°, \|γ\| ≤ 12.00°, n ≤ 1.414, rates at 60.00 °/s / 8.00 °/s / 2.000 m/s² |
| L1 capture, 12 releases (straight + circular) | all settle | worst settling **1010 m** straight, **1075 m** circular |
| steady-state cross-track error | < 0.01 / 0.05 m | **5.3e-9 m** straight, **1.9e-8 m** circular |
| stress-test min pairwise separation | ≥ 60 m | **69.0 m** (margin +9.0 m, at t = 85.2 s, pair 24–26) |
| stress-test collisions (< 15 m) | 0 | **0** |
| stress-test goal completion | 36/36 | **36/36** by 172.0 s |
| ORCA reciprocity `u_A = −u_B`, 1500 pairs | < 1e-9 m/s | **0.0** (exact) |
| reciprocal split resolves conflicts | 55/55 | **55/55**, landing on the obstacle boundary to **1 − 4e-16** |
| Hildreth QP vs scipy SLSQP, 115 problems | < 1e-6 | **2.0e-11** |
| fixed-seed regression vs stored signature | ≤ 1e-9 | **0.0**, reruns bit-identical |
| Dryden realised RMS (u/v/w) | within 3 % | **1.81 % / 0.37 % / 0.13 %** |
| Dryden median PSD ratio, 0.03–30 rad/s | within 8 % | **1.011 / 1.010 / 1.007** |
| Dryden log₁₀ PSD-ratio RMS, 24 bins | < 0.05 dec | **0.040 / 0.038 / 0.020** |

## Scenario results

Zero separation violations, zero collisions and zero no-fly-zone incursions in
all four scenarios; envelope respected throughout (peak load factor 1.414 vs a
1.6 limit).

| scenario | vehicles | min separation | goals | path efficiency | hard-infeasible fallbacks |
|---|---|---|---|---|---|
| `transit` | 32 | 68.4 m | 32/32 by 314.6 s | 1.011 | 0 |
| `swap` | 36 | **69.0 m** | 36/36 by 172.0 s | **1.006** | 138 |
| `gust` | 32 | 65.7 m | 32/32 by 328.7 s | 1.020 | 4 |
| `failure` | 36 | 64.9 m | 35/36 (the failed vehicle is the exception) | 1.007 | 98 |

Robustness sweep, 4 turbulence levels × 5 seeds (20 runs): zero collisions, zero
zone incursions, 100 % goal completion, mean path efficiency 1.016. Worst min
separation by level: 60.9 / 58.8 / 63.5 / **44.6** m at σ_w = 0 / 1 / 2 / 3 m/s.

## Limitations (short form — full list in README)

1. **Separation degrades at the top of the turbulence sweep**: 44.6 m worst case
   at σ_w = 3 m/s, against the 60 m requirement. Still zero collisions and 100 %
   completion, but the invariant is asserted at nominal wind, not there.
2. **The CBF guarantee is for a single integrator**, not for an airframe with
   roll and airspeed lags — which is why separation is measured everywhere here
   rather than claimed as a proof.
3. **No sensing model.** Neighbour states are exact: no noise, latency, dropout
   or track association. Latency is the biggest gap to a flyable system — even
   200 ms of staleness would consume much of the 9 m margin.
4. **Turbulence is spatially uncorrelated between vehicles** (conservative for
   separation, but not physical), and gusts enter kinematically only.
5. **One failure mode** (locked bank + non-cooperative); no sensor failures,
   partial degradation or simultaneous failures.
6. **Planning is horizontal and obstacles are discs**; no polygons or terrain.
7. **Vertical separation does a lot of the work** — minimum *horizontal*
   separation reaches 0.5 m in the stress test while 3-D separation stays above
   69 m, i.e. vehicles pass over and under each other.
8. **Safety is measured on 10 Hz samples**; the sampling-gap bound (5.4 m) is
   reported, giving a guaranteed continuous-time margin of ≥ 63.6 m.
9. **The QP's hard rows were jointly infeasible on 0.13 % of vehicle-steps** in
   the stress test, each time falling back to a max-margin heuristic with no
   guarantee attached. Counted and reported, not hidden.
10. **Not validated against flight data** — every check is internal consistency
    (analytic solutions, convergence orders, spectra, invariants, cross-solver
    agreement).

## Where things are

- `run_all.py` — one-command reproduction; `make all` calls it.
- `src/swarmsim/` — the implementation; `avoidance.py`/`qp.py` are the readable
  reference and `_kernels.py` is the numba equivalent, held to 1e-9 agreement by
  `tests/test_kernel_equivalence.py`.
- `validation/*.json` — every validation number, including per-case detail.
- `results/summary.json` — machine-readable headline numbers.
- `make_hero.py` — the renderer; `--cam key=value,...` overrides the camera for
  reframing without touching the code.
