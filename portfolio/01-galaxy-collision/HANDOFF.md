# HANDOFF — 01-galaxy-collision

**Status: complete.** `make all` runs the whole project from an empty checkout
(no network) and exits 0 — verified from a `make distclean` state. It took
25.7 min on the recorded run, which shared its four cores with two other
pipelines; the same stages on an idle machine take ≈ 13 min (simulation stage
7.16 min vs 14.47 min). `make test` runs 58 pytest tests in ~6 s. Every number
below is read back from an artifact the pipeline wrote (`results/summary.json`,
`results/morphology.json`, `validation/*.json`).

The pipeline is fully deterministic: two independent full runs produced
byte-identical `media/hero.mp4` and identical validation numbers to every
digit.

## Headline result

Two Milky-Way-like spirals (72,000 particles, live dark-matter halos) on a
prograde encounter with a 16.7 kpc first pericentre develop tidal tails and a
bridge, merge at **t = 1.03 Gyr**, and leave a remnant whose stellar shape has
gone from a thin disk (**c/a = 0.216**) to a triaxial spheroid
(**c/a = 0.516, b/a = 0.745**) — disk destruction by a major merger, from
Newtonian gravity alone. The longest tidal tail reaches **130 kpc**; 8.4 % of
the stellar mass ends up beyond 30 kpc.

Simulation cost on 4 idle cores: **7.16 min wall clock**, 274 ms/step,
**3.80 µs per step per particle**; Barnes–Hut is **19.7–22.5× faster** than an
equally optimised direct O(N²) sum at the same N (one full direct force
evaluation at N = 72,000 takes 10.9 s against 0.55 s for the tree).

## Validation status — all five checks PASS

| Check | Result | Threshold |
|---|---|---|
| Kepler two-body, error at production dt | 2.77 × 10⁻³ of semi-major axis | < 10⁻² |
| **Measured convergence order** | **2.0005** (ratios 4.004 / 4.001 / 4.000 / 4.000) | 2.00 ± 0.15 |
| Isolated disk half-mass radius, 1.2 Gyr | +9.9 % | ±10 % |
| Isolated disk scale length | +14.2 % | ±15 % |
| Isolated disk median \|z\| | +41.2 % (0.219 → 0.310 kpc) | < 60 % |
| Energy, full 2.2 Gyr run | max \|ΔE\|/\|E₀\| = 1.40 × 10⁻³ | < 5 × 10⁻³ |
| Energy is bounded, not secular | post-merger drift = 2.7 % of peak error | < 25 % |
| Control: RK2 vs leapfrog | RK2 drifts 2.07 × 10⁻³/Gyr, leapfrog −4.8 × 10⁻⁷/Gyr | RK2 > 3× worse |
| Angular momentum | max \|ΔL\|/\|L₀\| = 1.10 × 10⁻³ | < 10⁻² |
| Barnes–Hut force error at θ = 0.7 | median 3.3 × 10⁻³, p99 3.2 × 10⁻² | < 10⁻² / < 5 × 10⁻² |
| Barnes–Hut speed-up at θ = 0.7 | 19.7× | > 5× |
| Spurious COM drift (tree force asymmetry) | 0.129 km/s, 0.15 kpc over 2.2 Gyr | < 1 km/s |

## Deliverables

* `media/hero.png` — 1920 × 1080, t = 0.50 Gyr, just after first pericentre.
* `media/hero.mp4` — 1920 × 1080, 30 fps, 18 s, 540 frames, approach → first
  passage with tails and bridge → merger → remnant.
* `figures/fig1..fig6` — six technical PNGs.
* `results/summary.json` — machine-readable key numbers.

## One thing worth knowing

The isolated-galaxy validation earned its keep. The first model used a disk
scale height of 0.3 kpc with 0.25 kpc softening; the test showed the disk
thickening 65 % in 1.2 Gyr, because a softened vertical force cannot hold a disk
thinner than the softening. Setting z_d = 2 ε (0.4 kpc vs 0.20 kpc) cut it to
41 %. The threshold was not moved — the model was fixed.

## Main limitations

Collisionless only (no gas/star formation/feedback, so the remnant is less
concentrated than a real gas-rich merger); residual numerical disk heating of
tens of percent from halo particles ~9× more massive than disk particles; a
fixed global timestep that is ~2× too coarse for the densest 1 % of particles
(visible as a step in the energy error at pericentre); monopole-only tree at
θ = 0.7 (0.33 % median force error); orbit is e = 0.85 rather than parabolic, so
the merger completes inside the integrated time; equal-mass (1:1) merger only.
Full list in README §10.
