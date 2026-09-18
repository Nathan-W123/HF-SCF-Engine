# Simulation portfolio — three independent projects

Three self-contained computational-physics / autonomy projects, each built
end to end with real governing equations, automated tests, executed
validation against numeric thresholds, a one-command reproduction, and a
hero visual rendered entirely from its own simulation output.

The projects share no code. Each directory is independently reproducible and
can be extracted into its own repository with `git subtree split`.

| # | Project | Tests | Validation | Hero |
|---|---------|-------|------------|------|
| 1 | [Galaxy collision](01-galaxy-collision/) | 58 pass | 5 suites, all pass | [png](01-galaxy-collision/media/hero.png) · [mp4](01-galaxy-collision/media/hero.mp4) |
| 2 | [Autonomous fixed-wing swarm](02-swarm-autonomy/) | 97 cases pass | 33 checks, all pass | [png](02-swarm-autonomy/media/hero.png) · [mp4](02-swarm-autonomy/media/hero.mp4) |
| 3 | [Hypersonic reentry](03-hypersonic-reentry/) | 65 pass | 44 checks, all pass | [png](03-hypersonic-reentry/media/hero.png) · [mp4](03-hypersonic-reentry/media/hero.mp4) |

## 1 — Galaxy collision

Barnes–Hut N-body simulation of a major spiral–spiral merger. 72,000
particles across two galaxies, each an exponential disk + Hernquist bulge +
a **live** Hernquist dark-matter halo holding 89% of the mass, so dynamical
friction drives the merger self-consistently rather than being imposed.
Numba-JIT octree with a cell-based group walk, Barnes δ-corrected opening
criterion, symmetric Plummer softening, kick–drift–kick leapfrog over
2.2 Gyr.

**Headline:** progenitor disks at axis ratio c/a = 0.216 merge into a
triaxial pressure-supported remnant at c/a = 0.516, b/a = 0.745 — two
spirals become an elliptical from Newtonian gravity alone. Longest tidal
tail 130 kpc; nuclei merge at t = 1.030 Gyr.

**Validation:** measured leapfrog convergence order 2.0005; energy conserved
to 1.402e-3 and shown *bounded rather than secular* against an RK2 control
of the same formal order (RK2 drifts 2.07e-3/Gyr vs leapfrog -4.8e-7/Gyr);
angular momentum to 1.097e-3; tree force error at θ=0.7 median 3.31e-3 with
a 19.7× speed-up over exact O(N²).

## 2 — Autonomous fixed-wing swarm

Decentralised multi-vehicle autonomy, up to 36 fixed-wing aircraft, no
central allocator. Per-vehicle stack: tangent visibility graph + A* →
L1 nonlinear guidance → 3-D ORCA (soft QP rows) + braking-distance control
barrier function + no-fly-zone barriers (hard rows) → projection onto the
reachable set → coordinated-turn autopilot over an RK4 3-DOF model. Wind is
steady + MIL-F-8785C Dryden turbulence, sampled exactly via van Loan.

**Headline:** 36 aircraft whose nominal routes all pass through one point
hold 68.99 m against a 60 m requirement — zero collisions, zero violations,
100% goal completion — at a mean path-length cost of 0.63%.

**Validation:** turn radius matches V²/(g tan φ) to 4.5e-13; RK4 order
4.035; steady-state cross-track 5.3e-9 m; ORCA reciprocity exact over 1500
pairs; QP agrees with SLSQP to 2.0e-11; fixed-seed runs bit-identical;
realised Dryden RMS within 0.13–1.81% of analytic.

## 3 — Hypersonic atmospheric reentry

3-DOF reentry of a 70° sphere-cone from 120 km at 7.8 km/s. USSA76 built
from the geopotential layer table with every layer base pressure *derived*;
modified-Newtonian aerodynamics by numerical surface integration over the
real geometry, so C_D and C_L come from the shape rather than a fitted Mach
curve; Sutton–Graves convective heating with two cross-checks; DOP853 with
terminal-altitude and skip-out events.

**Headline:** flying the same entry at 20° trim angle of attack
(L/D = 0.3116, from the surface integral) halves peak deceleration from
15.57 g to 7.85 g but raises total heat load 50%, from 7513 to
11,274 J/cm².

**Validation:** Allen–Eggers reproduced to 2.5e-9 in the exact limit, and
under full physics the residual is explained by the neglected gravity-work
term to within 0.78 percentage points; energy budget closes to 8.9e-15;
RK4 observed order 4.01–4.04; USSA76 matches the published table to 1.8e-7
(p) and 2.3e-5 (ρ); vacuum orbit conserves energy to 7.6e-16.

## Hero visuals

Every hero is rendered from real simulation output by a dedicated script.
None uses `plt.scatter`. The shared technique — additive splatting into a
float HDR buffer, multi-scale bloom, filmic tone mapping — is documented in
[RENDERING_PLAYBOOK.md](RENDERING_PLAYBOOK.md).

Colour is derived from simulated quantities, not hand-picked gradients:

- **Galaxy** — stellar population and initial disk radius.
- **Swarm** — cruise altitude sets base hue; bank + avoidance deflection
  blends it to amber, so the orange ribbons *are* the avoidance manoeuvres.
- **Reentry** — trail brightness is the simulated Sutton–Graves heat flux
  and its colour is the Planck-locus colour of the corresponding
  radiative-equilibrium wall temperature, so the fireball is coloured by
  the physics.

## Reproducing

Each project is independent:

```sh
cd 01-galaxy-collision && make all     # ~13 min on 4 idle cores
cd 02-swarm-autonomy   && make all     # ~9-11 min
cd 03-hypersonic-reentry && make all   # ~15 min
```

`make test` runs the test suite, `make validate` the validation suites,
`make figures` the supporting figures and `make hero` the hero assets.
Bulk regenerated state (N-body snapshot series, scenario pickles) is
gitignored and rebuilt by `make all`.

## Scope and honesty notes

Each README carries its own limitations section; the most important:

- **Galaxy** is collisionless — no gas, star formation or feedback — so the
  remnant is less concentrated than a real gas-rich merger. Residual
  numerical disk heating means remnant *thickness* is trustworthy only to
  tens of percent, though tidal structure grows ~10× faster than the
  heating.
- **Swarm** has no sensing model: exact neighbour states, no noise, latency
  or dropout. This is the largest gap to a flyable system. Separation
  degrades to 44.6 m at the top of the turbulence sweep (still zero
  collisions), so the invariant holds at nominal wind, not everywhere. The
  CBF guarantee is for a single integrator, not an airframe with lags —
  hence separation is *measured* rather than claimed as proved.
- **Reentry** omits shock-layer radiative heating, so the 11 km/s corridor
  fluxes are lower bounds. One cross-check coefficient could not be
  verified offline and is flagged everywhere it appears; it is never used
  for a reported number. Earth surface detail in the hero render is
  procedural, and is labelled as such.

No project is validated against flight or observational data; all checks are
analytic benchmarks, conservation laws, convergence studies and internal
consistency.
