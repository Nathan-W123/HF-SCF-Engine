# HANDOFF — 03-hypersonic-reentry

**Status: complete.** `make all` runs clean from scratch, `pytest` passes (65 tests,
~60 s), all four validations execute with numeric thresholds and write their reports,
`media/hero.png` and `media/hero.mp4` exist at 1920×1080 from real simulation output, and
six technical figures are in `figures/`.

## What it is

A 3-DOF hypersonic atmospheric-reentry simulator for a 1200 kg, 70° sphere-cone capsule:
USSA76 atmosphere built from the standard's own geopotential layer table, modified-Newtonian
aerodynamics obtained by integrating the pressure field over the real sphere-cone surface
(no fitted `C_D(M)` curve anywhere — the Mach dependence comes from the Rayleigh pitot
`Cp_max`), Sutton–Graves stagnation-point heating with a Detra–Kemp–Riddell-form
cross-check, inverse-square gravity over a spherical Earth, and DOP853 integration with
terminal events. Earth rotation is implemented and validated but switched off for all
reported results.

## Headline result

Flying the **same** 120 km / 7.8 km/s / −5.5° entry at a 20° trim angle of attack with the
lift vector up (`L/D` = 0.3116, computed from the Newtonian surface integral) **halves the
peak deceleration, 15.57 g → 7.85 g**, and nearly doubles the downrange (861 → 1533 km),
but costs a **50 % larger integrated heat load, 7513 → 11 274 J/cm²**. Peak stagnation heat
flux for the ballistic case is **155.1 W/cm² at 55.2 km and 6.73 km/s**, giving a
radiative-equilibrium wall temperature of **2382 K**.

Entry-corridor sweep (90 trajectories): at LEO speed there is **no** skip-out boundary
(7800 m/s is sub-circular); at a super-circular 11 000 m/s the skip-out boundary is located
by bisection at **γ_e = −5.006°**, and the usable corridor between it and the first
captured case exceeding 10 g (−5.75° on the 0.25° sweep grid) is **under 0.75° wide**.

## Validation status — 44 / 44 numeric checks pass

| # | benchmark | key measured number | threshold |
|---|---|---|---|
| 1 | vacuum circular orbit | `\|ΔE/E\|` drift **7.6e-16**, `\|Δh/h\|` **2.9e-16** over 3 periods | < 1e-11 |
| 1 | Kepler period `2π√(a³/μ)` | position closure **1.09e-8 m** | < 1 m |
| 1 | rotating-Earth terms vs independent inertial Cartesian 2-body | max `\|Δr\|` **4.1e-5 m** over 3000 s | < 1e-2 m |
| 2 | Allen–Eggers exact limit | peak deceleration rel. error **−2.5e-9**; whole-profile deviation **5.1e-11** of `a_max` | < 1e-6 / 1e-8 |
| 2 | Allen–Eggers, gravity off, γ = −90° | **−1.6e-12 %** (constant-γ assumption becomes exact) | < 1e-6 |
| 2 | Allen–Eggers, full physics, γ ≤ −30° | worst `a_max` error **4.90 %**, worst peak-altitude error **0.161 km**; residual explained by the gravity-work estimate to **0.78 pp** | < 6 % / 0.5 km / 1.2 pp |
| 3 | USSA76 vs published table, 8 layer boundaries | max rel. error: T **1.5e-16**, p **1.8e-7**, ρ **2.3e-5** | < 1e-9 / 1e-6 / 5e-5 |
| 3 | hydrostatic residual `\|dp/dz + ρg\|/(ρg)`, 0–500 km | median **1.1e-9**, max **5.8e-7** | < 1e-8 / 1e-6 |
| 4 | energy budget `Δ(V²/2 − μ/r) + ∫(D/m)V dt` | **8.9e-15** (ballistic), **9.5e-14** (lifting) | < 1e-10 |
| 4 | RK4 convergence order over 5 refinements | **4.043, 4.022, 4.011, 4.006, 4.008** | 3.8 – 4.3 |
| 4 | DOP853 vs fixed-step RK4 on the full entry | Δaltitude **4.6e-6 m**, Δspeed **2.7e-7 m/s** | < 1 m / 1e-3 m/s |

Reports: `validation/validation_report.txt` (human) and `.json` (machine), plus
`val_01..val_04.{json,txt}`. Every number in the README comes from these files,
`results/summary.json`, `results/cases.json`, `results/corridor.json` or
`media/hero_manifest.json`.

## Hero visual

`media/hero.png` (1920×1080, 2× supersampled) and `media/hero.mp4` (1920×1080, 30 fps,
18 s, 540 frames). The trail's brightness is the simulated Sutton–Graves heat flux; its
colour is the Planck-locus colour of `T_w = (q̇/(εσ))^(1/4)` from that same flux (deep red
on approach → orange-white at the 2382 K peak); the limb glow is a single-scattering
integral of the real USSA76 density along every view ray. Rendered with a numba ray marcher,
additive splatting, three-scale bloom and an ACES-like filmic curve.

## Limitations (short form; full list in README §5)

1. 3-DOF point mass — no attitude dynamics; the lifting case *prescribes* trim and bank.
2. Newtonian aerodynamics is a hypersonic approximation; it is not valid below M ≈ 1.5, yet
   trajectories are integrated to Mach 0.38. No skin friction, no base drag.
3. **Radiative (shock-layer) heating is not modelled.** Negligible at 7.8 km/s, *not*
   negligible for the 11 km/s corridor sweep — those heat fluxes are lower bounds.
4. The DKR cross-check coefficient (1.1037e8) is quoted from memory and could not be
   verified offline; it is labelled as such everywhere and never used for a reported number.
5. Above 86 km the USSA76 kinetic-temperature profile is exact but the mean molecular weight
   is held at its sea-level value (no species diffusion), so densities above ~100 km are
   model output, not validated. Deliberately **no** published USSA76 densities above 86 km
   are quoted, because they could not be verified offline.
6. Static non-rotating atmosphere for all reported results (rotation implemented and
   validated, but disabled); no winds, no density dispersion, no J2, spherical Earth.
7. Earth's surface, clouds and night lights in the render are procedural noise, not data;
   photographic parameters (exposure, bloom, wake persistence) are artistic choices.

## Suggested LinkedIn headline

> Lift halves the g-load and costs 50 % more heat: a validated 3-DOF hypersonic reentry
> simulator — Allen–Eggers reproduced to 2.5 × 10⁻⁹, energy closed to 9 × 10⁻¹⁵, and a hero
> frame whose fireball is coloured by the Planck curve of the simulated 2382 K heat shield.
