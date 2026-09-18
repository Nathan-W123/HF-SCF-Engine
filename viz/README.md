# viz — orbital animations

Square videos built from the output of `backend/scf_engine.py`. The SCF is run
here, the converged MO coefficients are evaluated on a Cartesian grid, and the
isosurfaces are extracted from that grid. Nothing is drawn by hand and no
external chemistry package is involved — the geometry on screen is the
wavefunction.

```bash
pip install numpy scipy numba scikit-image imageio imageio-ffmpeg pillow
python viz/make_animations.py                 # all three, 1080x1080, into renders/
python viz/make_animations.py --which a       # one of them
python viz/make_animations.py --preview       # 480px and short, for checking layout
```

Each run writes `<name>.mp4` (H.264, yuv420p, faststart — plays inline on
LinkedIn, X and Slack) and a `<name>.png` poster frame next to it. SCF results
are cached under `viz/.cache/` so re-renders skip the calculation.

## The three clips

| | molecule | what it shows |
|---|---|---|
| **A** `A_orbital_rotation.mp4` | benzene, RHF/STO-3G | the HOMO turning through a full revolution — a seamless loop |
| **B** `B_scf_convergence.mp4` | water, RHF/6-31G* | the density settling cycle by cycle, error surfaces evaporating, `max\|ΔP\|` falling ten orders of magnitude, then the converged HOMO |
| **C** `C_orbital_switch.mp4` | benzene, RHF/STO-3G | electron density → electrostatic potential → HOMO−1 → HOMO → LUMO |

## Modules

- **`fields.py`** — grid evaluation. Atomic orbital amplitudes, molecular
  orbitals, the electron density, and the electrostatic potential. The
  potential is a genuine Poisson solve: the electronic term is the 1/r
  convolution of the converged density, done by FFT on a grid zero-padded to
  twice its size so the result is free-space rather than periodic, with the
  analytic nuclear term added on top.
- **`render.py`** — a small software renderer. Atoms and bonds are ray-traced
  analytically as spheres and cylinders, so they stay smooth at any zoom;
  isosurfaces come from marching cubes and are drawn as a depth-buffered point
  splat, with points scattered over each triangle in proportion to its area.
  Surfaces and molecule composite by depth, so lobes and atoms interleave
  correctly.
- **`overlay.py`** — captions, legends, the convergence chart and the colour
  bar, drawn with PIL at output resolution so type stays crisp.
- **`scene.py`** — geometries, the cached SCF driver, camera fitting, colour
  ramps, video output.
- **`make_animations.py`** — the three sequences.

## Checks worth keeping

The renders double as a sanity check on the engine, because a wrong
wavefunction looks wrong:

- ∫ρ dr over the grid comes back within ~0.5% of the electron count.
- The electrostatic potential on the ρ = 0.002 a.u. surface of water runs
  −57 to +55 kcal/mol/e, and benzene's is negative over the π faces and
  positive around the hydrogen rim — both are what the literature maps show.
- Benzene's HOMO and LUMO come out as the degenerate π (e₁g) and π* (e₂u)
  pairs the point group requires.

## Notes on the science shown

- STO-3G is a minimal basis. It gives the right orbital *shapes* and the right
  symmetry, which is what these clips are about, but its orbital energies are
  not quantitative — benzene's Koopmans ionisation energy comes out near
  7.6 eV against an experimental 9.24 eV.
- In B the energy is not monotonic. Intermediate DIIS densities are not
  N-representable, so the energy can dip below the converged value before
  settling; `max|ΔP|` is the honest convergence measure and is what the chart
  plots.
- The `HOMO−1 → HOMO → LUMO` steps in C are dissolves between separately
  computed orbitals, not a physical process.
