# HF-SCF-Engine — validation benchmarks

Reference implementation: **PySCF 2.14.0** (libcint integrals), RHF converged to
`conv_tol = 1e-12`.

The engine uses **Cartesian** Gaussians throughout, so every PySCF reference is
built with `mol.cart = True`. Geometries and basis data are shared between the
two codes where noted, so that each benchmark isolates one layer.

Environment: Python 3.11, NumPy 1.26.4, Numba 0.59.1, 4 cores.

Reproduce:

```bash
cd backend
python benchmarks/bench_boys.py
python benchmarks/bench_integrals.py
python benchmarks/bench_basis_data.py
python benchmarks/bench_energy.py --cases "H2O:cc-pvdz;CH4:6-31g*;C6H6:6-31g"
```

---

## 1. RHF total energy vs PySCF

`bench_energy.py` — engine vs PySCF on identical geometry and basis.
**35/35 molecule–basis combinations converged. Worst residual 0.2 µHa.**

| Molecule | Basis | N | E(engine) / Ha | Residual / mHa | t(engine) |
|---|---|---:|---:|---:|---:|
| H₂     | STO-3G      |  2 | −1.116759310   | −0.000003 | 0.26 s |
| H₂     | cc-pVDZ     | 10 | −1.128700093   | +0.000001 | 0.16 s |
| LiH    | 6-31G       | 11 | −7.979267827   | +0.000000 | 0.24 s |
| H₂O    | STO-3G      |  7 | −74.962932627  | −0.000014 | 0.12 s |
| H₂O    | 6-31G       | 13 | −75.983998495  | +0.000006 | 0.17 s |
| H₂O    | 6-31G*      | 19 | −76.010528600  | −0.000010 | 0.40 s |
| H₂O    | 6-31G**     | 25 | −76.023161383  | −0.000004 | 0.85 s |
| H₂O    | cc-pVDZ     | 25 | −76.027137209  | −0.000003 | 1.29 s |
| NH₃    | 6-31G*      | 21 | −56.184193988  | +0.000103 | 0.77 s |
| CH₄    | 6-31G*      | 23 | −40.195072527  | −0.000006 | 0.72 s |
| HF     | cc-pVDZ     | 20 | −100.019837078 | −0.000002 | 0.82 s |
| CO     | 6-31G       | 18 | −112.667204538 | +0.000021 | 0.62 s |
| N₂     | 6-31G       | 18 | −108.867763298 | +0.000078 | 0.60 s |
| HCl    | 6-31G       | 15 | −460.036920610 | +0.000203 | 0.58 s |
| HCl    | cc-pVDZ     | 24 | −460.089733039 | +0.000001 | 5.09 s |
| H₂CO   | cc-pVDZ     | 40 | −113.876256886 | −0.000003 | 8.08 s |
| C₂H₄   | cc-pVDZ     | 50 | −78.040034731  | −0.000004 | 15.99 s |
| C₆H₆   | STO-3G      | 36 | −227.890600601 | +0.000004 | 8.04 s |
| C₆H₆   | 6-31G       | 66 | −230.623507131 | +0.000027 | 44.27 s |

*(19 of 35 rows shown; the full sweep covers H₂, LiH, H₂O, NH₃, CH₄, HF, CO, N₂,
HCl, H₂CO, C₂H₄ and C₆H₆ over STO-3G, 6-31G, 6-31G\*, 6-31G\*\* and cc-pVDZ.)*

- **Max |residual|: 0.000203 mHa (0.2 µHa)** — HCl/6-31G
- **Median |residual|: 0.000006 mHa (6 nHa)**
- The largest residuals track differences between the Basis Set Exchange tables
  and PySCF's internal ones, not the SCF code — see §3.

## 2. Integrals vs libcint

`bench_integrals.py` — both codes are given the **same geometry (in Bohr) and the
same basis table**, so this measures the integral code alone. Engine AOs are
matched to PySCF AOs by (atom, `lmn`, exponents) and rescaled to PySCF's
normalisation (a diagonal transform, which leaves the RHF energy invariant).

| Quantity | Worst absolute deviation over 12 molecule/basis combinations |
|---|---:|
| Overlap `S`             | 1.0 × 10⁻¹⁵ |
| Kinetic `T`             | 1.1 × 10⁻¹³ |
| Nuclear attraction `V`  | 3.0 × 10⁻¹³ |
| Two-electron `(μν\|λσ)` | 6.5 × 10⁻¹⁴ |

Covers H₂O, CH₄ and HCl over STO-3G, 6-31G, 6-31G\* and cc-pVDZ — i.e. s, p and
d shells, SP shells and general contractions. **Agreement is at machine
precision.**

## 3. Basis-set data

`bench_basis_data.py` — the engine's basis table and PySCF's own are each run
through *PySCF's* SCF, so any energy difference is attributable purely to the
basis data. All sets agree to **< 0.0002 mHa**, the residual being BSE's rounded
digits versus PySCF's internal tables.

## 4. Boys function

`bench_boys.py` — `F_n(x)` against 40-digit mpmath quadrature over
n ∈ {0,1,2,3,4,6,8,10} × x ∈ [0, 200] (128 points):

**Worst relative error 1.0 × 10⁻¹⁵** (both the scalar and the Numba array path).

## 5. Timing

Engine wall time, warm Numba JIT, single-threaded:

| N (basis functions) | System | Time |
|---:|---|---:|
| 7   | H₂O / STO-3G   | 0.12 s |
| 25  | H₂O / cc-pVDZ  | 1.29 s |
| 36  | C₆H₆ / STO-3G  | 8.04 s |
| 50  | C₂H₄ / cc-pVDZ | 15.99 s |
| 66  | C₆H₆ / 6-31G   | 44.27 s |

Consistent with the O(N⁴) integral build that dominates at these sizes. PySCF is
25–45× faster, as expected against a hand-written NumPy/Numba implementation
versus libcint.

---

## Correctness fixes this validation uncovered

Before these fixes the engine's H₂O/STO-3G energy was **927 mHa** from the
reference. Four independent defects, each confirmed against PySCF:

1. **Boys function Taylor series** (`integrals.py`) — the term ratio omitted the
   `(2n+2k−1)` factor, so every term from k = 2 on was wrong. Relative error
   reached **103 %** at n = 4, x = 0.99. The `x < 1` branch covers essentially
   every bonding integral, which is why the error was so large. Replaced with the
   ascending, all-positive series `F_n = e^{-x} Σ (2x)^k (2n−1)!!/(2n+2k+1)!!`,
   plus a stable downward recurrence in the array path.
2. **ERI symmetry projection** (`scf_engine.py`) — screened on
   `Γ(a)=Γ(b) and Γ(c)=Γ(d)`, but an integral vanishes only when
   `Γ(a)⊗Γ(b)⊗Γ(c)⊗Γ(d)` does not contain the totally symmetric irrep. It was
   zeroing non-zero integrals such as `(a₁b₁|a₁b₁)`. Since it built the full
   tensor before projecting, it also saved no work; removed. Symmetry is still
   used where it is exact — the Fock matrix is diagonalised block by block.
3. **Linear-dependency cutoff** (`scf_engine.py`) — set to 0.10, which discards
   well-conditioned basis functions (overlap eigenvalues of 10⁻² are routine in
   split-valence sets). Worth 25 mHa on H₂O/6-31G. This was compensating for
   defect 1; restored to the conventional 10⁻⁶.
4. **Basis-set data** (`basis_data.py`, `basis_fetcher.py`) — the hardcoded Pople
   tables matched published values only for H, He and C. Oxygen 6-31G was wrong
   in the third decimal of every contraction coefficient (≈10 mHa on water) and
   Na–Ar was wrong outright, including the number of p primitives (6-31G needs
   six; the table had three) — **HCl/6-31G came out 28 Ha high**. Separately the
   NWChem parser read only the first coefficient column of a general
   contraction, so every cc-pVXZ set was silently missing most of its shells
   (cc-pVDZ oxygen became 1s1p1d instead of 3s2p1d). The parser now expands all
   columns, and the Pople sets are fetched from the Basis Set Exchange like
   every other set rather than being retyped by hand.
