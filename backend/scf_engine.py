"""
Hartree-Fock SCF engine — pure NumPy/SciPy, no PySCF.

Implements restricted Hartree-Fock (RHF) with:
  - Roothaan-Hall equations
  - DIIS convergence acceleration
  - Gaussian .cube file generation for MO visualization
"""
from __future__ import annotations

import logging
import os
import tempfile
from typing import Optional

import numpy as np
from math import sqrt

from basis_data import BASES, BASIS_LABELS, AVAILABLE_BASES, FETCHED_BASES
from symmetry import identify_point_group, format_group
from integrals import overlap, kinetic, nuclear, eri, norm_const, _eri_contracted, _njit, eri_3c, eri_2c, _eri_3c_contracted, _eri_2c_contracted
from salc import build_salc

logger = logging.getLogger(__name__)

ANGSTROM_TO_BOHR = 1.8897259886
HARTREE_TO_EV    = 27.211386245
DEBYE_PER_AU     = 2.5417464519  # 1 ea₀ in Debye

ATOMIC_NUMBERS = {
    "H": 1,  "He": 2,
    "Li": 3, "Be": 4, "B": 5,  "C": 6,  "N": 7,  "O": 8,  "F": 9,  "Ne": 10,
    "Na": 11,"Mg": 12,"Al": 13,"Si": 14,"P": 15, "S": 16, "Cl": 17,"Ar": 18,
    "K": 19, "Ca": 20,
    "Sc": 21,"Ti": 22,"V": 23, "Cr": 24,"Mn": 25,"Fe": 26,"Co": 27,"Ni": 28,
    "Cu": 29,"Zn": 30,"Ga": 31,"Ge": 32,"As": 33,"Se": 34,"Br": 35,"Kr": 36,
}


# ── BasisFunction ──────────────────────────────────────────────────────────────

class BasisFunction:
    """Contracted Cartesian Gaussian basis function."""

    def __init__(self, center, lmn, exponents, coefficients, atom_idx=0):
        self.center      = np.array(center, dtype=float)   # Bohr
        self.lmn         = tuple(int(x) for x in lmn)     # (lx, ly, lz)
        self.exponents   = np.array(exponents, dtype=float)
        self.coefficients= np.array(coefficients, dtype=float)
        self.atom_idx    = atom_idx
        l, m, n = self.lmn
        self.norms = np.array([norm_const(a, l, m, n) for a in self.exponents])


# ── Geometry parsing ───────────────────────────────────────────────────────────

def parse_xyz_block(xyz_block: str) -> list[tuple[str, float, float, float]]:
    """Parse XYZ block (with or without count/comment header). Returns Angstrom."""
    lines = [ln.strip() for ln in xyz_block.strip().splitlines() if ln.strip()]
    start = 2 if lines and lines[0].isdigit() else 0
    atoms = []
    for ln in lines[start:]:
        parts = ln.split()
        if len(parts) < 4:
            continue
        sym = parts[0].capitalize()
        try:
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        except ValueError:
            raise ValueError(f"Cannot parse coordinates: '{ln}'")
        atoms.append((sym, x, y, z))
    if not atoms:
        raise ValueError("No valid atom coordinates found in XYZ input")
    return atoms


# ── Basis set construction ─────────────────────────────────────────────────────

def build_basis(
    atoms_bohr: list[tuple[str, float, float, float]],
    basis_name: str,
) -> list[BasisFunction]:
    """Build list of BasisFunction objects for the given atoms and basis set."""
    basis_name = basis_name.lower()
    if basis_name in BASES:
        basis_data = BASES[basis_name]
    elif basis_name in FETCHED_BASES:
        from basis_fetcher import get_basis
        basis_data = get_basis(basis_name)
    else:
        raise ValueError(
            f"Unknown basis '{basis_name}'. Available: {AVAILABLE_BASES}"
        )

    # Cartesian angular momentum components for each l
    AM_COMPONENTS = {
        0: [(0, 0, 0)],
        1: [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
        2: [(2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 0), (1, 0, 1), (0, 1, 1)],
        3: [(3,0,0),(0,3,0),(0,0,3),(2,1,0),(2,0,1),(1,2,0),(0,2,1),(1,0,2),(0,1,2),(1,1,1)],
    }

    bfs = []
    for idx, (sym, x, y, z) in enumerate(atoms_bohr):
        sym = sym.capitalize()
        if sym not in basis_data:
            raise ValueError(
                f"Element '{sym}' not available in basis '{basis_name}'. "
                f"Available elements: {sorted(basis_data.keys())}"
            )
        center = np.array([x, y, z])
        for shell in basis_data[sym]:
            if shell[0] == "SP":
                _, exps, s_coeffs, p_coeffs = shell
                # s part
                bfs.append(BasisFunction(center, (0, 0, 0), exps, s_coeffs, idx))
                # p part
                for lmn in AM_COMPONENTS[1]:
                    bfs.append(BasisFunction(center, lmn, exps, p_coeffs, idx))
            else:
                l, exps, coeffs = shell
                for lmn in AM_COMPONENTS.get(l, []):
                    bfs.append(BasisFunction(center, lmn, exps, coeffs, idx))
    return bfs


# ── Calculation time estimator ─────────────────────────────────────────────────

# Primitive overhead factors relative to STO-3G (calibrated empirically).
# Accounts for larger primitive counts in higher-quality basis sets.
_PRIM_FACTOR: dict[str, float] = {
    "sto-3g":      1.0,
    "6-31g":       2.5,
    "6-31g*":      3.0,
    "6-31g**":     3.0,
    "cc-pvdz":     4.0,
    "cc-pvtz":     7.0,
    "cc-pvqz":    12.0,
    "aug-cc-pvdz": 5.0,
    "aug-cc-pvtz": 10.0,
    "aug-cc-pvqz": 18.0,
    # Calendar — similar primitive cost to aug, slightly fewer shells
    "jul-cc-pvdz": 4.8,  "jul-cc-pvtz": 9.5,  "jul-cc-pvqz": 17.0,
    "jun-cc-pvdz": 4.6,  "jun-cc-pvtz": 9.0,  "jun-cc-pvqz": 16.0,
    "may-cc-pvdz": 4.3,  "may-cc-pvtz": 8.5,
    "apr-cc-pvdz": 4.1,  "apr-cc-pvtz": 8.0,
    "mar-cc-pvdz": 4.0,
}

# Calibrated from benzene STO-3G benchmark: N=36, ~12.5 s (warm Numba JIT).
# n_pairs = 36*37//2 = 666 → n_quartets = 666*667//2 = 222,111
# k = 12.5 / 222111 ≈ 5.63e-5 s per unique quartet (STO-3G primitives)
_K_SECONDS_PER_QUARTET = 5.63e-5

# Calibration for JIT-compiled 3-centre integrals (RI-JK B tensor build).
# 3c integrals loop over 3 primitive indices vs 4 for 4c; empirically ~0.35× the cost.
# → K_3c ≈ 0.35 × K_4c ≈ 2e-5 s per 3c integral at pf=1 (STO-3G-like primitives).
_K_SECONDS_PER_3C = 2.0e-5

# Orbital basis → fitting auxiliary basis (BSE name).
# Used by RI-JK to auto-select the JKFIT basis.
_AUX_BASIS_MAP: dict[str, str] = {
    "sto-3g":      "def2-svp-jkfit",
    "6-31g":       "def2-svp-jkfit",
    "6-31g*":      "def2-svp-jkfit",
    "6-31g**":     "def2-svp-jkfit",
    "def2-svp":    "def2-svp-jkfit",
    "def2-tzvp":   "def2-tzvp-jkfit",
    "cc-pvdz":     "cc-pvdz-jkfit",
    "cc-pvtz":     "cc-pvtz-jkfit",
    "cc-pvqz":     "cc-pvqz-jkfit",
    "aug-cc-pvdz": "aug-cc-pvdz-jkfit",
    "aug-cc-pvtz": "aug-cc-pvtz-jkfit",
    # Calendar bases use the corresponding aug JKFIT auxiliary
    "jul-cc-pvdz": "aug-cc-pvdz-jkfit",
    "jul-cc-pvtz": "aug-cc-pvtz-jkfit",
    "jul-cc-pvqz": "aug-cc-pvqz-jkfit",
    "jun-cc-pvdz": "aug-cc-pvdz-jkfit",
    "jun-cc-pvtz": "aug-cc-pvtz-jkfit",
    "jun-cc-pvqz": "aug-cc-pvqz-jkfit",
    "may-cc-pvdz": "aug-cc-pvdz-jkfit",
    "may-cc-pvtz": "aug-cc-pvtz-jkfit",
    "apr-cc-pvdz": "aug-cc-pvdz-jkfit",
    "apr-cc-pvtz": "aug-cc-pvtz-jkfit",
    "mar-cc-pvdz": "aug-cc-pvdz-jkfit",
}


def estimate_rhf(
    xyz_block: str,
    basis: str,
    charge: int = 0,
    spin: int = 0,
) -> dict:
    """
    Estimate RHF calculation time without running it.

    Returns a dict with:
      n_basis, n_electrons, n_dominant_integrals, method,
      estimated_seconds_{low,mid,high}, memory_mb, warning
    """
    atoms_ang = parse_xyz_block(xyz_block)
    atoms_bohr = [
        (s, x * ANGSTROM_TO_BOHR, y * ANGSTROM_TO_BOHR, z * ANGSTROM_TO_BOHR)
        for s, x, y, z in atoms_ang
    ]

    bfs = build_basis(atoms_bohr, basis)
    N = len(bfs)

    n_elec = sum(ATOMIC_NUMBERS.get(s, 0) for s, *_ in atoms_ang) - charge

    pf = _PRIM_FACTOR.get(basis.lower(), 3.0)

    if N < 100:
        # ── Stored ERI tensor path ─────────────────────────────────────────────
        # One-time O(N⁴) compute; each SCF cycle uses a fast einsum over the tensor.
        n_pairs     = N * (N + 1) // 2
        n_integrals = n_pairs * (n_pairs + 1) // 2   # unique (ij|kl) quartets
        t_mid  = _K_SECONDS_PER_QUARTET * n_integrals * pf
        t_low  = t_mid * 0.4
        t_high = t_mid * 3.0
        memory_mb = N ** 4 * 8 / 1_000_000
        method = "stored-ERI"
    else:
        # ── RI-JK density-fitting path ─────────────────────────────────────────
        # Bottleneck: building B[P,μ,ν] via JIT-compiled 3-centre integrals (once).
        # Per-SCF einsum cost O(M·N²) is negligible in comparison.
        M_approx    = 5 * N                           # JKFIT aux basis ≈ 5× orbital
        n_3c        = M_approx * N * (N + 1) // 2    # unique (μν|P) triples
        n_2c        = M_approx * (M_approx + 1) // 2 # unique (P|Q) metric pairs
        # 3c primitive loops span 3 indices (vs 4 for 4c) → pf scales as pf^(3/4)
        pf_3c  = pf ** 0.75
        t_mid  = (n_3c + n_2c * 0.3) * _K_SECONDS_PER_3C * pf_3c
        t_low  = t_mid * 0.35   # JIT + good hardware
        t_high = t_mid * 2.5    # slow hardware / first JIT compile
        memory_mb   = M_approx * N * N * 8 / 1_000_000
        n_integrals = n_3c
        method = "RI-JK"

    warning: str | None = None
    if N < 100 and memory_mb > 4_000:
        warning = (f"ERI tensor requires ~{memory_mb / 1000:.1f} GB RAM — "
                   "consider a smaller basis set")
    elif t_mid > 3600:
        warning = "Very long calculation expected — consider a smaller basis set or molecule"

    return {
        "n_basis": N,
        "n_electrons": n_elec,
        "n_dominant_integrals": n_integrals,
        "method": method,
        "estimated_seconds_low": t_low,
        "estimated_seconds_mid": t_mid,
        "estimated_seconds_high": t_high,
        "memory_mb": memory_mb,
        "warning": warning,
    }


# ── One-electron integral matrices ────────────────────────────────────────────

def compute_one_electron(
    bfs: list[BasisFunction],
    atoms_bohr: list[tuple[str, float, float, float]],
    nuclear_charges: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute overlap S, kinetic T, and nuclear V matrices."""
    N = len(bfs)
    S = np.zeros((N, N))
    T = np.zeros((N, N))
    V = np.zeros((N, N))

    for i in range(N):
        for j in range(i, N):
            S[i, j] = S[j, i] = overlap(bfs[i], bfs[j])
            T[i, j] = T[j, i] = kinetic(bfs[i], bfs[j])
            v_ij = 0.0
            for (sym, cx, cy, cz), Z in zip(atoms_bohr, nuclear_charges):
                C = np.array([cx, cy, cz])
                v_ij += nuclear(bfs[i], bfs[j], C, Z)
            V[i, j] = V[j, i] = v_ij

    return S, T, V


# ── Two-electron repulsion integrals ──────────────────────────────────────────

def _salc_symmetry_project(
    ERI: np.ndarray,
    U: np.ndarray,
    sym_blocks: list,
) -> int:
    """
    Project the AO ERI tensor onto the SALC symmetry-allowed subspace in-place.

    Selection rule in the SALC basis: (ãb̃|c̃d̃) = 0 unless Γ(ã)=Γ(b̃) AND Γ(c̃)=Γ(d̃).

    Algorithm:
      1. Forward 4-index transform: ERI_salc = U^T ⊗ U^T ⊗ U^T ⊗ U^T · ERI_ao
      2. Zero all (a,b,c,d) where block(a)≠block(b) or block(c)≠block(d)
      3. Back 4-index transform: ERI_ao = U ⊗ U ⊗ U ⊗ U · ERI_salc

    Returns the number of SALC-basis quartets set to zero.
    """
    N = U.shape[0]

    # Map each SALC index to its block index
    block_of = np.empty(N, dtype=np.intp)
    for b_idx, b in enumerate(sym_blocks):
        block_of[b] = b_idx

    # Build allowed mask: (N,N,N,N) bool — True where selection rule is satisfied
    same_bra = block_of[:, None] == block_of[None, :]   # (N,N)
    same_ket = same_bra                                   # same shape, reuse
    allowed = (same_bra[:, :, np.newaxis, np.newaxis]
               & same_ket[np.newaxis, np.newaxis, :, :])  # (N,N,N,N)

    n_sym_skipped = int(np.count_nonzero(~allowed))

    # Forward transform: AO → SALC  (contract each index with U)
    E = np.einsum("ia,ijkl->ajkl", U, ERI)
    E = np.einsum("jb,ajkl->abkl", U, E)
    E = np.einsum("kc,abkl->abcl", U, E)
    E = np.einsum("ld,abcl->abcd", U, E)

    # Apply selection rule
    E *= allowed

    # Back transform: SALC → AO  (contract each index with U^T, i.e. U since U is orthogonal)
    E = np.einsum("ia,abcd->ibcd", U, E)
    E = np.einsum("jb,ibcd->ijcd", U, E)
    E = np.einsum("kc,ijcd->ijkd", U, E)
    E = np.einsum("ld,ijkd->ijkl", U, E)

    ERI[:] = E
    return n_sym_skipped


def compute_eri(
    bfs: list[BasisFunction],
    cs_tol: float = 1e-9,
    U: np.ndarray | None = None,
    sym_blocks: list | None = None,
) -> np.ndarray:
    """
    Compute the full (μν|λσ) ERI tensor.

    Optimisations applied:
      1. 8-fold permutation symmetry  (μν|λσ) = (νμ|λσ) = (μν|σλ) = (λσ|μν) ...
      2. Compound index restriction ij >= kl  -> unique quartets only
      3. Cauchy-Schwarz screening: |(μν|λσ)| <= sqrt(μν|μν) * sqrt(λσ|λσ)
      4. SALC symmetry projection (when U and sym_blocks provided):
         Transform to SALC basis, zero cross-irrep quartets, transform back.
         Selection rule: (ãb̃|c̃d̃) = 0 unless Γ(ã)=Γ(b̃) AND Γ(c̃)=Γ(d̃).
    """
    N = len(bfs)

    # Step 1: diagonal shell-pair norms  Q[i,j] = sqrt|(ij|ij)|
    Q = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1):
            v = abs(eri(bfs[i], bfs[j], bfs[i], bfs[j])) ** 0.5
            Q[i, j] = Q[j, i] = v

    # Step 2: screened ERI build
    ERI = np.zeros((N, N, N, N))
    n_computed = 0
    n_screened = 0

    for i in range(N):
        for j in range(i + 1):
            ij  = i * (i + 1) // 2 + j
            Qij = Q[i, j]
            if Qij < cs_tol:
                n_screened += ij + 1
                continue
            for k in range(N):
                for l in range(k + 1):
                    kl = k * (k + 1) // 2 + l
                    if ij < kl:
                        continue
                    if Qij * Q[k, l] < cs_tol:
                        n_screened += 1
                        continue
                    val = eri(bfs[i], bfs[j], bfs[k], bfs[l])
                    n_computed += 1
                    for a, b, c, d in (
                        (i, j, k, l), (j, i, k, l), (i, j, l, k), (j, i, l, k),
                        (k, l, i, j), (l, k, i, j), (k, l, j, i), (l, k, j, i),
                    ):
                        ERI[a, b, c, d] = val

    logger.info(
        "ERI: %d computed, %d Cauchy-Schwarz screened",
        n_computed, n_screened,
    )

    # Step 3: SALC symmetry projection (correct selection rule in irrep basis)
    if U is not None and sym_blocks is not None and len(sym_blocks) > 1:
        n_sym_skipped = _salc_symmetry_project(ERI, U, sym_blocks)
        total = N ** 4
        sym_pct = 100.0 * n_sym_skipped / total if total else 0.0
        logger.info(
            "ERI symmetry projection: %d / %d SALC quartets zeroed (%.1f%%)",
            n_sym_skipped, total, sym_pct,
        )

    return ERI


# ── Direct J/K build (no N⁴ tensor storage) ───────────────────────────────────

_CS_TOL = 1e-9  # Cauchy-Schwarz screening threshold (shared with compute_eri default)


def _compute_cs_norms(bfs: list[BasisFunction]) -> np.ndarray:
    """
    Cauchy-Schwarz diagonal: Q[i,j] = |⟨ij|ij⟩|^{1/2}.

    Computed once per geometry; used every SCF cycle to screen quartets via
    |(μν|λσ)| ≤ Q[μ,ν] · Q[λ,σ].
    """
    N = len(bfs)
    Q = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1):
            v = abs(eri(bfs[i], bfs[j], bfs[i], bfs[j])) ** 0.5
            Q[i, j] = Q[j, i] = v
    return Q


def _pack_bf_arrays(bfs: list[BasisFunction]):
    """Pack BasisFunction list into flat numpy arrays for Numba JIT consumption."""
    N      = len(bfs)
    max_np = max(len(bf.exponents) for bf in bfs)
    bf_exps    = np.zeros((N, max_np))
    bf_coeffs  = np.zeros((N, max_np))
    bf_norms   = np.zeros((N, max_np))
    bf_lmn     = np.zeros((N, 3), dtype=np.int64)
    bf_centers = np.zeros((N, 3))
    bf_nexps   = np.zeros(N, dtype=np.int64)
    for i, bf in enumerate(bfs):
        np_ = len(bf.exponents)
        bf_exps[i, :np_]   = bf.exponents
        bf_coeffs[i, :np_] = bf.coefficients
        bf_norms[i, :np_]  = bf.norms
        bf_lmn[i]          = bf.lmn
        bf_centers[i]      = bf.center
        bf_nexps[i]        = np_
    return bf_exps, bf_coeffs, bf_norms, bf_lmn, bf_centers, bf_nexps


@_njit
def _jk_direct_core(
    N, bf_exps, bf_coeffs, bf_norms, bf_lmn, bf_centers, bf_nexps,
    P, Q, tol,
):
    """
    Numba-JIT quartet loop: builds J and K without storing any N⁴ tensor.

    Screening applied before every ERI evaluation:
      Cauchy-Schwarz + density pre-screen: Q[i,j] * Q[k,l] * P_max < tol → skip
    Both tests collapse to one multiply/compare so there is no branch overhead.
    """
    J = np.zeros((N, N))
    K = np.zeros((N, N))

    # Density pre-screen value: max |P| element (computed once per call).
    P_max = 0.0
    for r in range(N):
        for s in range(N):
            v = P[r, s] if P[r, s] >= 0.0 else -P[r, s]
            if v > P_max:
                P_max = v

    for i in range(N):
        for j in range(i + 1):
            ij  = i * (i + 1) // 2 + j
            Qij = Q[i, j]
            if Qij * P_max < tol:
                continue
            npi = bf_nexps[i]
            npj = bf_nexps[j]
            for k in range(N):
                for l in range(k + 1):
                    kl = k * (k + 1) // 2 + l
                    if ij < kl:
                        continue
                    if Qij * Q[k, l] * P_max < tol:
                        continue
                    npk = bf_nexps[k]
                    npl = bf_nexps[l]

                    val = _eri_contracted(
                        bf_exps[i, :npi], bf_coeffs[i, :npi], bf_norms[i, :npi],
                        bf_lmn[i, 0], bf_lmn[i, 1], bf_lmn[i, 2],
                        bf_centers[i, 0], bf_centers[i, 1], bf_centers[i, 2],
                        bf_exps[j, :npj], bf_coeffs[j, :npj], bf_norms[j, :npj],
                        bf_lmn[j, 0], bf_lmn[j, 1], bf_lmn[j, 2],
                        bf_centers[j, 0], bf_centers[j, 1], bf_centers[j, 2],
                        bf_exps[k, :npk], bf_coeffs[k, :npk], bf_norms[k, :npk],
                        bf_lmn[k, 0], bf_lmn[k, 1], bf_lmn[k, 2],
                        bf_centers[k, 0], bf_centers[k, 1], bf_centers[k, 2],
                        bf_exps[l, :npl], bf_coeffs[l, :npl], bf_norms[l, :npl],
                        bf_lmn[l, 0], bf_lmn[l, 1], bf_lmn[l, 2],
                        bf_centers[l, 0], bf_centers[l, 1], bf_centers[l, 2],
                    )

                    cross = not (i == k and j == l)
                    fkl   = 2 - int(k == l)
                    J[i, j] += fkl * P[k, l] * val
                    if i != j:
                        J[j, i] += fkl * P[k, l] * val
                    if cross:
                        fij = 2 - int(i == j)
                        J[k, l] += fij * P[i, j] * val
                        if k != l:
                            J[l, k] += fij * P[i, j] * val

                    K[i, k] += P[j, l] * val
                    if cross:
                        K[k, i] += P[j, l] * val
                    if i != j:
                        K[j, k] += P[i, l] * val
                        if cross:
                            K[k, j] += P[i, l] * val
                    if k != l:
                        K[i, l] += P[j, k] * val
                        if cross:
                            K[l, i] += P[j, k] * val
                    if i != j and k != l:
                        K[j, l] += P[i, k] * val
                        if cross:
                            K[l, j] += P[i, k] * val

    return J, K


@_njit
def _ri_3c_build_core(
    orb_exps, orb_coeffs, orb_norms, orb_lmn, orb_centers, orb_nexps,
    aux_exps, aux_coeffs, aux_norms, aux_lmn, aux_centers, aux_nexps,
    B0,
):
    """
    Numba-JIT triple loop: fill B0[P, μ, ν] = (μν|P) for all P, μ≥ν,
    then mirror to the upper triangle.
    """
    N = orb_exps.shape[0]
    M = aux_exps.shape[0]
    for P in range(M):
        nP = aux_nexps[P]
        l3 = aux_lmn[P, 0]; m3 = aux_lmn[P, 1]; n3 = aux_lmn[P, 2]
        Cx = aux_centers[P, 0]; Cy = aux_centers[P, 1]; Cz = aux_centers[P, 2]
        for mu in range(N):
            nmu = orb_nexps[mu]
            l1 = orb_lmn[mu, 0]; m1 = orb_lmn[mu, 1]; n1 = orb_lmn[mu, 2]
            Ax = orb_centers[mu, 0]; Ay = orb_centers[mu, 1]; Az = orb_centers[mu, 2]
            for nu in range(mu + 1):
                nnu = orb_nexps[nu]
                l2 = orb_lmn[nu, 0]; m2 = orb_lmn[nu, 1]; n2 = orb_lmn[nu, 2]
                Bx = orb_centers[nu, 0]; By = orb_centers[nu, 1]; Bz = orb_centers[nu, 2]
                v = _eri_3c_contracted(
                    orb_exps[mu, :nmu], orb_coeffs[mu, :nmu], orb_norms[mu, :nmu],
                    l1, m1, n1, Ax, Ay, Az,
                    orb_exps[nu, :nnu], orb_coeffs[nu, :nnu], orb_norms[nu, :nnu],
                    l2, m2, n2, Bx, By, Bz,
                    aux_exps[P, :nP], aux_coeffs[P, :nP], aux_norms[P, :nP],
                    l3, m3, n3, Cx, Cy, Cz,
                )
                B0[P, mu, nu] = v
                B0[P, nu, mu] = v


@_njit
def _ri_2c_metric_core(
    aux_exps, aux_coeffs, aux_norms, aux_lmn, aux_centers, aux_nexps,
    J_metric,
):
    """Numba-JIT double loop: fill J_metric[P, Q] = (P|Q) for all P≥Q."""
    M = aux_exps.shape[0]
    for P in range(M):
        nP = aux_nexps[P]
        l1 = aux_lmn[P, 0]; m1 = aux_lmn[P, 1]; n1 = aux_lmn[P, 2]
        Ax = aux_centers[P, 0]; Ay = aux_centers[P, 1]; Az = aux_centers[P, 2]
        for Q in range(P + 1):
            nQ = aux_nexps[Q]
            l2 = aux_lmn[Q, 0]; m2 = aux_lmn[Q, 1]; n2 = aux_lmn[Q, 2]
            Bx = aux_centers[Q, 0]; By = aux_centers[Q, 1]; Bz = aux_centers[Q, 2]
            v = _eri_2c_contracted(
                aux_exps[P, :nP], aux_coeffs[P, :nP], aux_norms[P, :nP],
                l1, m1, n1, Ax, Ay, Az,
                aux_exps[Q, :nQ], aux_coeffs[Q, :nQ], aux_norms[Q, :nQ],
                l2, m2, n2, Bx, By, Bz,
            )
            J_metric[P, Q] = v
            J_metric[Q, P] = v


def _build_jk_direct(
    bfs: list[BasisFunction],
    P: np.ndarray,
    Q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build Coulomb J and exchange K directly — no N⁴ ERI tensor stored.

    Packs BasisFunction objects into flat arrays and delegates to the
    Numba-JIT _jk_direct_core.  Applies Cauchy-Schwarz + density (P_max)
    pre-screening so negligible quartets are skipped before any ERI evaluation.
    """
    packed = _pack_bf_arrays(bfs)
    return _jk_direct_core(len(bfs), *packed, P, Q, _CS_TOL)


# ── RI-JK density fitting ─────────────────────────────────────────────────────

# Angular momentum → Cartesian components (up to l=5 for JKFIT aux bases)
_AUX_AM_COMPONENTS: dict[int, list] = {
    0: [(0,0,0)],
    1: [(1,0,0),(0,1,0),(0,0,1)],
    2: [(2,0,0),(0,2,0),(0,0,2),(1,1,0),(1,0,1),(0,1,1)],
    3: [(3,0,0),(0,3,0),(0,0,3),(2,1,0),(2,0,1),(1,2,0),(0,2,1),(1,0,2),(0,1,2),(1,1,1)],
    4: [(4,0,0),(0,4,0),(0,0,4),(3,1,0),(3,0,1),(1,3,0),(0,3,1),(1,0,3),(0,1,3),
        (2,2,0),(2,0,2),(0,2,2),(2,1,1),(1,2,1),(1,1,2)],
    5: [(5,0,0),(0,5,0),(0,0,5),(4,1,0),(4,0,1),(1,4,0),(0,4,1),(1,0,4),(0,1,4),
        (3,2,0),(3,0,2),(2,3,0),(0,3,2),(2,0,3),(0,2,3),
        (3,1,1),(1,3,1),(1,1,3),(2,2,1),(2,1,2),(1,2,2)],
}


def _build_aux_basis(atoms_bohr: list, aux_basis_name: str) -> list:
    """
    Build auxiliary BasisFunction list from a BSE JKFIT/RIFIT basis.
    Fetches via basis_fetcher (internet needed once; cached thereafter).
    """
    from basis_fetcher import get_basis
    basis_data = get_basis(aux_basis_name)
    aux_bfs: list[BasisFunction] = []
    for sym, x, y, z in atoms_bohr:
        center = np.array([x, y, z])
        shells = basis_data.get(sym, [])
        for shell in shells:
            if shell[0] == "SP":
                _, exps, s_coeffs, p_coeffs = shell
                ea = np.array(exps); sc = np.array(s_coeffs); pc = np.array(p_coeffs)
                aux_bfs.append(BasisFunction(center, (0,0,0), ea, sc))
                for lmn in [(1,0,0),(0,1,0),(0,0,1)]:
                    aux_bfs.append(BasisFunction(center, lmn, ea, pc))
            else:
                l_val, exps, coeffs = shell
                ea = np.array(exps); ca = np.array(coeffs)
                for lmn in _AUX_AM_COMPONENTS.get(l_val, []):
                    aux_bfs.append(BasisFunction(center, lmn, ea, ca))
    return aux_bfs


def _compute_ri_B_tensor(
    bfs: list, atoms_bohr: list, orbital_basis: str
) -> np.ndarray:
    """
    Build the fitted 3-index RI tensor B[P, μ, ν] = Σ_Q (J^{-1/2})[QP] (μν|Q).

    Steps
    -----
    1. Fetch auxiliary basis.
    2. Compute raw 3-centre integrals  B0[P, μ, ν] = (μν|P).
    3. Compute 2-centre Coulomb metric  J[P, Q] = (P|Q).
    4. Factorise J = V D V^T  →  J^{-1/2} = V D^{-1/2} V^T.
    5. Return  B = J^{-1/2} ⊗ B0  (contraction over auxiliary index).

    Returns B of shape (M, N, N) where M = len(aux_bfs), N = len(bfs).
    Raises RuntimeError if the auxiliary basis cannot be fetched.
    """
    aux_name = _AUX_BASIS_MAP.get(orbital_basis.lower())
    if aux_name is None:
        raise RuntimeError(
            f"No RI auxiliary basis mapped for '{orbital_basis}'. "
            f"Supported: {sorted(_AUX_BASIS_MAP)}"
        )

    aux_bfs = _build_aux_basis(atoms_bohr, aux_name)
    N = len(bfs)
    M = len(aux_bfs)
    logger.info("RI-JK: %d auxiliary functions (%s)", M, aux_name)

    # ── Step 2: raw 3-centre integrals (JIT-compiled outer loop) ──────────────
    orb_arrs = _pack_bf_arrays(bfs)
    aux_arrs = _pack_bf_arrays(aux_bfs)
    B0 = np.zeros((M, N, N))
    _ri_3c_build_core(*orb_arrs, *aux_arrs, B0)
    logger.info("  RI 3c integrals: done (%d aux × %d orb pairs)", M, N * (N + 1) // 2)

    # ── Step 3: 2-centre Coulomb metric (JIT-compiled outer loop) ─────────────
    J_metric = np.zeros((M, M))
    _ri_2c_metric_core(*aux_arrs, J_metric)

    # ── Step 4: J^{-1/2} via eigendecomposition ───────────────────────────────
    eigvals, eigvecs = np.linalg.eigh(J_metric)
    # Drop near-zero eigenvalues (linear dependencies in aux basis)
    inv_sqrt = np.where(eigvals > 1e-10, eigvals ** -0.5, 0.0)
    J_inv_sqrt = eigvecs @ np.diag(inv_sqrt) @ eigvecs.T

    # ── Step 5: contract B0 with J^{-1/2} ─────────────────────────────────────
    # B[P, μ, ν] = Σ_Q J_inv_sqrt[Q, P] * B0[Q, μ, ν]
    B = np.einsum("QP,Qmn->Pmn", J_inv_sqrt, B0)
    return B


def _build_jk_ri(B: np.ndarray, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    RI-JK Coulomb and exchange matrices from the fitted B tensor.

    B[P, μ, ν] = Σ_Q (J^{-1/2})[QP] (μν|Q)  — precomputed, shape (M, N, N).

    J[μ,ν]  = Σ_P d_P B[P,μ,ν]       d_P = Σ_{λσ} B[P,λ,σ] P[λ,σ]
    K[μ,λ]  = Σ_{P,ν,σ} B[P,μ,ν] P[ν,σ] B[P,λ,σ]
            = Σ_P (B[P] P B[P]^T)[μ,λ]   (O(M·N²) per cycle)
    """
    d = np.einsum("Pmn,mn->P",   B, P)        # (M,)
    J = np.einsum("Pmn,P->mn",   B, d)        # (N,N)
    T = np.tensordot(B, P, axes=([2], [0]))   # (M,N,N): T[P,μ,σ] = Σ_ν B[P,μ,ν] P[ν,σ]
    K = np.einsum("Pms,Pls->ml", T, B)        # (N,N)
    return J, K


# ── DIIS extrapolation ────────────────────────────────────────────────────────

class DIIS:
    """Pulay DIIS error-vector extrapolation (m=6 history)."""

    def __init__(self, m: int = 6):
        self.m = m
        self._focks  = []
        self._errors = []

    def reset(self):
        self._focks.clear()
        self._errors.clear()

    def push(self, F: np.ndarray, err: np.ndarray):
        self._focks.append(F.copy())
        self._errors.append(err.copy())
        if len(self._focks) > self.m:
            self._focks.pop(0)
            self._errors.pop(0)

    def extrapolate(self) -> Optional[np.ndarray]:
        n = len(self._focks)
        if n < 2:
            return None
        # Build B matrix: B_ij = Tr(e_i · e_j)
        B = np.zeros((n + 1, n + 1))
        B[-1, :] = B[:, -1] = -1.0
        B[-1, -1] = 0.0
        for i in range(n):
            for j in range(n):
                B[i, j] = float(np.einsum("ij,ij->", self._errors[i], self._errors[j]))
        rhs = np.zeros(n + 1)
        rhs[-1] = -1.0
        # lstsq handles near-singular B (near-duplicate history) via minimum-norm
        # solution, which reduces to equal-weight averaging — stable near convergence.
        try:
            coeff, _, _, _ = np.linalg.lstsq(B, rhs, rcond=None)
        except np.linalg.LinAlgError:
            return None
        # Guard against wildly oscillating coefficients
        if np.max(np.abs(coeff[:-1])) > 50.0:
            return None
        F_new = np.zeros_like(self._focks[0])
        for i in range(n):
            F_new += coeff[i] * self._focks[i]
        return F_new


# ── Dipole integral matrix ────────────────────────────────────────────────────

def _dipole_matrix(bfs: list[BasisFunction], component: int) -> np.ndarray:
    """⟨χ_μ|x_c|χ_ν⟩ matrix, component = 0/1/2 for x/y/z."""
    from math import pi, sqrt
    from integrals import _E

    N = len(bfs)
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(i, N):
            val = 0.0
            bf1, bf2 = bfs[i], bfs[j]
            for a, da, Na in zip(bf1.exponents, bf1.coefficients, bf1.norms):
                for b, db, Nb in zip(bf2.exponents, bf2.coefficients, bf2.norms):
                    p = a + b
                    rt = sqrt(pi / p)
                    Q = bf1.center - bf2.center

                    # Per-dimension E0 coefficients
                    E = [_E(bf1.lmn[k], bf2.lmn[k], 0, Q[k], a, b, {}) * rt
                         for k in range(3)]

                    # dipole in dim = component:
                    # ⟨x^l1 e^{-a x²}|x|(x-Bx)^l2 e^{-b x²}⟩
                    #   = (E0^{l1+1,l2} + Ax * E0^{l1,l2}) * sqrt(π/p)
                    # then multiply by the other two 1D overlaps
                    c = component
                    lmn1 = bf1.lmn[c]
                    lmn2 = bf2.lmn[c]
                    Qc   = Q[c]
                    Ax   = bf1.center[c]
                    dip_1d = (_E(lmn1 + 1, lmn2, 0, Qc, a, b, {}) + Ax * _E(lmn1, lmn2, 0, Qc, a, b, {})) * rt

                    # Product of other two overlaps
                    other = 1.0
                    for k in range(3):
                        if k != c:
                            other *= E[k]

                    val += Na * Nb * da * db * dip_1d * other
            D[i, j] = D[j, i] = val
    return D


# ── SAD initial guess ─────────────────────────────────────────────────────────

def _sad_initial_guess(
    bfs: list,
    atoms_bohr: list,
    nuclear_charges: list,
) -> np.ndarray:
    """
    Superposition of Atomic Densities (SAD) initial guess.

    For each atom, build a per-atom H_core (using only that atom's nuclear
    charge), orthogonalize, diagonalize, and fill with fractional Aufbau
    (equal occupation within degenerate groups).  The resulting per-atom
    density matrices are placed on the block diagonal of the global P.
    """
    N = len(bfs)
    P_sad = np.zeros((N, N))

    # Group basis-function objects and their global indices by atom
    atom_bfs: dict[int, list] = {}
    atom_bf_idx: dict[int, list] = {}
    for i, bf in enumerate(bfs):
        a = bf.atom_idx
        atom_bfs.setdefault(a, []).append(bf)
        atom_bf_idx.setdefault(a, []).append(i)

    for a_idx, (sym, ax, ay, az) in enumerate(atoms_bohr):
        Z_a  = float(nuclear_charges[a_idx])
        lbfs = atom_bfs.get(a_idx, [])
        gidx = atom_bf_idx.get(a_idx, [])
        n_a  = len(lbfs)
        if n_a == 0:
            continue

        # Per-atom one-electron integrals (nuclear attraction from this atom only)
        Sa = np.zeros((n_a, n_a))
        Ta = np.zeros((n_a, n_a))
        Va = np.zeros((n_a, n_a))
        Ac = np.array([ax, ay, az])

        for i_loc in range(n_a):
            for j_loc in range(i_loc, n_a):
                Sa[i_loc, j_loc] = Sa[j_loc, i_loc] = overlap(lbfs[i_loc], lbfs[j_loc])
                Ta[i_loc, j_loc] = Ta[j_loc, i_loc] = kinetic(lbfs[i_loc], lbfs[j_loc])
                Va[i_loc, j_loc] = Va[j_loc, i_loc] = nuclear(lbfs[i_loc], lbfs[j_loc], Ac, Z_a)

        Ha = Ta + Va

        # Canonical Löwdin orthogonalization of per-atom basis
        va, wa = np.linalg.eigh(Sa)
        inv_sqrt = np.where(va > 1e-8, va ** -0.5, 0.0)
        Xa = wa @ np.diag(inv_sqrt) @ wa.T

        # Diagonalize per-atom H_core in orthogonal basis
        Haz       = Xa.T @ Ha @ Xa
        eps_a, Cp = np.linalg.eigh(Haz)
        Ca        = Xa @ Cp

        # Fractional Aufbau: distribute Z_a electrons over MOs.
        # Degenerate orbitals (within 1e-4 Ha) receive equal fractional fill.
        occ_a     = np.zeros(n_a)
        remaining = Z_a
        i = 0
        while i < n_a and remaining > 1e-10:
            j = i + 1
            while j < n_a and abs(eps_a[j] - eps_a[i]) < 1e-4:
                j += 1
            deg     = j - i
            fill    = min(remaining, 2.0 * deg)
            for k in range(i, j):
                occ_a[k] = fill / deg
            remaining -= fill
            i = j

        # Assemble per-atom density and add to global P
        Pa = sum(occ_a[k] * np.outer(Ca[:, k], Ca[:, k])
                 for k in range(n_a) if occ_a[k] > 1e-10)
        if isinstance(Pa, np.ndarray):
            for i_loc, i_glob in enumerate(gidx):
                for j_loc, j_glob in enumerate(gidx):
                    P_sad[i_glob, j_glob] += Pa[i_loc, j_loc]

    return P_sad


# ── Symmetry block helpers ─────────────────────────────────────────────────────

def _enforce_blocks(M: np.ndarray, blocks: list) -> None:
    """Zero off-block elements of M in-place (maintains block-diagonal structure)."""
    if len(blocks) <= 1:
        return
    mask = np.empty(len(M), dtype=np.intp)
    for b_idx, b in enumerate(blocks):
        mask[b] = b_idx
    for i in range(len(M)):
        for j in range(len(M)):
            if mask[i] != mask[j]:
                M[i, j] = 0.0


def _block_eigh(
    M: np.ndarray, blocks: list
) -> tuple[np.ndarray, np.ndarray]:
    """
    Block-diagonal diagonalisation of M.

    Diagonalises each symmetry block independently then returns all
    (eigenvalue, eigenvector) pairs sorted globally by eigenvalue.
    """
    N = len(M)
    pairs: list[tuple[float, np.ndarray]] = []
    for b in blocks:
        sub        = M[np.ix_(b, b)]
        ek, Ck     = np.linalg.eigh(sub)
        for k in range(len(b)):
            vec    = np.zeros(N)
            vec[b] = Ck[:, k]
            pairs.append((float(ek[k]), vec))
    pairs.sort(key=lambda x: x[0])
    eps = np.array([p[0] for p in pairs])
    C   = np.column_stack([p[1] for p in pairs])
    return eps, C


# ── Main RHF routine ──────────────────────────────────────────────────────────

def run_rhf(
    xyz_block: str,
    basis: str = "sto-3g",
    charge: int = 0,
    spin: int = 0,
    max_cycles: int = 200,
    conv_tol: float = 1e-9,
) -> dict:
    """
    Run a restricted Hartree-Fock calculation.

    Returns a dict with: converged, total_energy, homo/lumo energies, dipole,
    orbitals, n_electrons, n_basis, atoms, _bfs, _C, _mo_energies (internal).
    """
    # ── Parse geometry ────────────────────────────────────────────────────────
    try:
        atoms_ang = parse_xyz_block(xyz_block)
    except ValueError as e:
        return {"error": str(e), "converged": False}

    atoms_bohr = [
        (sym, x * ANGSTROM_TO_BOHR, y * ANGSTROM_TO_BOHR, z * ANGSTROM_TO_BOHR)
        for sym, x, y, z in atoms_ang
    ]

    # ── Nuclear charges ───────────────────────────────────────────────────────
    try:
        nuclear_charges = [ATOMIC_NUMBERS[sym] for sym, *_ in atoms_bohr]
    except KeyError as e:
        return {"error": f"Unknown element: {e}", "converged": False}

    n_electrons = sum(nuclear_charges) - charge
    if n_electrons <= 0:
        return {"error": f"Invalid charge: {n_electrons} electrons", "converged": False}
    if spin < 0 or (n_electrons - spin) % 2 != 0:
        return {"error": f"Inconsistent spin multiplicity (spin={spin})", "converged": False}
    n_occ = (n_electrons - spin) // 2

    # ── Build basis ───────────────────────────────────────────────────────────
    try:
        bfs = build_basis(atoms_bohr, basis)
    except (ValueError, RuntimeError) as e:
        return {"error": str(e), "converged": False}
    N = len(bfs)

    # ── Point group detection ─────────────────────────────────────────────────
    try:
        pg_raw   = identify_point_group(atoms_bohr)
        pg_label = format_group(pg_raw)
    except Exception:
        pg_raw   = "C1"
        pg_label = "C1"

    logger.info(
        "RHF: %d atoms, %d basis functions, %d electrons, point group %s",
        len(atoms_bohr), N, n_electrons, pg_raw,
    )

    # ── One-electron integrals ────────────────────────────────────────────────
    S, T, V = compute_one_electron(bfs, atoms_bohr, nuclear_charges)
    H_core = T + V

    # ── SALCs: symmetry-adapted orthogonal basis ──────────────────────────────
    # U: AO → SALC transform (columns = SALCs).
    # Build block-diagonal S^{-1/2} in SALC basis → combined transform Z.
    # Z replaces the standard X = S^{-1/2}; the Fock matrix in the Z-basis
    # is block-diagonal by irrep so each block can be diagonalised independently.
    U, sym_blocks = build_salc(bfs, atoms_bohr, S)
    use_sym = len(sym_blocks) > 1

    # ── ERI strategy ──────────────────────────────────────────────────────────
    # N < 100  : stored tensor + fast einsum (ERIs computed once).
    # N ≥ 100  : RI-JK via 3-centre integrals (O(N³) per cycle, no N⁴ tensor).
    #            Falls back to direct Cauchy-Schwarz build if aux basis unavailable.
    _DIRECT_THRESHOLD = 100
    N = len(bfs)
    ERI  = None
    Q_cs = None
    B_ri = None

    if N < _DIRECT_THRESHOLD:
        logger.info("Computing ERI tensor (N=%d < %d)...", N, _DIRECT_THRESHOLD)
        ERI = compute_eri(bfs, U=U, sym_blocks=sym_blocks)
    else:
        try:
            B_ri = _compute_ri_B_tensor(bfs, atoms_bohr, basis)
        except Exception as _ri_exc:
            logger.warning(
                "RI-JK init failed (%s); falling back to direct build.", _ri_exc
            )
            Q_cs = _compute_cs_norms(bfs)

    S_salc = U.T @ S @ U
    X_salc = np.zeros_like(S)
    # Linear-dependency threshold: drop overlap eigenvectors with eigenvalue < 0.10.
    # Split-valence basis sets (6-31G and similar) contain inner/outer contracted
    # shell pairs that overlap at ~0.8, yielding small S eigenvalues (0.02–0.09).
    # Keeping these with canonical S^{-1/2} amplifies them by up to 7–37×, driving
    # the SCF into unphysical electronic states.  Dropping them limits Z_max to ~3,
    # matching the condition seen in well-behaved basis sets (STO-3G, cc-pVDZ).
    _lindep_thresh = 0.10
    for b in sym_blocks:
        Sb = S_salc[np.ix_(b, b)]
        vb, wb = np.linalg.eigh(Sb)
        n_drop = int(np.sum(vb < _lindep_thresh))
        if n_drop:
            logger.info(
                "Dropping %d near-linearly-dependent function(s) in overlap block "
                "(size %d, min eigenvalue %.3e < %.2f).",
                n_drop, len(b), float(vb[0]), _lindep_thresh,
            )
        inv_sqrt = np.where(vb >= _lindep_thresh, vb ** -0.5, 0.0)
        X_salc[np.ix_(b, b)] = wb @ np.diag(inv_sqrt) @ wb.T
    # Z: combined AO → orthogonal-SALC transform
    Z = U @ X_salc

    # ── Initial density matrix (SAD guess) ────────────────────────────────────
    # SAD provides a physically motivated starting density that avoids the
    # near-linear-dependence catastrophe of the H_core guess for split-valence
    # basis sets (e.g. 6-31G), where inner/outer 2s functions have S~0.8.
    P = _sad_initial_guess(bfs, atoms_bohr, nuclear_charges)

    # ── SCF loop with DIIS + oscillation-recovery damping ────────────────────
    diis         = DIIS()
    converged    = False
    delta        = float("inf")
    _damp_cycles = 0   # Remaining cycles of 50/50 density damping after oscillation

    # Incremental Fock build state (direct-build path only).
    # G_prev / P_prev track the last full G and the density it was built from.
    # Every _INCR_REBUILD_FREQ cycles we do a full rebuild to prevent FP drift.
    _G_prev:             np.ndarray | None = None
    _P_prev:             np.ndarray | None = None
    _incr_cycle:         int               = 0
    _INCR_REBUILD_FREQ:  int               = 15

    for cycle in range(1, max_cycles + 1):
        delta_prev = delta

        # Coulomb and exchange contributions
        # Stored-tensor path  (N<100)  : full einsum over pre-stored N⁴ tensor.
        # RI-JK path          (N≥100)  : O(MN²) einsum using fitted B tensor.
        # Direct-build path   (fallback): incremental Cauchy-Schwarz build.
        if ERI is not None:
            J = np.einsum("kl,mnkl->mn", P, ERI)
            K = np.einsum("kl,mknl->mn", P, ERI)
            G = J - 0.5 * K
        elif B_ri is not None:
            J, K = _build_jk_ri(B_ri, P)
            G    = J - 0.5 * K
        else:
            _do_full = (
                _P_prev is None
                or _incr_cycle >= _INCR_REBUILD_FREQ
                or float(np.max(np.abs(P - _P_prev))) > 0.5
            )
            if _do_full:
                J, K        = _build_jk_direct(bfs, P, Q_cs)
                G           = J - 0.5 * K
                _incr_cycle = 0
            else:
                dP       = P - _P_prev
                dJ, dK   = _build_jk_direct(bfs, dP, Q_cs)
                G        = _G_prev + dJ - 0.5 * dK
                _incr_cycle += 1
            _G_prev = G
            _P_prev = P.copy()
        F_phys = H_core + G

        # DIIS error vector in the orthonormal Z-basis:
        # e_Z = Z.T (FPS − SPF) Z  (properly normalised; reduces to FPS−SPF when Z≈I)
        err_ao = F_phys @ P @ S - S @ P @ F_phys
        err = Z.T @ err_ao @ Z
        diis.push(F_phys, err)

        # Oscillation recovery: delta jumped after being near-converged → stale
        # DIIS history has grown inconsistent; reset and damp for several cycles.
        if cycle > 1 and delta_prev < 1.0 and delta_prev * 50 < delta:
            diis.reset()
            diis.push(F_phys, err)
            _damp_cycles  = 4
            _incr_cycle   = _INCR_REBUILD_FREQ  # force full rebuild next cycle

        F_diis = diis.extrapolate()
        F_for_diag = F_diis if F_diis is not None else F_phys

        # Diagonalise in orthogonal-SALC basis, block by block.
        Fz = Z.T @ F_for_diag @ Z
        if use_sym:
            eps, Cp = _block_eigh(Fz, sym_blocks)
        else:
            eps, Cp = np.linalg.eigh(Fz)
        C = Z @ Cp

        # New density matrix (Aufbau: lowest n_occ orbitals)
        P_new = 2.0 * C[:, :n_occ] @ C[:, :n_occ].T

        # Density damping: mix 50% new / 50% old to recover from oscillation without
        # corrupting the physical Fock matrices stored in DIIS history.
        if _damp_cycles > 0:
            P_new = 0.5 * P_new + 0.5 * P
            _damp_cycles -= 1

        # Convergence check: max change in density matrix
        delta = float(np.max(np.abs(P_new - P)))
        P = P_new

        if delta < conv_tol:
            converged = True
            logger.info(f"Converged in {cycle} iterations (ΔP={delta:.2e})")
            break

        if cycle % 20 == 0:
            logger.info(f"  cycle {cycle}, ΔP={delta:.2e}")

    if not converged:
        logger.warning(f"SCF did not converge after {max_cycles} cycles")

    # ── Electronic and total energy ───────────────────────────────────────────
    # Recompute the physical Fock from the final density so the energy is
    # always E = ½ Tr[P(H+F_phys)], independent of DIIS extrapolation state.
    if ERI is not None:
        J_final = np.einsum("kl,mnkl->mn", P, ERI)
        K_final = np.einsum("kl,mknl->mn", P, ERI)
    elif B_ri is not None:
        J_final, K_final = _build_jk_ri(B_ri, P)
    else:
        J_final, K_final = _build_jk_direct(bfs, P, Q_cs)
    F_phys = H_core + J_final - 0.5 * K_final
    E_elec = 0.5 * float(np.einsum("mn,mn->", P, H_core + F_phys))
    E_nuc = 0.0
    for i, (sym_i, xi, yi, zi) in enumerate(atoms_bohr):
        for j, (sym_j, xj, yj, zj) in enumerate(atoms_bohr):
            if j <= i:
                continue
            Rij = sqrt((xi - xj) ** 2 + (yi - yj) ** 2 + (zi - zj) ** 2)
            E_nuc += nuclear_charges[i] * nuclear_charges[j] / Rij
    E_total = E_elec + E_nuc

    # ── Orbital metadata ──────────────────────────────────────────────────────
    mo_energies = eps  # eigenvalues of F in orthogonal basis
    homo_idx    = n_occ - 1
    lumo_idx    = n_occ

    homo_e = float(mo_energies[homo_idx]) if homo_idx >= 0 else None
    lumo_e = float(mo_energies[lumo_idx]) if lumo_idx < N else None

    orbitals = []
    for i, e in enumerate(mo_energies):
        occ = 2.0 if i < n_occ else 0.0
        if i == homo_idx:
            label = "HOMO"
        elif i == lumo_idx:
            label = "LUMO"
        elif i < homo_idx:
            label = f"HOMO-{homo_idx - i}"
        elif i > lumo_idx:
            label = f"LUMO+{i - lumo_idx}"
        else:
            label = "core"
        orbitals.append({
            "index":         i,
            "energy_hartree":float(e),
            "energy_ev":     float(e) * HARTREE_TO_EV,
            "occupation":    occ,
            "label":         label,
        })

    # ── Dipole moment ─────────────────────────────────────────────────────────
    dip_au = np.zeros(3)
    for c in range(3):
        Dc = _dipole_matrix(bfs, c)
        # electronic (negative) contribution
        dip_au[c] = -float(np.einsum("mn,mn->", P, Dc))
    # nuclear contribution
    for (sym, x, y, z), Z in zip(atoms_bohr, nuclear_charges):
        dip_au += Z * np.array([x, y, z])
    dip_debye = dip_au * DEBYE_PER_AU
    dipole = {
        "x":     float(dip_debye[0]),
        "y":     float(dip_debye[1]),
        "z":     float(dip_debye[2]),
        "total": float(np.linalg.norm(dip_debye)),
    }

    return {
        "converged":     converged,
        "total_energy":  float(E_total),
        "n_iterations":  cycle,
        "homo_idx":      homo_idx,
        "homo_energy":   homo_e,
        "lumo_energy":   lumo_e,
        "homo_lumo_gap": (lumo_e - homo_e) if (homo_e is not None and lumo_e is not None) else None,
        "dipole":        dipole,
        "orbitals":      orbitals,
        "basis_label":   BASIS_LABELS.get(basis.lower(), basis),
        "n_electrons":   n_electrons,
        "n_basis":       N,
        "point_group":       pg_raw,
        "point_group_label": pg_label,
        "n_sym_blocks":      len(sym_blocks),
        "atoms":        [{"symbol": sym, "x": x / ANGSTROM_TO_BOHR,
                          "y": y / ANGSTROM_TO_BOHR, "z": z / ANGSTROM_TO_BOHR}
                         for sym, x, y, z in atoms_bohr],
        # Internal objects for cube generation (stripped before DB storage)
        "_bfs":         bfs,
        "_C":           C,
        "_mo_energies": mo_energies,
    }


# ── Orbital cube generation ───────────────────────────────────────────────────

def generate_cube(
    result: dict,
    orbital_idx: int,
    nx: int = 80,
    ny: int = 80,
    nz: int = 80,
) -> str:
    """Generate a Gaussian .cube file for a specific molecular orbital."""
    bfs  = result.get("_bfs")
    C    = result.get("_C")
    atoms = result.get("atoms", [])

    if bfs is None or C is None:
        raise ValueError("Calculation data not available; re-run the calculation.")
    if not result.get("converged"):
        raise ValueError("SCF did not converge — cannot generate cube.")
    if orbital_idx < 0 or orbital_idx >= C.shape[1]:
        raise ValueError(f"orbital_idx {orbital_idx} out of range.")

    # MO coefficients for this orbital
    c_mo = C[:, orbital_idx]  # (N_ao,)

    # Grid: bounding box around molecule + 4 Bohr margin
    coords_bohr = np.array([
        [a["x"] * ANGSTROM_TO_BOHR, a["y"] * ANGSTROM_TO_BOHR, a["z"] * ANGSTROM_TO_BOHR]
        for a in atoms
    ])
    margin = 4.0  # Bohr
    lo = coords_bohr.min(axis=0) - margin
    hi = coords_bohr.max(axis=0) + margin
    dx = (hi - lo) / np.array([nx - 1, ny - 1, nz - 1])

    # Evaluate all AOs on grid using numpy vectorization
    X = lo[0] + np.arange(nx) * dx[0]
    Y = lo[1] + np.arange(ny) * dx[1]
    Z = lo[2] + np.arange(nz) * dx[2]
    # shape: (nx, ny, nz, 3) grid of positions
    gx, gy, gz = np.meshgrid(X, Y, Z, indexing="ij")
    grid = np.stack([gx, gy, gz], axis=-1)  # (nx, ny, nz, 3)
    grid_flat = grid.reshape(-1, 3)          # (nx*ny*nz, 3)

    N_pts = grid_flat.shape[0]
    mo_vals = np.zeros(N_pts)

    for mu, bf in enumerate(bfs):
        dr = grid_flat - bf.center  # (N_pts, 3)
        lx, ly, lz = bf.lmn
        poly = dr[:, 0] ** lx * dr[:, 1] ** ly * dr[:, 2] ** lz
        ao_val = np.zeros(N_pts)
        r2 = np.sum(dr ** 2, axis=1)
        for alpha, d, Nk in zip(bf.exponents, bf.coefficients, bf.norms):
            ao_val += Nk * d * np.exp(-alpha * r2)
        ao_val *= poly
        mo_vals += c_mo[mu] * ao_val

    mo_grid = mo_vals.reshape(nx, ny, nz)

    # ── Write .cube format ────────────────────────────────────────────────────
    lines = []
    lines.append(f"MO {orbital_idx}  {result.get('basis_label','')}\n")
    lines.append("Generated by HF-SCF Calculator\n")
    lines.append(f"{len(atoms):5d}   {lo[0]:12.6f}   {lo[1]:12.6f}   {lo[2]:12.6f}\n")
    lines.append(f"{nx:5d}   {dx[0]:12.6f}   {'0.000000':12}   {'0.000000':12}\n")
    lines.append(f"{ny:5d}   {'0.000000':12}   {dx[1]:12.6f}   {'0.000000':12}\n")
    lines.append(f"{nz:5d}   {'0.000000':12}   {'0.000000':12}   {dx[2]:12.6f}\n")

    for a, (sym, ax, ay, az) in enumerate(
        [(a["symbol"], a["x"] * ANGSTROM_TO_BOHR,
                        a["y"] * ANGSTROM_TO_BOHR,
                        a["z"] * ANGSTROM_TO_BOHR) for a in atoms]
    ):
        Z = ATOMIC_NUMBERS.get(sym, 0)
        lines.append(f"{Z:5d}   {float(Z):12.6f}   {ax:12.6f}   {ay:12.6f}   {az:12.6f}\n")

    # Data: loop x slowest, z fastest (Gaussian convention)
    for ix in range(nx):
        for iy in range(ny):
            row = mo_grid[ix, iy, :]
            for k, v in enumerate(row):
                lines.append(f" {v:12.5e}")
                if (k + 1) % 6 == 0:
                    lines.append("\n")
            lines.append("\n")

    return "".join(lines)


def strip_internal(result: dict) -> dict:
    """Remove non-serializable internal objects before JSON response."""
    return {k: v for k, v in result.items() if not k.startswith("_")}
