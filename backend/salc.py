"""
Symmetry-Adapted Linear Combinations (SALCs) of Atomic Orbitals.

Algorithm
---------
1. Generate all molecular symmetry operations as (3×3 matrix R, atom permutation perm).
2. Build the AO representation matrix D(R) for each operation.
3. Form a random linear combination  M = Σ_R w_R D(R)  (fixed seed → reproducible).
4. Diagonalise M → eigenvectors U = SALC transformation.
   • Abelian groups (C2v, C2h, D2h …): all eigenvalues distinct → 1-D blocks (irreps).
   • Non-Abelian groups (C3v, Td, D6h…): degenerate eigenvalues → multi-D blocks (irreps).
5. Return (U, blocks) where blocks groups column indices by irrep.

Falls back to identity (no symmetry) for C1 or on any error.
"""
from __future__ import annotations

import logging
from math import factorial

import numpy as np

logger = logging.getLogger(__name__)

# Cartesian component ordering — must match build_basis in scf_engine.py
_AM_COMPS: dict[int, list[tuple]] = {
    0: [(0, 0, 0)],
    1: [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
    2: [(2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 0), (1, 0, 1), (0, 1, 1)],
    3: [(3,0,0),(0,3,0),(0,0,3),(2,1,0),(2,0,1),
        (1,2,0),(0,2,1),(1,0,2),(0,1,2),(1,1,1)],
}
_COMP_IDX = {L: {c: i for i, c in enumerate(cs)} for L, cs in _AM_COMPS.items()}


# ── Cartesian polynomial expansion ────────────────────────────────────────────

def _mono_coeffs(R: np.ndarray, lmn: tuple) -> dict:
    """
    Expand  (R[:,0]·ρ)^l (R[:,1]·ρ)^m (R[:,2]·ρ)^n  as {(px,py,pz): coeff}.

    This gives the polynomial to which the basis function φ_{lmn}(r) maps
    when the coordinate is replaced by R^T r  (i.e. when R acts on the space).
    """
    l, m, n = lmn

    def ppow(col: np.ndarray, pw: int) -> dict:
        if pw == 0:
            return {(0, 0, 0): 1.0}
        ax, ay, az = float(col[0]), float(col[1]), float(col[2])
        res: dict = {}
        for px in range(pw + 1):
            for py in range(pw - px + 1):
                pz = pw - px - py
                c = (factorial(pw)
                     / (factorial(px) * factorial(py) * factorial(pz))
                     * ax**px * ay**py * az**pz)
                if abs(c) > 1e-14:
                    res[(px, py, pz)] = res.get((px, py, pz), 0.0) + c
        return res

    def pmul(p1: dict, p2: dict) -> dict:
        res: dict = {}
        for (a1, b1, c1), v1 in p1.items():
            for (a2, b2, c2), v2 in p2.items():
                key = (a1 + a2, b1 + b2, c1 + c2)
                res[key] = res.get(key, 0.0) + v1 * v2
        return res

    return pmul(pmul(ppow(R[:, 0], l), ppow(R[:, 1], m)), ppow(R[:, 2], n))


def _am_T(R: np.ndarray, L: int) -> np.ndarray:
    """
    (n_comp × n_comp) transformation matrix for Cartesian monomials of degree L.
    T[i, j] = coefficient of output monomial i when input monomial j is transformed by R.
    """
    comps = _AM_COMPS[L]
    idx   = _COMP_IDX[L]
    n     = len(comps)
    T     = np.zeros((n, n))
    for j, lmn in enumerate(comps):
        for (l_, m_, n_), c in _mono_coeffs(R, lmn).items():
            i = idx.get((l_, m_, n_))
            if i is not None:
                T[i, j] = c
    return T


# ── AO representation matrix ───────────────────────────────────────────────────

def _build_D(bfs: list, R: np.ndarray, perm: list[int]) -> np.ndarray:
    """
    N×N AO representation matrix D(R) for symmetry operation R with atom permutation perm.
    D[ν, μ] = coefficient of basis function ν in the image of basis function μ under R.
    """
    from collections import defaultdict

    N = len(bfs)
    D = np.zeros((N, N))

    # Cache angular-momentum transformation matrices
    L_vals = {sum(bf.lmn) for bf in bfs}
    T_cache = {L: _am_T(R, L) for L in L_vals if L in _AM_COMPS}

    # Index: (target_atom, L, exponents_key) → [(component_index, global_bf_idx), ...]
    bf_map: dict = defaultdict(list)
    for i, bf in enumerate(bfs):
        L  = sum(bf.lmn)
        ci = _COMP_IDX.get(L, {}).get(bf.lmn, -1)
        key = (bf.atom_idx, L, tuple(round(x, 10) for x in bf.exponents))
        bf_map[key].append((ci, i))

    for mu, bf in enumerate(bfs):
        L = sum(bf.lmn)
        T = T_cache.get(L)
        if T is None:
            continue
        j = _COMP_IDX.get(L, {}).get(bf.lmn, -1)
        if j < 0:
            continue
        tgt = perm[bf.atom_idx]
        key = (tgt, L, tuple(round(x, 10) for x in bf.exponents))
        for ci, nu in bf_map[key]:
            if ci >= 0:
                D[nu, mu] = T[ci, j]

    return D


# ── Symmetry operation enumeration ────────────────────────────────────────────

def _op_key(R: np.ndarray, perm: list[int]) -> tuple:
    return tuple(np.round(R.ravel(), 5)) + tuple(perm)


def _atom_perm(symbols: list[str], coords: np.ndarray,
               R: np.ndarray, tol: float) -> list[int] | None:
    """Return atom permutation induced by R, or None if R is not a symmetry op."""
    n = len(symbols)
    Rc = (R @ coords.T).T
    perm = [-1] * n
    for i in range(n):
        for j in range(n):
            if symbols[i] == symbols[j] and np.linalg.norm(Rc[i] - coords[j]) < tol:
                perm[i] = j
                break
        if perm[i] < 0:
            return None
    return perm


def _collect_generators(symbols: list[str], coords: np.ndarray,
                        tol: float) -> list[tuple]:
    """Find all valid symmetry operations (generators) for the molecule."""
    from symmetry import _candidate_axes, _is_cn, _is_mirror, _has_inversion

    ops: dict = {}

    # Identity
    E = (np.eye(3), list(range(len(symbols))))
    ops[_op_key(*E)] = E

    axes = _candidate_axes(coords)

    # Proper rotations C_n^k
    for order in (2, 3, 4, 5, 6):
        for ax in axes:
            if not _is_cn(symbols, coords, ax, order, tol):
                continue
            for k in range(1, order):
                theta = 2.0 * np.pi * k / order
                u = ax / np.linalg.norm(ax)
                c, s = np.cos(theta), np.sin(theta)
                K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
                R = c * np.eye(3) + (1 - c) * np.outer(u, u) + s * K
                p = _atom_perm(symbols, coords, R, tol)
                if p:
                    k2 = _op_key(R, p)
                    if k2 not in ops:
                        ops[k2] = (R, p)

    # Mirror planes
    for ax in axes:
        n = ax / np.linalg.norm(ax)
        R = np.eye(3) - 2.0 * np.outer(n, n)
        p = _atom_perm(symbols, coords, R, tol)
        if p:
            k2 = _op_key(R, p)
            if k2 not in ops:
                ops[k2] = (R, p)

    # Inversion
    if _has_inversion(symbols, coords, tol):
        R = -np.eye(3)
        p = _atom_perm(symbols, coords, R, tol)
        if p:
            k2 = _op_key(R, p)
            if k2 not in ops:
                ops[k2] = (R, p)

    return list(ops.values())


def _close_group(generators: list, max_order: int = 240) -> list:
    """Close generators under composition to get the complete group."""
    ops = {_op_key(*g): g for g in generators}
    changed = True
    while changed and len(ops) <= max_order:
        changed = False
        cur = list(ops.values())
        for R1, p1 in cur:
            for R2, p2 in cur:
                R  = R1 @ R2
                p  = [p1[p2[i]] for i in range(len(p2))]
                k  = _op_key(R, p)
                if k not in ops:
                    ops[k] = (R, p)
                    changed = True
                    if len(ops) > max_order:
                        break
            if len(ops) > max_order:
                break
    return list(ops.values())


# ── Block detection ────────────────────────────────────────────────────────────

def _find_blocks(evals: np.ndarray, tol: float = 2e-4) -> list[np.ndarray]:
    """Group indices by eigenvalue cluster → symmetry blocks."""
    order    = np.argsort(evals)
    sorted_e = evals[order]
    blocks   = []
    start    = 0
    for i in range(1, len(sorted_e) + 1):
        if i == len(sorted_e) or sorted_e[i] - sorted_e[i - 1] > tol:
            blocks.append(order[start:i])
            start = i
    return blocks


# ── Public API ─────────────────────────────────────────────────────────────────

def build_salc(
    bfs: list,
    atoms_bohr: list[tuple],
    S_ao: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Build the SALC transformation matrix U and symmetry block structure.

    Parameters
    ----------
    bfs       : list of BasisFunction
    atoms_bohr: atom list as (symbol, x, y, z) in Bohr
    S_ao      : N×N overlap matrix in AO basis

    Returns
    -------
    U      : N×N unitary matrix; U[:,k] is SALC k in AO coefficients.
             AO → SALC:  M_salc = U.T @ M_ao @ U
    blocks : list of np.ndarray of column indices, one per symmetry block.
             All indices in a block belong to the same irrep (or degenerate set).

    On failure returns (np.eye(N), [np.arange(N)]).
    """
    N        = len(bfs)
    fallback = (np.eye(N), [np.arange(N)])

    symbols = [s for s, *_ in atoms_bohr]
    coords  = np.array([[x, y, z] for _, x, y, z in atoms_bohr])

    try:
        gens    = _collect_generators(symbols, coords, tol=0.35)
        all_ops = _close_group(gens)
        h       = len(all_ops)

        if h <= 1:
            return fallback

        logger.info("SALC: %d symmetry operations found", h)

        # Random weighted sum of all representation matrices (fixed seed)
        rng     = np.random.default_rng(42)
        weights = rng.standard_normal(h)
        M       = np.zeros((N, N))
        for w, (R, p) in zip(weights, all_ops):
            M += w * _build_D(bfs, R, p)
        M = 0.5 * (M + M.T)   # symmetrise for numerical stability

        evals, U = np.linalg.eigh(M)
        blocks   = _find_blocks(evals)

        n_blocks = len(blocks)
        sizes    = [len(b) for b in blocks]
        logger.info("SALC: %d symmetry blocks, sizes %s", n_blocks, sizes)

        # Sanity check: off-block elements of S_salc should be small
        S_salc    = U.T @ S_ao @ U
        off_block = _off_block_rms(S_salc, blocks)
        if off_block > 1e-3:
            logger.warning(
                "SALC off-block S rms = %.2e (> 1e-3); symmetry may be broken, "
                "falling back to no symmetry.", off_block
            )
            return fallback

        return U, blocks

    except Exception as exc:
        logger.warning("SALC construction failed (%s), skipping symmetry", exc)
        return fallback


def _off_block_rms(M: np.ndarray, blocks: list[np.ndarray]) -> float:
    """RMS of off-block elements of M."""
    mask = np.zeros(len(M), dtype=int)
    for b_idx, b in enumerate(blocks):
        mask[b] = b_idx
    total = 0.0
    count = 0
    for i in range(len(M)):
        for j in range(len(M)):
            if mask[i] != mask[j]:
                total += M[i, j] ** 2
                count += 1
    return float(np.sqrt(total / max(count, 1)))
