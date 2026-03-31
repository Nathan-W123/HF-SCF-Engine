"""
Molecular point group detection (Schoenflies notation).

Algorithm (standard decision tree):
  1. Translate geometry to centroid
  2. Check for linearity
  3. Find all valid Cn rotation axes (n = 2..6)
  4. Detect cubic groups (T, O, I families) via axis counts
  5. Find principal axis (highest-order Cn)
  6. Count perpendicular C2 axes, mirror planes (σh, σv/σd)
  7. Classify into Schoenflies symbol

Returns strings like "C2v", "D6h", "Td", "Oh", "C1", "Cinfv", etc.
"""
from __future__ import annotations

import logging
import numpy as np

logger = logging.getLogger(__name__)

# Tolerance in Bohr for geometry matching.
# Generous enough for user-supplied geometries that may be slightly off.
_DEFAULT_TOL = 0.35


# ── Public API ────────────────────────────────────────────────────────────────

def identify_point_group(
    atoms_bohr: list[tuple[str, float, float, float]],
    tol: float = _DEFAULT_TOL,
) -> str:
    """Return the Schoenflies point group symbol for the molecule."""
    symbols = [s for s, *_ in atoms_bohr]
    coords = np.array([[x, y, z] for _, x, y, z in atoms_bohr], dtype=float)
    n = len(symbols)

    if n == 1:
        return "Kh"

    # Center at centroid of nuclear positions
    coords = coords - coords.mean(axis=0)

    if _is_linear(coords, tol):
        return "Dinfh" if _has_inversion(symbols, coords, tol) else "Cinfv"

    axes = _candidate_axes(coords)
    has_i = _has_inversion(symbols, coords, tol)

    # ── Find all proper rotation axes ────────────────────────────────────────
    cn: dict[int, list[np.ndarray]] = {}
    for order in (6, 5, 4, 3, 2):
        for ax in axes:
            if _is_cn(symbols, coords, ax, order, tol):
                bucket = cn.setdefault(order, [])
                if not any(abs(np.dot(ax, a)) > 1.0 - 1e-3 for a in bucket):
                    bucket.append(np.asarray(ax, dtype=float))

    n3 = len(cn.get(3, []))
    n4 = len(cn.get(4, []))
    n5 = len(cn.get(5, []))

    # ── Cubic groups ─────────────────────────────────────────────────────────
    if n5 >= 6:
        return "Ih" if has_i else "I"

    if n4 >= 3:
        return "Oh" if has_i else "O"

    if n3 >= 4:
        n_planes = _count_planes(symbols, coords, axes, principal_ax=None, tol=tol)
        if has_i:
            return "Th"
        return "Td" if n_planes >= 6 else "T"

    # ── Principal axis ───────────────────────────────────────────────────────
    if cn:
        pn = max(cn)
        p_ax = cn[pn][0]
    else:
        pn = 1
        # Fallback: pick the axis of largest variance (first PCA component)
        _, evecs = np.linalg.eigh(coords.T @ coords)
        p_ax = evecs[:, -1]

    # ── Perpendicular C2, mirror planes ──────────────────────────────────────
    n_perp_c2 = _count_perp_cn(symbols, coords, p_ax, 2, axes, tol) if pn >= 2 else 0
    sigma_h    = _has_sigma_h(symbols, coords, p_ax, tol)
    n_sigma_v  = _count_planes(symbols, coords, axes, p_ax, tol)

    if pn >= 2 and n_perp_c2 >= pn:
        # Dn family
        if sigma_h:
            return f"D{pn}h"
        if n_sigma_v >= pn:
            return f"D{pn}d"
        return f"D{pn}"

    if pn >= 2:
        # Cn family
        if sigma_h:
            return f"C{pn}h"
        if n_sigma_v > 0:
            return f"C{pn}v"
        # Check S2n improper axis
        if pn == 2 and has_i:
            return "Ci"   # shouldn't reach here normally (Ci caught below)
        return f"C{pn}"

    # ── No proper rotation (pn == 1) ─────────────────────────────────────────
    if has_i:
        return "Ci"
    total_planes = _count_planes(symbols, coords, axes, None, tol)
    if total_planes > 0:
        return "Cs"
    return "C1"


def format_group(pg: str) -> str:
    """
    Convert internal symbol (e.g. 'C2v', 'D6h', 'Td') to a
    readable Unicode string (e.g. 'C₂ᵥ', 'D₆ₕ', 'Tᴅ').
    """
    _sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    _sup_map = {"h": "ₕ", "v": "ᵥ", "d": "ᴅ", "i": "ᵢ"}

    if pg in ("C1", "Ci", "Cs", "Kh", "I", "O", "T"):
        return pg
    if pg in ("Cinfv",):
        return "C∞ᵥ"
    if pg in ("Dinfh",):
        return "D∞ₕ"

    # Split trailing suffix letter (h, v, d, s)
    if len(pg) > 1 and pg[-1].isalpha():
        base, suffix = pg[:-1], pg[-1]
        return base.translate(_sub) + _sup_map.get(suffix, suffix)
    return pg.translate(_sub)


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _is_linear(coords: np.ndarray, tol: float) -> bool:
    """True if all atoms lie along a single line."""
    if len(coords) <= 2:
        return True
    # SVD: linear ↔ second singular value near zero
    c = coords - coords.mean(axis=0)
    sv = np.linalg.svd(c, compute_uv=False)
    return float(sv[1]) < tol


def _has_inversion(symbols: list[str], coords: np.ndarray, tol: float) -> bool:
    for s, c in zip(symbols, coords):
        if not any(s2 == s and np.linalg.norm(c + c2) < tol
                   for s2, c2 in zip(symbols, coords)):
            return False
    return True


def _configs_match(
    symbols: list[str],
    ref: np.ndarray,
    transformed: np.ndarray,
    tol: float,
) -> bool:
    """Check if 'transformed' is a valid permutation of 'ref' (by element label)."""
    used = [False] * len(symbols)
    for s, c in zip(symbols, ref):
        found = False
        for j, (s2, c2) in enumerate(zip(symbols, transformed)):
            if not used[j] and s == s2 and np.linalg.norm(c - c2) < tol:
                used[j] = True
                found = True
                break
        if not found:
            return False
    return True


# ── Symmetry operation checks ─────────────────────────────────────────────────

def _rotate(coords: np.ndarray, ax: np.ndarray, n: int) -> np.ndarray:
    """Rodrigues rotation of coords by 2π/n about ax."""
    theta = 2.0 * np.pi / n
    u = ax / np.linalg.norm(ax)
    c, s = np.cos(theta), np.sin(theta)
    K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    R = c * np.eye(3) + (1.0 - c) * np.outer(u, u) + s * K
    return coords @ R.T


def _reflect(coords: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Reflect coords through the plane whose normal is 'normal'."""
    n = normal / np.linalg.norm(normal)
    return coords - 2.0 * np.outer(coords @ n, n)


def _is_cn(
    symbols: list[str], coords: np.ndarray, ax: np.ndarray, n: int, tol: float
) -> bool:
    return _configs_match(symbols, coords, _rotate(coords, ax, n), tol)


def _is_mirror(
    symbols: list[str], coords: np.ndarray, normal: np.ndarray, tol: float
) -> bool:
    return _configs_match(symbols, coords, _reflect(coords, normal), tol)


# ── Axis / plane counting helpers ─────────────────────────────────────────────

def _candidate_axes(coords: np.ndarray) -> list[np.ndarray]:
    """
    Generate a list of candidate unit vectors for symmetry axes / plane normals.
    Covers: atom-center vectors, midpoints, cross-products, PCA eigenvectors.
    """
    raw: list[np.ndarray] = []
    n = len(coords)

    # Vectors from center to each atom
    for c in coords:
        norm = float(np.linalg.norm(c))
        if norm > 1e-3:
            raw.append(c / norm)

    # Pairwise combinations
    for i in range(n):
        for j in range(i + 1, n):
            for v in (
                coords[i] + coords[j],
                coords[i] - coords[j],
                np.cross(coords[i], coords[j]),
            ):
                norm = float(np.linalg.norm(v))
                if norm > 1e-3:
                    raw.append(v / norm)

    # PCA eigenvectors (moment-of-inertia axes)
    _, evecs = np.linalg.eigh(coords.T @ coords)
    for col in range(3):
        ev = evecs[:, col]
        raw.append(ev / np.linalg.norm(ev))

    # Standard Cartesian axes
    for std in np.eye(3):
        raw.append(std.copy())

    # Deduplicate (antipodal = same for Cn purposes)
    unique: list[np.ndarray] = []
    for a in raw:
        if not any(abs(float(np.dot(a, u))) > 1.0 - 1e-3 for u in unique):
            unique.append(a)
    return unique


def _count_perp_cn(
    symbols: list[str],
    coords: np.ndarray,
    principal_ax: np.ndarray,
    n: int,
    axes: list[np.ndarray],
    tol: float,
) -> int:
    """Count Cn axes perpendicular to principal_ax."""
    count = 0
    pa = principal_ax / np.linalg.norm(principal_ax)
    checked: list[np.ndarray] = []
    for ax in axes:
        if abs(float(np.dot(ax, pa))) > 0.15:   # not perpendicular
            continue
        if any(abs(float(np.dot(ax, ch))) > 1.0 - 1e-3 for ch in checked):
            continue
        checked.append(ax)
        if _is_cn(symbols, coords, ax, n, tol):
            count += 1
    return count


def _has_sigma_h(
    symbols: list[str], coords: np.ndarray, principal_ax: np.ndarray, tol: float
) -> bool:
    """Check for a mirror plane perpendicular to the principal axis."""
    return _is_mirror(symbols, coords, principal_ax, tol)


def _count_planes(
    symbols: list[str],
    coords: np.ndarray,
    axes: list[np.ndarray],
    principal_ax: np.ndarray | None,
    tol: float,
) -> int:
    """
    Count mirror planes.
    If principal_ax given: only count planes *containing* the principal axis
    (i.e., with normal perpendicular to it) — these are σv / σd planes.
    """
    count = 0
    checked: list[np.ndarray] = []
    pa: np.ndarray | None = (
        principal_ax / np.linalg.norm(principal_ax) if principal_ax is not None else None
    )
    for ax in axes:
        # For σv, the plane normal must be ⊥ to the principal axis
        if pa is not None and abs(float(np.dot(ax, pa))) > 0.15:
            continue
        if any(abs(float(np.dot(ax, ch))) > 1.0 - 1e-3 for ch in checked):
            continue
        checked.append(ax)
        if _is_mirror(symbols, coords, ax, tol):
            count += 1
    return count
