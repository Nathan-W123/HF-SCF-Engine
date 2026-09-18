"""
Scalar fields on a Cartesian grid, evaluated from a converged RHF wavefunction.

Everything here is driven by the engine in backend/ — the basis functions, the
MO coefficients and the density matrix all come straight out of run_rhf().  The
only thing added on top is grid evaluation:

    AO amplitudes   φ_μ(r)          -> ao_grid()
    MO amplitudes   ψ_k(r)          -> mo_field()
    electron density ρ(r)           -> density_field()
    electrostatic potential V(r)    -> esp_field()   (FFT Poisson solve)

Lengths are in Ångström at the interface and Bohr internally, matching
scf_engine's convention.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))

from scf_engine import ANGSTROM_TO_BOHR, ATOMIC_NUMBERS  # noqa: E402

BOHR_TO_ANGSTROM = 1.0 / ANGSTROM_TO_BOHR
HARTREE_TO_KCAL = 627.509474


# ── Grid ──────────────────────────────────────────────────────────────────────

@dataclass
class Grid:
    """Uniform Cartesian grid in Bohr."""
    origin:  np.ndarray   # (3,) lower corner
    spacing: np.ndarray   # (3,) step
    shape:   tuple        # (nx, ny, nz)

    @classmethod
    def around(cls, atoms, margin_ang: float = 3.2, spacing_ang: float = 0.11) -> "Grid":
        """Bounding box of `atoms` (scf_engine dicts, Ångström) plus a margin."""
        coords = np.array([[a["x"], a["y"], a["z"]] for a in atoms]) * ANGSTROM_TO_BOHR
        margin = margin_ang * ANGSTROM_TO_BOHR
        h = spacing_ang * ANGSTROM_TO_BOHR
        lo = coords.min(axis=0) - margin
        hi = coords.max(axis=0) + margin
        shape = tuple(int(np.ceil((hi[i] - lo[i]) / h)) + 1 for i in range(3))
        return cls(origin=lo, spacing=np.array([h, h, h]), shape=shape)

    @property
    def n_points(self) -> int:
        return int(np.prod(self.shape))

    @property
    def spacing_ang(self) -> np.ndarray:
        return self.spacing * BOHR_TO_ANGSTROM

    @property
    def origin_ang(self) -> np.ndarray:
        return self.origin * BOHR_TO_ANGSTROM

    def axes(self):
        return [self.origin[i] + np.arange(self.shape[i]) * self.spacing[i] for i in range(3)]

    def flat_coords(self) -> np.ndarray:
        """(n_points, 3) positions in Bohr, C-ordered to match shape."""
        X, Y, Z = np.meshgrid(*self.axes(), indexing="ij")
        return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)


# ── Orbital / density evaluation ──────────────────────────────────────────────

def ao_grid(bfs, grid: Grid, dtype=np.float32) -> np.ndarray:
    """
    Evaluate every atomic orbital on the grid: returns (n_ao, nx, ny, nz).

    Each contracted Cartesian Gaussian is
        φ(r) = (x-Ax)^l (y-Ay)^m (z-Az)^n Σ_k N_k d_k exp(-α_k |r-A|²)
    with the same normalisation constants the integral code uses, so the
    amplitudes are directly comparable with the MO coefficients.
    """
    pts = grid.flat_coords()
    out = np.empty((len(bfs),) + grid.shape, dtype=dtype)

    for mu, bf in enumerate(bfs):
        dr = pts - bf.center
        r2 = np.einsum("ij,ij->i", dr, dr)
        radial = np.zeros(r2.shape)
        for alpha, d, Nk in zip(bf.exponents, bf.coefficients, bf.norms):
            radial += (Nk * d) * np.exp(-alpha * r2)
        lx, ly, lz = bf.lmn
        if lx or ly or lz:
            radial *= dr[:, 0] ** lx * dr[:, 1] ** ly * dr[:, 2] ** lz
        out[mu] = radial.reshape(grid.shape).astype(dtype, copy=False)

    return out


def mo_field(ao: np.ndarray, C: np.ndarray, k: int) -> np.ndarray:
    """Amplitude ψ_k(r) of molecular orbital k."""
    return np.tensordot(C[:, k].astype(ao.dtype), ao, axes=(0, 0))


def density_field(ao: np.ndarray, C: np.ndarray, n_occ: int) -> np.ndarray:
    """Electron density ρ(r) = 2 Σ_i^occ |ψ_i(r)|² in electrons / Bohr³."""
    rho = np.zeros(ao.shape[1:], dtype=ao.dtype)
    for i in range(n_occ):
        psi = mo_field(ao, C, i)
        rho += psi * psi
    return 2.0 * rho


def n_electrons_on_grid(rho: np.ndarray, grid: Grid) -> float:
    """Integrate ρ — a direct check that the grid resolves the density."""
    return float(rho.sum()) * float(np.prod(grid.spacing))


# ── Electrostatic potential ───────────────────────────────────────────────────

def esp_field(rho: np.ndarray, grid: Grid, atoms) -> np.ndarray:
    """
    Molecular electrostatic potential in Hartree/e:

        V(r) = Σ_A Z_A/|r - R_A|  -  ∫ ρ(r')/|r - r'| dr'

    The electronic term is the Poisson solution for the density that is
    already on the grid, evaluated as a convolution with the 1/r kernel.  The
    grid is zero-padded to twice its size in each direction so the FFT gives
    the free-space (non-periodic) result, and the singular cell of the kernel
    is replaced by the mean of 1/r over one voxel.

    Sampled on the ρ ≈ 0.002 a.u. isosurface this is the standard MEP map;
    values on the nuclei themselves are not meaningful at finite grid spacing.
    """
    nx, ny, nz = grid.shape
    hx, hy, hz = grid.spacing
    dV = float(hx * hy * hz)

    px, py, pz = 2 * nx, 2 * ny, 2 * nz

    # 1/r kernel on the padded grid, wrapped so index 0 is the origin.
    def wrapped(n, h, p):
        i = np.arange(p)
        return np.where(i <= p // 2, i, i - p) * h

    KX = wrapped(nx, hx, px)[:, None, None]
    KY = wrapped(ny, hy, py)[None, :, None]
    KZ = wrapped(nz, hz, pz)[None, None, :]
    R = np.sqrt(KX ** 2 + KY ** 2 + KZ ** 2)
    with np.errstate(divide="ignore"):
        kernel = np.where(R > 0, 1.0 / np.where(R > 0, R, 1.0), 0.0)
    # Self term: ∫_voxel dr/r ≈ 2.38 h² for a cube of side h, i.e. 2.38/h per unit volume.
    kernel[0, 0, 0] = 2.38 / np.cbrt(dV)

    rho_pad = np.zeros((px, py, pz))
    rho_pad[:nx, :ny, :nz] = rho

    v_elec = np.fft.irfftn(
        np.fft.rfftn(rho_pad) * np.fft.rfftn(kernel), s=(px, py, pz)
    )[:nx, :ny, :nz] * dV

    # Nuclear term, analytic.
    X, Y, Z = np.meshgrid(*grid.axes(), indexing="ij")
    v_nuc = np.zeros(grid.shape)
    for a in atoms:
        Za = ATOMIC_NUMBERS[a["symbol"]]
        c = np.array([a["x"], a["y"], a["z"]]) * ANGSTROM_TO_BOHR
        d = np.sqrt((X - c[0]) ** 2 + (Y - c[1]) ** 2 + (Z - c[2]) ** 2)
        np.maximum(d, 0.35 * float(np.cbrt(dV)), out=d)   # soften at the nucleus
        v_nuc += Za / d

    return (v_nuc - v_elec).astype(np.float32)


def sample_field(field: np.ndarray, grid: Grid, points_bohr: np.ndarray) -> np.ndarray:
    """Trilinear sample of `field` at arbitrary points (n, 3) in Bohr."""
    from scipy.ndimage import map_coordinates

    idx = ((points_bohr - grid.origin) / grid.spacing).T
    return map_coordinates(field, idx, order=1, mode="nearest")
