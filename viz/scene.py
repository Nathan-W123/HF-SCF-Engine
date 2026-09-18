"""
Shared plumbing for the orbital animations: geometries, a cached SCF driver,
camera fitting, colour ramps and video output.
"""
from __future__ import annotations

import hashlib
import logging
import os
import pickle
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "backend"))

import render as R          # noqa: E402
from fields import Grid, ao_grid, mo_field, density_field  # noqa: E402

logger = logging.getLogger("viz")


# ── Geometries (Ångström) ─────────────────────────────────────────────────────

MOLECULES = {
    "water": (
        "O  0.000000  0.000000  0.000000\n"
        "H  0.000000  0.757197  0.586257\n"
        "H  0.000000 -0.757197  0.586257",
        "Water", "H₂O",
    ),
    "benzene": (
        "C  1.396000  0.000000  0.000000\n"
        "C  0.698000  1.209000  0.000000\n"
        "C -0.698000  1.209000  0.000000\n"
        "C -1.396000  0.000000  0.000000\n"
        "C -0.698000 -1.209000  0.000000\n"
        "C  0.698000 -1.209000  0.000000\n"
        "H  2.479000  0.000000  0.000000\n"
        "H  1.240000  2.147000  0.000000\n"
        "H -1.240000  2.147000  0.000000\n"
        "H -2.479000  0.000000  0.000000\n"
        "H -1.240000 -2.147000  0.000000\n"
        "H  1.240000 -2.147000  0.000000",
        "Benzene", "C₆H₆",
    ),
    "formaldehyde": (
        "C  0.000000  0.000000 -0.529000\n"
        "O  0.000000  0.000000  0.675000\n"
        "H  0.000000  0.937600 -1.114000\n"
        "H  0.000000 -0.937600 -1.114000",
        "Formaldehyde", "CH₂O",
    ),
    "ethylene": (
        "C  0.000000  0.000000  0.668000\n"
        "C  0.000000  0.000000 -0.668000\n"
        "H  0.000000  0.923000  1.238000\n"
        "H  0.000000 -0.923000  1.238000\n"
        "H  0.000000  0.923000 -1.238000\n"
        "H  0.000000 -0.923000 -1.238000",
        "Ethylene", "C₂H₄",
    ),
}

CACHE_DIR = os.environ.get("VIZ_CACHE", os.path.join(_HERE, ".cache"))


def calculate(name: str, basis: str, *, trace: bool = False, cache: bool = True):
    """
    Run (or reload) an RHF calculation.  With `trace`, also returns the
    per-cycle history captured through run_rhf's on_cycle hook.
    """
    from scf_engine import run_rhf

    xyz, label, formula = MOLECULES[name]
    key = hashlib.sha1(f"{name}|{basis}|{trace}|{xyz}".encode()).hexdigest()[:16]
    path = os.path.join(CACHE_DIR, f"{name}-{basis}-{key}.pkl")
    if cache and os.path.exists(path):
        with open(path, "rb") as fh:
            res, history = pickle.load(fh)
        logger.info("loaded cached SCF for %s/%s", name, basis)
        return res, history, label, formula

    history: list[dict] = []
    t0 = time.time()
    res = run_rhf(xyz, basis=basis, on_cycle=(history.append if trace else None))
    if res.get("error"):
        raise RuntimeError(res["error"])
    if not res.get("converged"):
        raise RuntimeError("SCF did not converge")
    logger.info("SCF %s/%s: E = %.6f Ha, %d cycles, %d basis fns, %.1f s",
                name, basis, res["total_energy"], res["n_iterations"],
                res["n_basis"], time.time() - t0)

    if cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump((res, history), fh)
    return res, history, label, formula


def align_phases(history, S):
    """
    Make the traced MO coefficients comparable across cycles.

    Each SCF cycle diagonalises a different Fock matrix, so the sign of every
    eigenvector is arbitrary and an orbital can flip blue/red from one cycle to
    the next for no physical reason.  Fix each column's sign against the
    previous cycle using the overlap ⟨ψ_prev|ψ_now⟩.
    """
    prev = None
    for h in history:
        C = h["C"]
        if prev is not None:
            sgn = np.sign(np.einsum("mi,mn,ni->i", prev, S, C))
            sgn[sgn == 0] = 1.0
            C = C * sgn
            h["C"] = C
        prev = C


# ── Camera ────────────────────────────────────────────────────────────────────

def atom_positions(atoms) -> np.ndarray:
    return np.array([[a["x"], a["y"], a["z"]] for a in atoms])


def fit_camera(atoms, size, *, extra_points=(), pad=1.12, min_radius=2.2):
    """
    Camera centred on the molecule, with a radius large enough that the molecule
    and every supplied surface point stay in frame at any orientation.
    """
    pos = atom_positions(atoms)
    centre = 0.5 * (pos.min(axis=0) + pos.max(axis=0))
    radius = float(np.linalg.norm(pos - centre, axis=1).max()) + 0.55
    for pts in extra_points:
        if pts is not None and len(pts):
            radius = max(radius, float(np.linalg.norm(pts - centre, axis=1).max()))
    return R.Camera(center=centre, radius=max(min_radius, radius * pad),
                    width=size, height=size)


# ── Colour ramps ──────────────────────────────────────────────────────────────

_ESP_STOPS = np.array([
    [0.00, 0.72, 0.05, 0.12],
    [0.25, 0.95, 0.33, 0.26],
    [0.50, 0.96, 0.96, 0.92],
    [0.75, 0.29, 0.55, 0.95],
    [1.00, 0.05, 0.20, 0.70],
])


def esp_ramp(t):
    """Diverging red -> white -> blue ramp; t in [0, 1], scalar or array."""
    t = np.clip(np.asarray(t, dtype=np.float64), 0.0, 1.0)
    out = np.stack([np.interp(t, _ESP_STOPS[:, 0], _ESP_STOPS[:, i + 1])
                    for i in range(3)], axis=-1)
    return out


def esp_colors(values, vmax):
    """Map electrostatic potentials (same units as vmax) to RGB."""
    return esp_ramp(0.5 * (1.0 + np.clip(np.asarray(values) / vmax, -1.0, 1.0)))


# ── Field helpers ─────────────────────────────────────────────────────────────

def orbital_field(res, grid, ao, index):
    return mo_field(ao, res["_C"], index)


def total_density(res, grid, ao):
    return density_field(ao, res["_C"], res["homo_idx"] + 1)


def surface_for(field, grid, iso, ppa, *, signed=True, seed=0):
    """Point clouds for ±iso (signed) or a single iso (unsigned)."""
    if signed:
        pos = R.SurfacePoints.from_field(field, grid, iso, points_per_A2=ppa, seed=seed)
        neg = R.SurfacePoints.from_field(field, grid, -iso, points_per_A2=ppa, seed=seed + 1)
        return pos, neg
    return R.SurfacePoints.from_field(field, grid, iso, points_per_A2=ppa, seed=seed), None


# ── Video ─────────────────────────────────────────────────────────────────────

class Video:
    """MP4 writer sized for social feeds, plus a poster frame."""

    def __init__(self, path, fps=30, poster_at=None, quiet=False):
        import imageio.v2 as imageio

        self.path = path
        self.poster_at = poster_at
        self.i = 0
        self.t0 = time.time()
        self.quiet = quiet
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.w = imageio.get_writer(
            path, fps=fps, codec="libx264", macro_block_size=1,
            ffmpeg_params=["-crf", "17", "-preset", "slow", "-pix_fmt", "yuv420p",
                           "-profile:v", "high", "-movflags", "+faststart"],
        )

    def add(self, img, total=None):
        self.w.append_data(np.asarray(img))
        if self.poster_at is not None and self.i == self.poster_at:
            img.save(os.path.splitext(self.path)[0] + ".png")
        self.i += 1
        if not self.quiet and (self.i % 10 == 0 or self.i == total):
            el = time.time() - self.t0
            msg = f"  frame {self.i}" + (f"/{total}" if total else "")
            if total:
                eta = el / self.i * (total - self.i)
                msg += f"  {el:5.0f}s elapsed, ~{eta:4.0f}s left"
            print(msg, flush=True)

    def close(self):
        self.w.close()
        mb = os.path.getsize(self.path) / 2 ** 20
        print(f"  wrote {self.path}  ({self.i} frames, {mb:.1f} MB, "
              f"{time.time() - self.t0:.0f}s)", flush=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
