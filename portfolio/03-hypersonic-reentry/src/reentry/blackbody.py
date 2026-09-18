"""Planck blackbody spectrum -> linear sRGB colour.

Used by the hero render so that the reentry trail is coloured by the *actual*
radiative-equilibrium wall temperature computed from the simulated
Sutton-Graves heat flux, rather than by an invented palette.

Chain
-----
1. Planck spectral radiance ``B(lambda, T)``.
2. Project onto the CIE 1931 2-deg colour-matching functions to get ``XYZ``.
   The CMFs use the multi-lobe Gaussian analytic fit of Wyman, Sloan & Shirley
   (JCGT 2013), which reproduces the tabulated CMFs to within about 1% -- more
   than enough for a visual, and it keeps the project free of data files.
3. Convert ``XYZ -> linear sRGB`` with the standard sRGB primaries matrix.
4. Normalise each colour to unit maximum channel (the *brightness* of the trail
   is carried separately by the heat flux; this function supplies hue only).

Self-consistency is checked in ``tests/test_heating_and_colour.py``: the locus must pass
close to the D65-ish white point near 6500 K, must be red-dominant below
~3500 K and blue-dominant above ~10000 K, and the chromaticity must vary
monotonically with temperature.
"""

from __future__ import annotations

import numpy as np

from .constants import BOLTZMANN_K, PLANCK_H, SPEED_OF_LIGHT

__all__ = ["planck_radiance", "cie_xyz_of_temperature", "blackbody_rgb", "make_blackbody_lut"]

# XYZ -> linear sRGB (IEC 61966-2-1, D65).
_XYZ_TO_SRGB = np.array(
    [
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ]
)


def _gauss(x, mu, s1, s2):
    s = np.where(x < mu, s1, s2)
    t = (x - mu) / s
    return np.exp(-0.5 * t * t)


def _cie_cmf(lam_nm: np.ndarray):
    """Analytic CIE 1931 2-deg colour matching functions (Wyman et al. 2013)."""
    x = (
        1.056 * _gauss(lam_nm, 599.8, 37.9, 31.0)
        + 0.362 * _gauss(lam_nm, 442.0, 16.0, 26.7)
        - 0.065 * _gauss(lam_nm, 501.1, 20.4, 26.2)
    )
    y = 0.821 * _gauss(lam_nm, 568.8, 46.9, 40.5) + 0.286 * _gauss(lam_nm, 530.9, 16.3, 31.1)
    z = 1.217 * _gauss(lam_nm, 437.0, 11.8, 36.0) + 0.681 * _gauss(lam_nm, 459.0, 26.0, 13.8)
    return x, y, z


def planck_radiance(lam_m, temperature):
    """Planck spectral radiance ``B_lambda`` [W sr^-1 m^-3]."""
    lam = np.asarray(lam_m, dtype=float)
    t = np.asarray(temperature, dtype=float)
    a = 2.0 * PLANCK_H * SPEED_OF_LIGHT**2 / lam**5
    b = PLANCK_H * SPEED_OF_LIGHT / (lam * BOLTZMANN_K * t)
    return a / np.expm1(np.clip(b, 1e-12, 700.0))


def cie_xyz_of_temperature(temperature, n_lambda: int = 471):
    """CIE XYZ tristimulus of a blackbody at ``temperature`` [K] (unnormalised)."""
    t = np.atleast_1d(np.asarray(temperature, dtype=float))
    lam_nm = np.linspace(360.0, 830.0, n_lambda)
    dlam = lam_nm[1] - lam_nm[0]
    xb, yb, zb = _cie_cmf(lam_nm)
    b = planck_radiance(lam_nm[None, :] * 1e-9, np.maximum(t, 1.0)[:, None])
    b = b / np.max(b, axis=1, keepdims=True)  # avoid overflow in the sum
    xyz = np.stack(
        [(b * xb).sum(axis=1), (b * yb).sum(axis=1), (b * zb).sum(axis=1)], axis=-1
    ) * dlam
    return xyz


def blackbody_rgb(temperature, n_lambda: int = 471):
    """Normalised linear-sRGB hue of a blackbody at ``temperature`` [K].

    Returns an array of shape ``(..., 3)`` with values in ``[0, 1]`` whose
    maximum channel is 1.  Out-of-gamut negatives are desaturated toward white
    rather than clipped, which keeps very cool (<1200 K) colours smooth.
    """
    t = np.asarray(temperature, dtype=float)
    scalar = t.ndim == 0
    xyz = cie_xyz_of_temperature(np.atleast_1d(t), n_lambda)
    rgb = xyz @ _XYZ_TO_SRGB.T
    # Desaturate out-of-gamut colours toward white instead of hard-clipping.
    deficit = np.maximum(-rgb.min(axis=1), 0.0)[:, None]
    rgb = rgb + deficit
    rgb = np.maximum(rgb, 0.0)
    peak = np.maximum(rgb.max(axis=1, keepdims=True), 1e-30)
    rgb = rgb / peak
    if scalar:
        return rgb[0]
    return rgb.reshape(t.shape + (3,))


def make_blackbody_lut(t_min=300.0, t_max=12000.0, n=512):
    """Return ``(temperatures, rgb_lut)`` for fast per-particle colour lookup."""
    temps = np.linspace(t_min, t_max, n)
    return temps, blackbody_rgb(temps)
