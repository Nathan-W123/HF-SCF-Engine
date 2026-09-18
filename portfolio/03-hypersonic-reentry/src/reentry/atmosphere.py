"""Atmosphere models.

Primary model: the **1976 U.S. Standard Atmosphere (USSA76)**, implemented from
its defining layer table rather than from a curve fit.

Structure
---------
* ``0 <= Z <= 86 km`` (geometric).  The standard defines seven geopotential
  layers with constant molecular-scale temperature gradients.  Temperature is
  piecewise linear in *geopotential* altitude ``H = r0*Z/(r0+Z)``; pressure
  follows from the hydrostatic/barometric relation integrated analytically
  inside each layer.  Only the sea-level values ``T0 = 288.15 K`` and
  ``P0 = 101325 Pa`` and the lapse-rate table are hard-coded -- every layer base
  pressure is *derived* by chaining the barometric formula upward.  The
  published USSA76 base pressures are therefore an independent check on the
  implementation (see ``validation/val_03_atmosphere_table.py``).
* ``86 km < Z <= 1000 km``.  The USSA76 *kinetic temperature* profile is used
  exactly (isothermal segment, elliptical segment, linear segment, exponential
  segment).  Pressure is obtained by numerically integrating the hydrostatic
  equation with height-varying gravity,

      d ln p / dZ = - g(Z) * M / (R* T(Z)),      g(Z) = g0 (r0/(r0+Z))^2

  **Documented simplification:** the mean molecular weight ``M`` is held at its
  sea-level value ``M0``.  The real USSA76 upper atmosphere solves coupled
  species-diffusion equations for N2/O/O2/Ar/He/H, so that ``M`` falls with
  altitude above the turbopause.  Holding ``M = M0`` makes the scale height too
  small and therefore *under*-estimates density above roughly 100 km.  The
  consequence for this project is small: for a 120 km entry the dynamic
  pressure at 100 km is ~4 orders of magnitude below its peak, and all peak
  heating/deceleration occurs below 80 km where the full standard model is used.
  This is stated again in the README under Limitations.

Secondary models: :class:`ExponentialAtmosphere` (needed for the Allen-Eggers
closed-form benchmark) and :class:`VacuumAtmosphere` (orbit-conservation test).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline

from .constants import (
    GAMMA_AIR,
    G0,
    M0_USSA76,
    R0_USSA76,
    R_AIR,
    RSTAR_USSA76,
)

__all__ = [
    "AtmosphereBase",
    "USSA76",
    "ExponentialAtmosphere",
    "VacuumAtmosphere",
    "USSA76_LAYER_TABLE",
    "USSA76_REFERENCE_TABLE",
    "geopotential_altitude",
    "geometric_altitude",
]

# ---------------------------------------------------------------------------
# USSA76 layer definition (the only hard-coded inputs below 86 km)
# ---------------------------------------------------------------------------
# (geopotential base altitude H_b [m'], molecular-scale lapse rate L_b [K/m'])
_H_B = np.array([0.0, 11.0e3, 20.0e3, 32.0e3, 47.0e3, 51.0e3, 71.0e3, 84.8520e3])
_L_B = np.array([-6.5e-3, 0.0, 1.0e-3, 2.8e-3, 0.0, -2.8e-3, -2.0e-3])
_T0 = 288.15  # K, sea-level standard temperature
_P0 = 101325.0  # Pa, sea-level standard pressure

#: g0*M0/R*  [K/m'] -- the exponent constant of the barometric formula.
_GMR = G0 * M0_USSA76 / RSTAR_USSA76


def _build_layer_bases() -> tuple[np.ndarray, np.ndarray]:
    """Derive base temperatures and base pressures by chaining upward."""
    n = len(_L_B)
    t_b = np.empty(n + 1)
    p_b = np.empty(n + 1)
    t_b[0] = _T0
    p_b[0] = _P0
    for b in range(n):
        dh = _H_B[b + 1] - _H_B[b]
        lb = _L_B[b]
        t_next = t_b[b] + lb * dh
        if lb == 0.0:
            p_next = p_b[b] * np.exp(-_GMR * dh / t_b[b])
        else:
            p_next = p_b[b] * (t_b[b] / t_next) ** (_GMR / lb)
        t_b[b + 1] = t_next
        p_b[b + 1] = p_next
    return t_b, p_b


_T_B, _P_B = _build_layer_bases()

#: Human-readable layer table (H_b [m'], L_b [K/m'], T_b [K], p_b [Pa]).
USSA76_LAYER_TABLE = tuple(
    (float(_H_B[b]), float(_L_B[b]) if b < len(_L_B) else float("nan"),
     float(_T_B[b]), float(_P_B[b]))
    for b in range(len(_H_B))
)

#: Published USSA76 values at the layer boundaries, used *only* as an external
#: reference for validation (geopotential altitude [km'], T [K], p [Pa],
#: rho [kg/m^3]).  Source: U.S. Standard Atmosphere, 1976, NOAA/NASA/USAF,
#: Table I.  These are LITERATURE values and are never used by the model.
USSA76_REFERENCE_TABLE = (
    (0.0, 288.150, 1.01325e5, 1.2250e0),
    (11.0, 216.650, 2.263206e4, 3.6391e-1),
    (20.0, 216.650, 5.474889e3, 8.8035e-2),
    (32.0, 228.650, 8.680187e2, 1.3225e-2),
    (47.0, 270.650, 1.109063e2, 1.4275e-3),
    (51.0, 270.650, 6.693887e1, 8.6160e-4),
    (71.0, 214.650, 3.956420e0, 6.4211e-5),
    (84.8520, 186.946, 3.733836e-1, 6.9578e-6),
)


def geopotential_altitude(z: np.ndarray | float) -> np.ndarray | float:
    """Geometric altitude ``z`` [m] -> geopotential altitude ``H`` [m']."""
    return R0_USSA76 * np.asarray(z, dtype=float) / (R0_USSA76 + np.asarray(z, dtype=float))


def geometric_altitude(h: np.ndarray | float) -> np.ndarray | float:
    """Geopotential altitude ``H`` [m'] -> geometric altitude ``z`` [m]."""
    h = np.asarray(h, dtype=float)
    return R0_USSA76 * h / (R0_USSA76 - h)


# ---------------------------------------------------------------------------
# USSA76 upper-atmosphere kinetic temperature (86 -- 1000 km geometric)
# ---------------------------------------------------------------------------
_Z_TOP_LOWER = 86.0e3  # m, top of the geopotential layer model
_Z_TOP = 1000.0e3  # m, top of USSA76

# Elliptical segment constants (91 -- 110 km), USSA76 eq. (27), km units.
_TC = 263.1905
_AA = -76.3232
_LITTLE_A = 19.9429  # km
# Exospheric segment constants (120 -- 1000 km), USSA76 eq. (31).
_T_INF = 1000.0
_T_120 = 360.0
_LAMBDA = 0.01875e-3  # 1/m


def _kinetic_temperature_upper(z: np.ndarray) -> np.ndarray:
    """USSA76 kinetic temperature for 86 km <= z <= 1000 km (z in metres)."""
    zk = z / 1.0e3  # km
    t = np.empty_like(zk)

    m1 = zk < 91.0
    t[m1] = 186.8673

    m2 = (zk >= 91.0) & (zk < 110.0)
    u = (zk[m2] - 91.0) / _LITTLE_A
    t[m2] = _TC + _AA * np.sqrt(np.maximum(1.0 - u * u, 0.0))

    m3 = (zk >= 110.0) & (zk < 120.0)
    t[m3] = 240.0 + 12.0e-3 * (z[m3] - 110.0e3)

    m4 = zk >= 120.0
    r0k = R0_USSA76
    xi = (z[m4] - 120.0e3) * (r0k + 120.0e3) / (r0k + z[m4])
    t[m4] = _T_INF - (_T_INF - _T_120) * np.exp(-_LAMBDA * xi)
    return t


def _build_upper_pressure_table(n: int = 45_701) -> tuple[np.ndarray, np.ndarray]:
    """Hydrostatic integration of ln p from 86 km to 1000 km (20 m steps)."""
    z = np.linspace(_Z_TOP_LOWER, _Z_TOP, n)
    t = _kinetic_temperature_upper(z)
    g = G0 * (R0_USSA76 / (R0_USSA76 + z)) ** 2
    f = -g * M0_USSA76 / (RSTAR_USSA76 * t)  # d(ln p)/dz
    dz = z[1] - z[0]
    integral = np.concatenate(([0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * dz)))
    ln_p = np.log(_P_B[-1]) + integral
    return z, ln_p


_Z_UP, _LNP_UP = _build_upper_pressure_table()
#: Cubic spline of ln p above 86 km.  A linear interpolant would be accurate in
#: p itself but would make dp/dz piecewise constant, which shows up immediately
#: in the hydrostatic-residual check of validation 3; the spline keeps the
#: derivative accurate too.
_LNP_UP_SPLINE = CubicSpline(_Z_UP, _LNP_UP)


# ---------------------------------------------------------------------------
# Model classes
# ---------------------------------------------------------------------------
class AtmosphereBase:
    """Interface shared by all atmosphere models (all altitudes geometric, m)."""

    name = "base"

    def temperature(self, z):  # pragma: no cover - abstract
        raise NotImplementedError

    def pressure(self, z):  # pragma: no cover - abstract
        raise NotImplementedError

    def density(self, z):  # pragma: no cover - abstract
        raise NotImplementedError

    def sound_speed(self, z):
        return np.sqrt(GAMMA_AIR * R_AIR * np.asarray(self.temperature(z), dtype=float))


class USSA76(AtmosphereBase):
    """The 1976 U.S. Standard Atmosphere, 0 -- 1000 km geometric altitude.

    Below 0 m the sea-level layer is extrapolated downward (the trajectory never
    goes there, but it keeps the ODE right-hand side finite).  Above 1000 km the
    profile is clamped to the 1000 km value, which is ~1e-20 kg/m^3 and
    dynamically irrelevant.

    Parameters
    ----------
    tabulated : bool
        If True (default) the model is pre-evaluated on a uniform 10 m
        geometric-altitude grid and queries are served by linear interpolation
        of ``T`` and ``ln p``.  This is purely a speed optimisation for the ODE
        right-hand side; ``tests/test_atmosphere.py`` measures the resulting
        error against the exact evaluation and asserts it stays below 1e-6
        relative.  Set to False to evaluate the closed-form model directly.
    """

    name = "USSA76"

    def __init__(self, tabulated: bool = True, table_step: float = 10.0) -> None:
        self.tabulated = bool(tabulated)
        self._table = None
        if self.tabulated:
            n = int(round(_Z_TOP / table_step)) + 1
            zt = np.linspace(0.0, _Z_TOP, n)
            # Insert the exact layer-boundary altitudes so the piecewise-linear
            # kinks in T(z) and the slope breaks in ln p(z) land on grid nodes.
            breaks = np.concatenate(
                [
                    np.asarray(geometric_altitude(_H_B)),
                    np.array([86.0e3, 91.0e3, 110.0e3, 120.0e3]),
                ]
            )
            breaks = breaks[(breaks > 0.0) & (breaks < _Z_TOP)]
            zt = np.unique(np.concatenate([zt, breaks, np.nextafter(breaks, _Z_TOP)]))
            pt, tt = self._exact(zt)
            self._table = (zt, np.log(pt), tt, table_step)

    # -- exact closed-form evaluation --------------------------------------
    def _exact(self, z):
        z = np.asarray(z, dtype=float)
        scalar = z.ndim == 0
        z = np.atleast_1d(z)
        t = np.empty_like(z)
        p = np.empty_like(z)

        lower = z <= _Z_TOP_LOWER
        if np.any(lower):
            h = geopotential_altitude(z[lower])
            b = np.clip(np.searchsorted(_H_B, h, side="right") - 1, 0, len(_L_B) - 1)
            tb = _T_B[b]
            pb = _P_B[b]
            lb = _L_B[b]
            dh = h - _H_B[b]
            tt = tb + lb * dh
            pp = np.where(
                lb == 0.0,
                pb * np.exp(-_GMR * dh / tb),
                pb * (tb / np.where(tt <= 0, tb, tt)) ** (_GMR / np.where(lb == 0.0, 1.0, lb)),
            )
            t[lower] = tt
            p[lower] = pp

        upper = ~lower
        if np.any(upper):
            zu = np.clip(z[upper], _Z_TOP_LOWER, _Z_TOP)
            t[upper] = _kinetic_temperature_upper(zu)
            p[upper] = np.exp(_LNP_UP_SPLINE(zu))

        if scalar:
            return float(p[0]), float(t[0])
        return p, t

    # -- dispatch -----------------------------------------------------------
    def _pressure_temperature(self, z):
        if self._table is None:
            return self._exact(z)
        zt, lnp, tt, _ = self._table
        za = np.asarray(z, dtype=float)
        scalar = za.ndim == 0
        zc = np.clip(za, 0.0, _Z_TOP)
        p = np.exp(np.interp(zc, zt, lnp))
        t = np.interp(zc, zt, tt)
        # Below sea level, continue the exact sea-level layer downward.
        if np.any(za < 0.0):
            p_ex, t_ex = self._exact(np.minimum(za, 0.0))
            below = za < 0.0
            p = np.where(below, p_ex, p)
            t = np.where(below, t_ex, t)
        if scalar:
            return float(p), float(t)
        return p, t

    def temperature(self, z):
        return self._pressure_temperature(z)[1]

    def pressure(self, z):
        return self._pressure_temperature(z)[0]

    def density(self, z):
        p, t = self._pressure_temperature(z)
        return p / (R_AIR * t)


@dataclass
class ExponentialAtmosphere(AtmosphereBase):
    """Isothermal exponential atmosphere ``rho = rho0 * exp(-z/H)``.

    This is the atmosphere for which the Allen-Eggers closed-form entry solution
    is exact, so it is what the analytic benchmark is run against.  The default
    parameters are the conventional low-altitude fit to USSA76.
    """

    rho0: float = 1.2250
    scale_height: float = 7200.0
    temperature_k: float = 250.0
    name: str = "exponential"

    def density(self, z):
        return self.rho0 * np.exp(-np.asarray(z, dtype=float) / self.scale_height)

    def temperature(self, z):
        return np.full_like(np.asarray(z, dtype=float), self.temperature_k)

    def pressure(self, z):
        return self.density(z) * R_AIR * self.temperature_k


@dataclass
class VacuumAtmosphere(AtmosphereBase):
    """Zero density everywhere -- used for the orbital-conservation test."""

    name: str = "vacuum"

    def density(self, z):
        return np.zeros_like(np.asarray(z, dtype=float))

    def temperature(self, z):
        return np.full_like(np.asarray(z, dtype=float), 250.0)

    def pressure(self, z):
        return np.zeros_like(np.asarray(z, dtype=float))
