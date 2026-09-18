"""Modified-Newtonian aerodynamics for a spherically-blunted cone.

Model
-----
The forebody of the vehicle is an axisymmetric *sphere-cone*: a spherical nose
cap of radius ``R_n`` tangent to a cone of half-angle ``theta_c`` that ends at
base radius ``R_b``.  Tangency occurs at polar angle ``phi_t = 90 deg -
theta_c`` measured from the stagnation point.

The local surface pressure coefficient is the **modified Newtonian** value

    Cp = Cp_max * sin^2(delta),     sin(delta) = -n_hat . V_hat_inf

where ``n_hat`` is the outward surface normal, ``V_hat_inf`` the freestream
direction in body axes, and only panels with ``n_hat . V_hat_inf < 0`` (i.e.
panels the flow can "see") contribute -- the rest lie in the Newtonian shadow
and are assigned ``Cp = 0``.

``Cp_max`` is **not** a tuned constant.  It is the stagnation-point pressure
coefficient of the actual flow:

* ``M > 1``: Rayleigh pitot formula (normal shock followed by isentropic
  compression to rest), which tends to 1.83938 as ``M -> inf`` for ``gamma =
  1.4``;
* ``M <= 1``: isentropic stagnation pressure coefficient, which tends to 1 as
  ``M -> 0`` and equals the Rayleigh value at ``M = 1`` (1.27560), so the two
  branches join continuously.

This is what produces the Mach dependence of ``C_D`` in this project: no
empirical ``C_D(M)`` curve is fitted anywhere.

Because ``Cp_max`` factors out of the surface integral, the *shape* integrals

    cA_hat(alpha) = C_A / Cp_max,   cN_hat(alpha) = C_N / Cp_max

depend only on the angle of attack.  They are evaluated once by numerical
surface integration and cached on a fine ``alpha`` grid.

Validity
--------
Newtonian impact theory is a hypersonic, strong-shock approximation: it is good
to a few percent for blunt bodies at ``M >~ 5``, degrades through the
supersonic range, and is *not* a valid model below ``M ~ 1.5``.  It also omits
skin friction and base pressure entirely, so ``C_D`` here is a forebody
pressure-drag coefficient.  All of this is restated in the README.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .constants import GAMMA_AIR

__all__ = [
    "SphereCone",
    "cp_max",
    "CP_MAX_HYPERSONIC_LIMIT",
]


def cp_max(mach: np.ndarray | float, gamma: float = GAMMA_AIR) -> np.ndarray | float:
    """Stagnation-point pressure coefficient versus Mach number.

    ``M > 1``  : Rayleigh pitot formula (normal shock + isentropic stagnation).
    ``M <= 1`` : isentropic stagnation pressure coefficient.
    Both branches give 1.275601... at ``M = 1``.
    """
    m = np.asarray(mach, dtype=float)
    m = np.maximum(m, 1.0e-6)
    g = gamma
    m2 = m * m

    # Subsonic / sonic: isentropic stagnation.
    cp_sub = (2.0 / (g * m2)) * ((1.0 + 0.5 * (g - 1.0) * m2) ** (g / (g - 1.0)) - 1.0)

    # Supersonic: Rayleigh pitot (p02/p1).
    m2s = np.maximum(m2, 1.0)
    term1 = ((g + 1.0) ** 2 * m2s / (4.0 * g * m2s - 2.0 * (g - 1.0))) ** (g / (g - 1.0))
    term2 = (1.0 - g + 2.0 * g * m2s) / (g + 1.0)
    cp_sup = (term1 * term2 - 1.0) / (0.5 * g * m2s)

    out = np.where(m > 1.0, cp_sup, cp_sub)
    return float(out) if np.ndim(mach) == 0 else out


#: Newtonian hypersonic limit of Cp_max for gamma = 1.4 (M -> infinity).
CP_MAX_HYPERSONIC_LIMIT = float(
    ((GAMMA_AIR + 1.0) ** 2 / (4.0 * GAMMA_AIR)) ** (GAMMA_AIR / (GAMMA_AIR - 1.0))
    * (2.0 * GAMMA_AIR / (GAMMA_AIR + 1.0))
    * 2.0
    / GAMMA_AIR
)


#: Cache of the (alpha-grid, C_A/Cp_max, C_N/Cp_max) surface integrals, keyed on
#: the geometry and the quadrature resolution.  The integrals are deterministic
#: functions of that key, so re-building them for an identical geometry is pure
#: waste; entry-corridor sweeps construct hundreds of identical vehicles.
_SHAPE_TABLE_CACHE: dict = {}


@dataclass
class SphereCone:
    """Spherically-blunted cone forebody with modified-Newtonian aerodynamics.

    Parameters
    ----------
    nose_radius : float
        Spherical nose radius ``R_n`` [m].
    base_radius : float
        Maximum (base) radius ``R_b`` [m].  Also sets the reference area
        ``A_ref = pi R_b^2``.
    half_angle_deg : float
        Cone half-angle ``theta_c`` measured from the symmetry axis [deg].
        ``theta_c = 0`` degenerates to a hemisphere-plus-cylinder, which is used
        as an analytic unit test (``C_D = Cp_max / 2``).
    """

    nose_radius: float = 0.65
    base_radius: float = 1.30
    half_angle_deg: float = 70.0
    n_meridional: int = 400
    n_circumferential: int = 512
    _alpha_grid: np.ndarray = field(init=False, repr=False)
    _ca_hat: np.ndarray = field(init=False, repr=False)
    _cn_hat: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.nose_radius > self.base_radius:
            raise ValueError("nose_radius must not exceed base_radius")
        theta = np.radians(self.half_angle_deg)
        phi_t = 0.5 * np.pi - theta
        r_t = self.nose_radius * np.sin(phi_t)
        if r_t > self.base_radius + 1e-12:
            raise ValueError("geometry is over-blunted: sphere exceeds base radius")
        self.theta_c = theta
        self.phi_t = phi_t
        self.r_tangency = r_t
        self.area_ref = np.pi * self.base_radius**2
        self._build_shape_table()

    # -- geometry -----------------------------------------------------------
    def _panels(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (normals [N,3], areas [N]) of the forebody surface panels."""
        npsi = self.n_circumferential
        psi = (np.arange(npsi) + 0.5) * (2.0 * np.pi / npsi)
        dpsi = 2.0 * np.pi / npsi
        cpsi, spsi = np.cos(psi), np.sin(psi)

        normals = []
        areas = []

        # --- spherical nose cap: phi in [0, phi_t] ---
        nphi = max(int(self.n_meridional * self.phi_t / (0.5 * np.pi)), 8)
        if self.phi_t > 1e-9:
            dphi = self.phi_t / nphi
            phi = (np.arange(nphi) + 0.5) * dphi
            da = self.nose_radius**2 * np.sin(phi) * dphi * dpsi  # [nphi]
            nx = -np.cos(phi)[:, None] * np.ones_like(cpsi)[None, :]
            ny = np.sin(phi)[:, None] * cpsi[None, :]
            nz = np.sin(phi)[:, None] * spsi[None, :]
            normals.append(np.stack([nx, ny, nz], axis=-1).reshape(-1, 3))
            areas.append(np.repeat(da, npsi))

        # --- conical frustum: r in [r_t, R_b] ---
        if self.base_radius - self.r_tangency > 1e-9:
            nr = self.n_meridional
            dr = (self.base_radius - self.r_tangency) / nr
            r = self.r_tangency + (np.arange(nr) + 0.5) * dr
            if self.theta_c > 1e-9:
                ds = dr / np.sin(self.theta_c)
            else:  # cylinder: integrate in x instead; contributes no axial force
                ds = dr  # degenerate, area weight is irrelevant (n_x = 0)
            da = r * ds * dpsi  # [nr]
            nx = np.full((nr, npsi), -np.sin(self.theta_c))
            ny = np.cos(self.theta_c) * np.ones(nr)[:, None] * cpsi[None, :]
            nz = np.cos(self.theta_c) * np.ones(nr)[:, None] * spsi[None, :]
            normals.append(np.stack([nx, ny, nz], axis=-1).reshape(-1, 3))
            areas.append(np.repeat(da, npsi))

        return np.concatenate(normals), np.concatenate(areas)

    # -- shape integrals ----------------------------------------------------
    def _build_shape_table(self, n_alpha: int = 181) -> None:
        key = (
            self.nose_radius, self.base_radius, self.half_angle_deg,
            self.n_meridional, self.n_circumferential, n_alpha,
        )
        cached = _SHAPE_TABLE_CACHE.get(key)
        if cached is not None:
            self._alpha_grid, self._ca_hat, self._cn_hat = cached
            return
        normals, areas = self._panels()
        alphas = np.radians(np.linspace(-90.0, 90.0, n_alpha))
        ca = np.empty(n_alpha)
        cn = np.empty(n_alpha)
        for i, a in enumerate(alphas):
            v = np.array([np.cos(a), 0.0, np.sin(a)])
            sin_delta = -(normals @ v)
            cp = np.where(sin_delta > 0.0, sin_delta**2, 0.0)  # / Cp_max
            # F/(q*Aref*Cp_max) = -(1/Aref) * sum Cp_hat * n_hat * dA
            f = -(cp * areas) @ normals / self.area_ref
            ca[i] = f[0]
            cn[i] = f[2]
        self._alpha_grid = alphas
        self._ca_hat = ca
        self._cn_hat = cn
        _SHAPE_TABLE_CACHE[key] = (alphas, ca, cn)

    # -- public API ---------------------------------------------------------
    def shape_coefficients(self, alpha_rad):
        """Return ``(C_A/Cp_max, C_N/Cp_max)`` at angle of attack [rad]."""
        a = np.asarray(alpha_rad, dtype=float)
        ca = np.interp(a, self._alpha_grid, self._ca_hat)
        cn = np.interp(a, self._alpha_grid, self._cn_hat)
        return ca, cn

    def coefficients(self, alpha_rad, mach):
        """Return ``(C_D, C_L)`` at angle of attack [rad] and Mach number.

        Drag is along the freestream, lift is perpendicular to it in the
        pitch plane.
        """
        a = np.asarray(alpha_rad, dtype=float)
        k = cp_max(mach)
        ca_hat, cn_hat = self.shape_coefficients(a)
        c_a = k * ca_hat
        c_n = k * cn_hat
        cd = c_a * np.cos(a) + c_n * np.sin(a)
        cl = c_n * np.cos(a) - c_a * np.sin(a)
        return cd, cl

    def cd(self, alpha_rad, mach):
        return self.coefficients(alpha_rad, mach)[0]

    def cl(self, alpha_rad, mach):
        return self.coefficients(alpha_rad, mach)[1]

    def lift_to_drag(self, alpha_rad, mach=25.0):
        cd, cl = self.coefficients(alpha_rad, mach)
        return cl / cd

    def describe(self) -> dict:
        cd0, cl0 = self.coefficients(0.0, 25.0)
        return {
            "nose_radius_m": self.nose_radius,
            "base_radius_m": self.base_radius,
            "cone_half_angle_deg": self.half_angle_deg,
            "reference_area_m2": float(self.area_ref),
            "tangency_polar_angle_deg": float(np.degrees(self.phi_t)),
            "cd_alpha0_mach25": float(cd0),
            "cl_alpha0_mach25": float(cl0),
            "cp_max_hypersonic_limit": CP_MAX_HYPERSONIC_LIMIT,
        }
