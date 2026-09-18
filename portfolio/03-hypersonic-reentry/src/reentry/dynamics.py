"""3-DOF point-mass entry dynamics over a spherical, optionally rotating Earth.

State vector (9 components)::

    y = [r, lon, lat, V, gamma, psi, Q, W, s]

with

===========  =========================================================
``r``        geocentric radius [m]  (altitude ``h = r - R_EARTH``)
``lon``      longitude [rad]
``lat``      geocentric latitude [rad]
``V``        speed relative to the (co-rotating) atmosphere [m/s]
``gamma``    flight-path angle, positive upward [rad]
``psi``      heading, measured clockwise from north [rad]
``Q``        integrated stagnation-point heat load [J/m^2]
``W``        specific work done by drag, ``int (D/m) V dt`` [J/kg]
``s``        ground-track downrange arc length [m]
===========  =========================================================

Equations of motion (Vinh/Regan form), with ``omega`` the planet rotation rate::

    dr/dt     = V sin(gamma)
    dlon/dt   = V cos(gamma) sin(psi) / (r cos(lat))
    dlat/dt   = V cos(gamma) cos(psi) / r
    dV/dt     = -D/m - (mu/r^2) sin(gamma)
                + omega^2 r cos(lat) (sin(gamma) cos(lat) - cos(gamma) sin(lat) cos(psi))
    dgamma/dt = (1/V) [ (L/m) cos(sigma) + (V^2/r - mu/r^2) cos(gamma)
                        + 2 omega V cos(lat) sin(psi)
                        + omega^2 r cos(lat) (cos(gamma) cos(lat)
                                              + sin(gamma) sin(lat) cos(psi)) ]
    dpsi/dt   = (1/V) [ (L/m) sin(sigma)/cos(gamma)
                        + (V^2/r) cos(gamma) sin(psi) tan(lat)
                        - 2 omega V (tan(gamma) cos(lat) cos(psi) - sin(lat))
                        + (omega^2 r / cos(gamma)) sin(lat) cos(lat) sin(psi) ]

Gravity is central inverse-square (``mu/r^2``); J2 and higher harmonics are not
modelled (see README limitations).  All reported results use ``omega = 0``
(non-rotating Earth); the rotating terms are implemented, exercised by
``validation/val_01_orbit_energy.py`` (which verifies that a vacuum trajectory
computed with ``omega = OMEGA_EARTH`` still conserves *inertial* energy and
angular momentum), and available via ``omega``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .atmosphere import AtmosphereBase
from .constants import MU_EARTH, R_EARTH
from .heating import heat_flux_sutton_graves
from .vehicle import Vehicle

__all__ = ["EntryModel", "state_to_cartesian", "specific_orbital_elements"]

IR, ILON, ILAT, IV, IGAM, IPSI, IQ, IW, IS = range(9)


@dataclass
class EntryModel:
    """Right-hand side of the 3-DOF entry equations of motion.

    Parameters
    ----------
    vehicle : Vehicle
    atmosphere : AtmosphereBase
    omega : float
        Planet rotation rate [rad/s]; 0 for a non-rotating Earth.
    mu, r_planet : float
        Gravitational parameter and planet radius.
    include_gravity : bool
        If False the gravity terms are dropped entirely.  Used only to reproduce
        the Allen-Eggers idealisation exactly.
    curvature : bool
        If False the ``V^2/r`` centrifugal term in ``dgamma/dt`` and the
        ``tan(lat)`` heading term in ``dpsi/dt`` are dropped.
    freeze_gamma : bool
        If True the flight-path angle is held constant.  Together with
        ``include_gravity=False`` and ``curvature=False`` this reproduces the
        Allen-Eggers flat-Earth, constant-gamma idealisation exactly.
    constant_cd : float or None
        If set, overrides the Newtonian aerodynamic model with a constant drag
        coefficient (and zero lift).  Used for the Allen-Eggers benchmark.
    """

    vehicle: Vehicle
    atmosphere: AtmosphereBase
    omega: float = 0.0
    mu: float = MU_EARTH
    r_planet: float = R_EARTH
    include_gravity: bool = True
    curvature: bool = True
    freeze_gamma: bool = False
    constant_cd: float | None = None

    # -- helpers ------------------------------------------------------------
    def altitude(self, r):
        return r - self.r_planet

    def aero_forces(self, r, v):
        """Return ``(drag_accel, lift_accel, rho, mach, q_dyn, cd, cl)``."""
        h = self.altitude(r)
        rho = self.atmosphere.density(h)
        q_dyn = 0.5 * rho * v * v
        if self.constant_cd is not None:
            cd = self.constant_cd
            cl = 0.0
            mach = v / self.atmosphere.sound_speed(h)
        else:
            mach = v / self.atmosphere.sound_speed(h)
            cd, cl = self.vehicle.aero(mach)
        f = q_dyn * self.vehicle.area_ref / self.vehicle.mass
        return f * cd, f * cl, rho, mach, q_dyn, cd, cl

    # -- ODE ----------------------------------------------------------------
    def rhs(self, t, y):
        r, lon, lat, v, gam, psi = y[IR], y[ILON], y[ILAT], y[IV], y[IGAM], y[IPSI]
        v = max(v, 1.0e-6) if np.isscalar(v) else np.maximum(v, 1.0e-6)
        sg, cg = np.sin(gam), np.cos(gam)
        sp, cp_ = np.sin(psi), np.cos(psi)
        sl, cl_ = np.sin(lat), np.cos(lat)
        cl_ = np.where(np.abs(cl_) < 1e-9, 1e-9, cl_) if not np.isscalar(cl_) else (
            1e-9 if abs(cl_) < 1e-9 else cl_
        )
        cg_safe = cg if abs(cg) > 1e-9 else np.sign(cg or 1.0) * 1e-9

        a_drag, a_lift, rho, mach, q_dyn, cd, cl_coef = self.aero_forces(r, v)
        sigma = self.vehicle.bank_rad
        w = self.omega

        g_r = self.mu / (r * r) if self.include_gravity else 0.0

        dr = v * sg
        dlon = v * cg * sp / (r * cl_)
        dlat = v * cg * cp_ / r

        dv = -a_drag - g_r * sg
        if w != 0.0:
            dv += w * w * r * cl_ * (sg * cl_ - cg * sl * cp_)

        if self.freeze_gamma:
            dgam = 0.0
        else:
            centrifugal = (v * v / r) if self.curvature else 0.0
            dgam = (a_lift * np.cos(sigma) + (centrifugal - g_r) * cg) / v
            if w != 0.0:
                dgam += (
                    2.0 * w * v * cl_ * sp
                    + w * w * r * cl_ * (cg * cl_ + sg * sl * cp_)
                ) / v

        dpsi = (a_lift * np.sin(sigma) / cg_safe) / v
        if self.curvature:
            dpsi += (v / r) * cg * sp * (sl / cl_)
        if w != 0.0:
            dpsi += (
                -2.0 * w * v * ((sg / cg_safe) * cl_ * cp_ - sl)
                + (w * w * r / cg_safe) * sl * cl_ * sp
            ) / v

        q_dot = heat_flux_sutton_graves(rho, v, self.vehicle.nose_radius)
        dW = a_drag * v
        ds = self.r_planet * v * cg / r

        out = np.empty(9, dtype=float)
        out[IR] = dr
        out[ILON] = dlon
        out[ILAT] = dlat
        out[IV] = dv
        out[IGAM] = dgam
        out[IPSI] = dpsi
        out[IQ] = q_dot
        out[IW] = dW
        out[IS] = ds
        return out


# ---------------------------------------------------------------------------
# State conversions
# ---------------------------------------------------------------------------
def state_to_cartesian(r, lon, lat, v, gam, psi, omega=0.0, t=0.0):
    """Convert the flight state to inertial Cartesian position and velocity.

    ``lon`` is the **planet-fixed** longitude, so the inertial longitude is
    ``lon + omega*t`` (the planet-fixed frame is taken to coincide with the
    inertial frame at ``t = 0``).  The relative velocity -- which is what
    ``V, gamma, psi`` describe -- is expressed in the local up/east/north triad
    and the planet-rotation term ``omega_vec x r`` is then added to obtain the
    inertial velocity.  With ``omega = 0`` the two frames coincide.
    """
    r = np.asarray(r, dtype=float)
    lon = np.asarray(lon, dtype=float) + omega * np.asarray(t, dtype=float)
    lat = np.asarray(lat, dtype=float)
    v = np.asarray(v, dtype=float)
    gam = np.asarray(gam, dtype=float)
    psi = np.asarray(psi, dtype=float)

    cl, sl = np.cos(lat), np.sin(lat)
    co, so = np.cos(lon), np.sin(lon)

    up = np.stack([cl * co, cl * so, sl], axis=-1)
    east = np.stack([-so, co, np.zeros_like(so)], axis=-1)
    north = np.stack([-sl * co, -sl * so, cl], axis=-1)

    pos = r[..., None] * up
    v_up = (v * np.sin(gam))[..., None]
    v_e = (v * np.cos(gam) * np.sin(psi))[..., None]
    v_n = (v * np.cos(gam) * np.cos(psi))[..., None]
    vel = v_up * up + v_e * east + v_n * north
    if omega != 0.0:
        w = np.array([0.0, 0.0, omega])
        vel = vel + np.cross(w, pos)
    return pos, vel


def specific_orbital_elements(pos, vel, mu=MU_EARTH):
    """Return ``(specific_energy, angular_momentum_vector, semi_major_axis)``."""
    pos = np.asarray(pos, dtype=float)
    vel = np.asarray(vel, dtype=float)
    rmag = np.linalg.norm(pos, axis=-1)
    vmag = np.linalg.norm(vel, axis=-1)
    energy = 0.5 * vmag**2 - mu / rmag
    hvec = np.cross(pos, vel)
    a = -mu / (2.0 * energy)
    return energy, hvec, a
