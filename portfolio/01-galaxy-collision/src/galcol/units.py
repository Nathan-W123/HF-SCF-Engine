"""Unit system for the galaxy-collision simulator.

Internal ("code") units
-----------------------
    length   1 kpc
    mass     1e10 M_sun
    velocity 1 km/s

Everything else follows:

    time     1 kpc / (km/s) = 0.977793 Gyr
    G        43007.1 kpc (km/s)^2 / (1e10 M_sun)

These are the classic GADGET-2 units (minus the h factors).  Velocities come
out in km/s, which makes rotation curves directly readable, and time is
converted to Gyr only at the reporting boundary.
"""

from __future__ import annotations

import numpy as np

# Gravitational constant in code units.
# G = 6.67430e-11 m^3 kg^-1 s^-2 expressed in kpc (km/s)^2 / (1e10 Msun).
G = 43007.1

#: 1 code time unit = 1 kpc / (km/s), expressed in Gyr.
TIME_UNIT_GYR = 0.9777922216807892

#: Mass unit in solar masses.
MASS_UNIT_MSUN = 1.0e10


def t_to_gyr(t):
    """Code time -> Gyr."""
    return np.asarray(t) * TIME_UNIT_GYR


def gyr_to_t(t_gyr):
    """Gyr -> code time."""
    return np.asarray(t_gyr) / TIME_UNIT_GYR


def msun(m_code):
    """Code mass -> solar masses."""
    return np.asarray(m_code) * MASS_UNIT_MSUN


# Particle type codes used throughout the package and stored in snapshots.
PTYPE_DISK = 0
PTYPE_BULGE = 1
PTYPE_HALO = 2
PTYPE_NAMES = {PTYPE_DISK: "disk", PTYPE_BULGE: "bulge", PTYPE_HALO: "halo"}
