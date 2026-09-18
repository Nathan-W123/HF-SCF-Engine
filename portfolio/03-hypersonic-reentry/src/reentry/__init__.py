"""Hypersonic atmospheric reentry simulator.

A physically grounded 3-DOF entry simulator: 1976 U.S. Standard Atmosphere,
modified-Newtonian sphere-cone aerodynamics, Sutton-Graves stagnation-point
heating, inverse-square gravity over a spherical Earth, DOP853 integration with
terminal events, and a validation suite against closed-form solutions.
"""

from __future__ import annotations

__version__ = "1.0.0"

from .aerodynamics import SphereCone, cp_max
from .analytic import allen_eggers_deceleration, allen_eggers_peak, allen_eggers_velocity
from .atmosphere import ExponentialAtmosphere, USSA76, VacuumAtmosphere
from .blackbody import blackbody_rgb, make_blackbody_lut
from .dynamics import EntryModel, specific_orbital_elements, state_to_cartesian
from .heating import (
    heat_flux_dkr,
    heat_flux_sutton_graves,
    radiative_equilibrium_temperature,
)
from .trajectory import TrajectoryResult, simulate
from .vehicle import Vehicle

__all__ = [
    "__version__",
    "SphereCone",
    "cp_max",
    "USSA76",
    "ExponentialAtmosphere",
    "VacuumAtmosphere",
    "Vehicle",
    "EntryModel",
    "simulate",
    "TrajectoryResult",
    "state_to_cartesian",
    "specific_orbital_elements",
    "heat_flux_sutton_graves",
    "heat_flux_dkr",
    "radiative_equilibrium_temperature",
    "allen_eggers_velocity",
    "allen_eggers_deceleration",
    "allen_eggers_peak",
    "blackbody_rgb",
    "make_blackbody_lut",
]
