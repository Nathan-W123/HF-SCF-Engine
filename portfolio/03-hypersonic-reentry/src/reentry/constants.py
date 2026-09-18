"""Physical and model constants.

Every value here is either an exact SI definition, a CODATA/defining constant, or
a constant that is part of the definition of the 1976 U.S. Standard Atmosphere
(USSA76).  Nothing in this module is fitted or tuned.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Earth / gravity
# ---------------------------------------------------------------------------

#: Earth gravitational parameter (WGS-84 / EGM-96 value), m^3 s^-2.
MU_EARTH = 3.986004418e14

#: Mean volumetric radius of the Earth, m.  (WGS-84 mean radius R1 = 6371.0088 km;
#: rounded to 6371.0 km and used consistently as *the* spherical Earth radius for
#: the trajectory dynamics.  Altitude is defined as ``r - R_EARTH``.)
R_EARTH = 6371.0e3

#: Earth sidereal rotation rate, rad/s (2*pi / 86164.0905 s).
OMEGA_EARTH = 7.292115e-5

#: Standard gravitational acceleration, m/s^2 (exact by definition).
G0 = 9.80665

# ---------------------------------------------------------------------------
# USSA76 defining constants (U.S. Standard Atmosphere, 1976)
# ---------------------------------------------------------------------------

#: Effective Earth radius used by USSA76 to convert geometric <-> geopotential
#: altitude, m.  This is *not* the same as R_EARTH: it is a defined constant of
#: the atmosphere model, chosen so that g(H) = g0 at 45.5425 deg latitude.
R0_USSA76 = 6356766.0

#: Universal gas constant used by USSA76, J/(kmol*K)  ->  expressed per mole.
RSTAR_USSA76 = 8.31432  # J/(mol*K)

#: Sea-level mean molecular weight of air, kg/mol (USSA76).
M0_USSA76 = 28.9644e-3

#: Specific gas constant of air implied by USSA76, J/(kg*K)  = R* / M0.
R_AIR = RSTAR_USSA76 / M0_USSA76  # 287.0528... J/(kg K)

#: Ratio of specific heats used for the speed of sound and the Rayleigh pitot
#: relation (USSA76 uses 1.40 exactly).
GAMMA_AIR = 1.40

#: Sea-level density of the standard atmosphere, kg/m^3 (derived, kept here for
#: convenience in correlations that normalise by it).
RHO_SL = 1.2250

# ---------------------------------------------------------------------------
# Radiation / heating
# ---------------------------------------------------------------------------

#: Stefan-Boltzmann constant, W m^-2 K^-4 (CODATA, exact under the 2019 SI).
SIGMA_SB = 5.670374419e-8

#: Sutton-Graves stagnation-point convective heating coefficient for air,
#: SI units so that qdot [W/m^2] = K_SUTTON_GRAVES * sqrt(rho/R_n) * V^3.
K_SUTTON_GRAVES = 1.7415e-4

#: Planck / Boltzmann / light constants for the blackbody colour model.
PLANCK_H = 6.62607015e-34  # J s   (exact)
BOLTZMANN_K = 1.380649e-23  # J/K   (exact)
SPEED_OF_LIGHT = 2.99792458e8  # m/s  (exact)

#: Circular orbital speed at the surface, sqrt(mu/R_E) -- used only as the
#: normalising reference velocity of the Detra-Kemp-Riddell-form correlation.
V_CIRC_SURFACE = float(np.sqrt(MU_EARTH / R_EARTH))  # 7909.8 m/s
