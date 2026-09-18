"""Stagnation-point aerothermal heating models.

Primary correlation -- **Sutton & Graves (1971)**::

    qdot_cw = k * sqrt(rho / R_n) * V^3,      k = 1.7415e-4  (SI, W/m^2)

This is the cold-wall stagnation-point *convective* heat flux for air.

Cross-check correlation -- **Detra-Kemp-Riddell form**::

    qdot_cw = C / sqrt(R_n) * (rho/rho_sl)^0.5 * (V/V_co)^3.15

with ``C = 1.1037e8 W/m^2`` (``R_n`` in m) and ``V_co = 7924.8 m/s``.  The two
correlations share the ``rho^0.5 R_n^-0.5`` scaling but differ in the velocity
exponent (3 vs 3.15) and in the leading coefficient.

.. warning::
   The DKR leading coefficient here is quoted from memory and could **not** be
   verified against the primary reference in this offline environment.  It is
   used only as a *sensitivity cross-check* on the velocity exponent and
   coefficient, never as an independently validated reference.  Sutton-Graves,
   whose coefficient is specified exactly, is the primary model for every
   reported number.  An additional, coefficient-free comparison is provided by
   :func:`heat_flux_exponent_sensitivity`, which re-anchors the 3.15 exponent to
   Sutton-Graves at ``V_co`` so the pure exponent effect is isolated.

Radiative-equilibrium wall temperature
--------------------------------------
A thin, non-ablating, radiatively cooled TPS surface in steady state balances
the incoming convective flux against its own re-radiation::

    epsilon * sigma * T_w^4 = qdot

so ``T_w = (qdot / (epsilon sigma))^(1/4)``.  This is the quantity the hero
render maps to colour through the Planck locus: the trail colour is literally
the colour of the glowing wall.
"""

from __future__ import annotations

import numpy as np

from .constants import K_SUTTON_GRAVES, RHO_SL, SIGMA_SB

__all__ = [
    "heat_flux_sutton_graves",
    "heat_flux_dkr",
    "heat_flux_exponent_sensitivity",
    "radiative_equilibrium_temperature",
    "hot_wall_factor",
    "DKR_COEFFICIENT",
    "DKR_REFERENCE_VELOCITY",
]

#: Detra-Kemp-Riddell-form leading coefficient, W/m^2 with R_n in metres.
DKR_COEFFICIENT = 1.1037e8
#: Circular-orbit reference velocity used to normalise the DKR form, m/s.
DKR_REFERENCE_VELOCITY = 7924.8
#: Specific heat of air used for the (small) hot-wall enthalpy correction.
CP_AIR = 1004.5


def heat_flux_sutton_graves(rho, velocity, nose_radius):
    """Cold-wall stagnation-point convective heat flux [W/m^2] (Sutton-Graves)."""
    rho = np.asarray(rho, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    return K_SUTTON_GRAVES * np.sqrt(np.maximum(rho, 0.0) / nose_radius) * velocity**3


def heat_flux_dkr(rho, velocity, nose_radius):
    """Cold-wall stagnation-point convective heat flux [W/m^2] (DKR form)."""
    rho = np.asarray(rho, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    return (
        DKR_COEFFICIENT
        / np.sqrt(nose_radius)
        * np.sqrt(np.maximum(rho, 0.0) / RHO_SL)
        * (velocity / DKR_REFERENCE_VELOCITY) ** 3.15
    )


def heat_flux_exponent_sensitivity(rho, velocity, nose_radius):
    """Sutton-Graves re-exponentiated to ``V^3.15``, anchored at ``V_co``.

    Equals :func:`heat_flux_sutton_graves` exactly at ``V = V_co`` and therefore
    isolates the effect of the velocity exponent with no coefficient ambiguity.
    """
    velocity = np.asarray(velocity, dtype=float)
    return heat_flux_sutton_graves(rho, velocity, nose_radius) * (
        velocity / DKR_REFERENCE_VELOCITY
    ) ** 0.15


def radiative_equilibrium_temperature(q, emissivity=0.85):
    """Radiative-equilibrium wall temperature [K] from a heat flux [W/m^2]."""
    q = np.maximum(np.asarray(q, dtype=float), 0.0)
    return (q / (emissivity * SIGMA_SB)) ** 0.25


def hot_wall_factor(velocity, wall_temperature, free_stream_temperature=250.0):
    """Hot-wall correction factor ``1 - h_w/h_0`` (dimensionless, in [0, 1]).

    ``h_0 = V^2/2 + cp*T_inf`` is the total enthalpy and ``h_w = cp*T_w`` the
    wall enthalpy, both with a calorically perfect ``cp``.  This understates the
    real-gas wall enthalpy somewhat, so the correction is a lower bound on the
    true reduction; it is typically only a few percent at entry speeds.
    """
    v = np.asarray(velocity, dtype=float)
    h0 = 0.5 * v**2 + CP_AIR * free_stream_temperature
    hw = CP_AIR * np.asarray(wall_temperature, dtype=float)
    return np.clip(1.0 - hw / np.maximum(h0, 1.0), 0.0, 1.0)
