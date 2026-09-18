"""Allen-Eggers closed-form ballistic entry solution.

Assumptions (Allen & Eggers, NACA Report 1381, 1958):

* exponential atmosphere ``rho = rho0 exp(-z/H)``;
* flat, non-rotating planet;
* constant drag coefficient, zero lift;
* the gravity component along the flight path is negligible compared with drag,
  so the flight-path angle ``gamma`` stays at its entry value.

With ``beta = m/(C_D A)`` the ballistic coefficient and ``K = rho0 H /
(2 beta |sin gamma|)`` the solution is

    V(z)  = V_e exp(-K exp(-z/H))
    a(z)  = rho(z) V(z)^2 / (2 beta)

whose maximum, obtained analytically by setting ``d a / d(exp(-z/H)) = 0``, is

    a_max = V_e^2 |sin gamma| / (2 e H)                  (e = Euler's number)
    z(a_max) = H ln( rho0 H / (beta |sin gamma|) )

Note that ``a_max`` is *independent of the ballistic coefficient*: a heavier or
sleeker vehicle decelerates just as hard, only lower down.  That is the classic
Allen-Eggers result and is a strong, parameter-free target for the simulator.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "allen_eggers_velocity",
    "allen_eggers_deceleration",
    "allen_eggers_peak",
]


def _k_factor(rho0, scale_height, beta, gamma_rad):
    return rho0 * scale_height / (2.0 * beta * abs(np.sin(gamma_rad)))


def allen_eggers_velocity(z, v_entry, gamma_rad, beta, rho0=1.2250, scale_height=7200.0):
    """Velocity [m/s] at geometric altitude ``z`` [m]."""
    k = _k_factor(rho0, scale_height, beta, gamma_rad)
    return v_entry * np.exp(-k * np.exp(-np.asarray(z, dtype=float) / scale_height))


def allen_eggers_deceleration(z, v_entry, gamma_rad, beta, rho0=1.2250, scale_height=7200.0):
    """Drag deceleration magnitude [m/s^2] at altitude ``z`` [m]."""
    z = np.asarray(z, dtype=float)
    rho = rho0 * np.exp(-z / scale_height)
    v = allen_eggers_velocity(z, v_entry, gamma_rad, beta, rho0, scale_height)
    return rho * v * v / (2.0 * beta)


def allen_eggers_peak(v_entry, gamma_rad, beta, rho0=1.2250, scale_height=7200.0):
    """Return ``(a_max [m/s^2], z_at_a_max [m], v_at_a_max [m/s])``."""
    a_max = v_entry**2 * abs(np.sin(gamma_rad)) / (2.0 * np.e * scale_height)
    k = _k_factor(rho0, scale_height, beta, gamma_rad)
    z_peak = scale_height * np.log(2.0 * k)
    v_peak = v_entry * np.exp(-0.5)  # V = V_e exp(-K * 1/(2K)) = V_e e^{-1/2}
    return float(a_max), float(z_peak), float(v_peak)
