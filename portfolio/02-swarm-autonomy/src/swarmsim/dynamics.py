"""3-DOF coordinated-turn ("Dubins airplane with dynamics") fixed-wing model.

State (per vehicle, 7 components, ENU world frame):

    0  x      East position                 [m]
    1  y      North position                [m]
    2  z      Up position / altitude        [m]
    3  V      true airspeed                 [m/s]
    4  psi    air-relative heading (azimuth from East toward North) [rad]
    5  gamma  air-relative flight-path angle, positive climbing     [rad]
    6  phi    bank angle, positive right                            [rad]

Commanded inputs (per vehicle, 3 components):

    0  V_cmd      commanded airspeed           [m/s]
    1  phi_cmd    commanded bank angle         [rad]
    2  gamma_cmd  commanded flight-path angle  [rad]

Equations of motion (W = (W_x, W_y, W_z) is the local wind in ENU):

    x_dot     = V cos(gamma) cos(psi) + W_x
    y_dot     = V cos(gamma) sin(psi) + W_y
    z_dot     = V sin(gamma)          + W_z
    V_dot     = sat( (V_cmd - V) / tau_V,         +/- accel_max )
    psi_dot   = -g tan(phi) / V                      (coordinated turn)
    gamma_dot = sat( (gamma_cmd - gamma)/tau_gamma, +/- gamma_dot_max )
    phi_dot   = sat( (phi_cmd - phi)/tau_phi,       +/- p_max )

Assumptions / limitations of the model
--------------------------------------
* Thrust, drag and mass are not modelled explicitly; the airspeed loop is
  abstracted as a rate-limited first-order lag, which is how a closed
  autothrottle behaves over the bandwidth of interest.
* Turns are perfectly coordinated (zero sideslip), so the only lateral
  acceleration available is ``g tan(phi)``.
* The load-factor limit is enforced through the bank limit,
  ``n = 1/cos(phi) <= n_max``, which is exact for a level coordinated turn and
  slightly conservative in a climb/descent.
* The wind enters only kinematically (ground velocity = air velocity + wind);
  gust-induced aerodynamic transients are not modelled.
* Integration is fixed-step RK4 with the commands held constant over the step
  (zero-order hold), which is what a digital autopilot actually does.

The integrator is vectorised over vehicles: all arrays are shaped ``(N, 7)`` /
``(N, 3)``.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from .config import G0, VehicleParams

IX, IY, IZ, IV, IPSI, IGAM, IPHI = range(7)


def wrap_pi(a):
    """Wrap angle(s) to (-pi, pi]."""
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


def state_derivative(state: np.ndarray,
                     cmd: np.ndarray,
                     wind: np.ndarray,
                     vp: VehicleParams) -> np.ndarray:
    """Continuous-time derivative of the vehicle state.

    Parameters
    ----------
    state : (N, 7) float array
    cmd   : (N, 3) float array of (V_cmd, phi_cmd, gamma_cmd)
    wind  : (N, 3) float array of ENU wind at each vehicle [m/s]
    """
    s = np.atleast_2d(state)
    c = np.atleast_2d(cmd)
    w = np.atleast_2d(wind)

    V = s[:, IV]
    psi = s[:, IPSI]
    gam = s[:, IGAM]
    phi = s[:, IPHI]

    d = np.empty_like(s)
    cg = np.cos(gam)
    d[:, IX] = V * cg * np.cos(psi) + w[:, 0]
    d[:, IY] = V * cg * np.sin(psi) + w[:, 1]
    d[:, IZ] = V * np.sin(gam) + w[:, 2]

    d[:, IV] = np.clip((c[:, 0] - V) / vp.tau_V, -vp.accel_max, vp.accel_max)
    # Coordinated turn.  Guard V away from zero (V_min > 0 is enforced anyway).
    d[:, IPSI] = -G0 * np.tan(phi) / np.maximum(V, 1.0)
    d[:, IGAM] = np.clip((c[:, 2] - gam) / vp.tau_gamma,
                         -vp.gamma_dot_max, vp.gamma_dot_max)
    d[:, IPHI] = np.clip((c[:, 1] - phi) / vp.tau_phi, -vp.p_max, vp.p_max)
    return d


def apply_state_limits(state: np.ndarray, vp: VehicleParams) -> np.ndarray:
    """Project the state back onto the admissible envelope (in place)."""
    np.clip(state[:, IV], vp.V_min, vp.V_max, out=state[:, IV])
    np.clip(state[:, IGAM], -vp.gamma_max, vp.gamma_max, out=state[:, IGAM])
    lim = vp.phi_limit
    np.clip(state[:, IPHI], -lim, lim, out=state[:, IPHI])
    state[:, IPSI] = wrap_pi(state[:, IPSI])
    return state


def rk4_step(state: np.ndarray,
             cmd: np.ndarray,
             dt: float,
             vp: VehicleParams,
             wind_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
             wind: Optional[np.ndarray] = None,
             enforce_limits: bool = True) -> np.ndarray:
    """One fixed-step RK4 update with zero-order-hold commands.

    ``wind`` (constant over the step) takes precedence; otherwise ``wind_fn`` is
    evaluated at the stage states.  Returns a new array.
    """
    s = np.array(state, dtype=float, copy=True)
    s = np.atleast_2d(s)
    c = np.atleast_2d(np.asarray(cmd, dtype=float))

    def w_of(st):
        if wind is not None:
            return wind
        if wind_fn is not None:
            return wind_fn(st)
        return np.zeros((st.shape[0], 3))

    k1 = state_derivative(s, c, w_of(s), vp)
    k2 = state_derivative(s + 0.5 * dt * k1, c, w_of(s + 0.5 * dt * k1), vp)
    k3 = state_derivative(s + 0.5 * dt * k2, c, w_of(s + 0.5 * dt * k2), vp)
    k4 = state_derivative(s + dt * k3, c, w_of(s + dt * k3), vp)
    out = s + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    if enforce_limits:
        out = apply_state_limits(out, vp)
    else:
        out[:, IPSI] = wrap_pi(out[:, IPSI])
    return out


def air_velocity(state: np.ndarray) -> np.ndarray:
    """Air-relative velocity vector(s) in ENU, shape (N, 3)."""
    s = np.atleast_2d(state)
    V, psi, gam = s[:, IV], s[:, IPSI], s[:, IGAM]
    cg = np.cos(gam)
    return np.stack([V * cg * np.cos(psi), V * cg * np.sin(psi), V * np.sin(gam)], axis=1)


def ground_velocity(state: np.ndarray, wind: np.ndarray) -> np.ndarray:
    """Ground velocity = air-relative velocity + wind, shape (N, 3)."""
    return air_velocity(state) + np.atleast_2d(wind)


def turn_radius(V: float, phi: float) -> float:
    """Analytic steady coordinated-turn radius R = V^2 / (g tan|phi|)."""
    t = abs(np.tan(phi))
    if t < 1e-12:
        return np.inf
    return float(V ** 2 / (G0 * t))


def load_factor(phi):
    """Load factor n = 1/cos(phi) for a level coordinated turn."""
    return 1.0 / np.cos(np.asarray(phi))


def commands_from_velocity(state: np.ndarray,
                           v_air_des: np.ndarray,
                           vp: VehicleParams,
                           L1: np.ndarray,
                           dt_ctrl: float) -> np.ndarray:
    """Map a desired air-relative velocity vector to (V_cmd, phi_cmd, gamma_cmd).

    The heading channel uses the L1 nonlinear-guidance law written in its
    heading-error form,

        psi_dot_cmd = (2 V / L1) * sin(eta),

    with ``eta`` the angle between the current air-relative heading and the
    desired one.  When the desired velocity points at the L1 reference point on
    the path this is *exactly* Park/Deyst/How L1 guidance (lateral acceleration
    ``a_s = 2 V^2 sin(eta) / L1``); when the collision-avoidance layer has
    deflected the desired velocity, the same law tracks that instead.

    The commanded bank follows from the coordinated-turn relation,
    ``phi_cmd = atan(V psi_dot_cmd / g)``, and is clipped to the bank/load-factor
    envelope.  The vertical channel becomes ``gamma_cmd = asin(v_z / |v|)``.
    """
    s = np.atleast_2d(state)
    v = np.atleast_2d(np.asarray(v_air_des, dtype=float))
    n = s.shape[0]

    speed = np.linalg.norm(v, axis=1)
    speed = np.maximum(speed, 1e-6)
    V_cmd = np.clip(speed, vp.V_min, vp.V_max)

    psi_des = np.arctan2(v[:, 1], v[:, 0])
    eta = wrap_pi(psi_des - s[:, IPSI])

    V = s[:, IV]
    L1 = np.broadcast_to(np.asarray(L1, dtype=float), (n,))
    psi_dot_cmd = (2.0 * V / np.maximum(L1, 1.0)) * np.sin(eta)

    # Rate cap consistent with the achievable turn rate at the bank limit.
    psi_dot_max = G0 * np.tan(vp.phi_limit) / np.maximum(V, 1.0)
    psi_dot_cmd = np.clip(psi_dot_cmd, -psi_dot_max, psi_dot_max)

    phi_cmd = np.arctan(-V * psi_dot_cmd / G0)
    phi_cmd = np.clip(phi_cmd, -vp.phi_limit, vp.phi_limit)

    gam_cmd = np.arcsin(np.clip(v[:, 2] / speed, -1.0, 1.0))
    gam_cmd = np.clip(gam_cmd, -vp.gamma_max, vp.gamma_max)

    return np.stack([V_cmd, phi_cmd, gam_cmd], axis=1)
