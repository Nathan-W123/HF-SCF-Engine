"""Configuration dataclasses for the fixed-wing swarm simulator.

All units are SI (metres, seconds, radians) unless a name says otherwise.
The world frame is East-North-Up (ENU):

    x -> East   [m]
    y -> North  [m]
    z -> Up     [m]  (== altitude h)

Heading ``psi`` is measured from East towards North (i.e. it is the standard
mathematical azimuth of the air-relative velocity in the x-y plane), the
flight-path angle ``gamma`` is positive climbing, and the bank angle ``phi`` is
positive for a right (clockwise-from-above) turn.  With this convention the
coordinated-turn relation is

    psi_dot = -g * tan(phi) / V

(the minus sign appears because a positive/right bank turns the aircraft
*clockwise* when viewed from above, which decreases a counter-clockwise-positive
azimuth).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Tuple

import numpy as np

G0 = 9.80665  # standard gravity [m/s^2]


@dataclass(frozen=True)
class VehicleParams:
    """Limits and lags of the 3-DOF coordinated-turn fixed-wing model.

    Numbers are representative of a ~15 kg Group-2 fixed-wing UAS.
    """

    # Airspeed envelope and longitudinal acceleration authority.
    V_min: float = 18.0          # stall-limited minimum airspeed [m/s]
    V_max: float = 34.0          # maximum level airspeed [m/s]
    V_nom: float = 25.0          # nominal cruise airspeed [m/s]
    accel_max: float = 2.0       # |dV/dt| limit from thrust/drag authority [m/s^2]

    # Bank / turn authority.
    phi_max: float = np.deg2rad(45.0)    # bank-angle limit [rad]
    load_factor_max: float = 1.6         # n = 1/cos(phi) limit [-]
    p_max: float = np.deg2rad(60.0)      # roll-rate limit [rad/s]

    # Flight-path-angle (climb) authority.
    gamma_max: float = np.deg2rad(12.0)      # |gamma| limit [rad]
    gamma_dot_max: float = np.deg2rad(8.0)   # |dgamma/dt| limit [rad/s]

    # First-order lags on the commanded inputs (closed inner loops).
    tau_V: float = 2.5           # airspeed lag [s]
    tau_phi: float = 0.35        # roll lag [s]
    tau_gamma: float = 1.2       # flight-path-angle lag [s]

    @property
    def phi_limit(self) -> float:
        """Effective bank limit: the tighter of geometric and load-factor caps."""
        phi_n = float(np.arccos(1.0 / self.load_factor_max))
        return float(min(self.phi_max, phi_n))

    @property
    def turn_radius_min(self) -> float:
        """Tightest steady turn radius at nominal airspeed [m]."""
        return float(self.V_nom ** 2 / (G0 * np.tan(self.phi_limit)))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["phi_limit_deg"] = float(np.rad2deg(self.phi_limit))
        d["turn_radius_min_m"] = self.turn_radius_min
        return d


@dataclass(frozen=True)
class GuidanceParams:
    """L1 nonlinear guidance + altitude / airspeed outer loops."""

    L1_period: float = 12.0      # L1 distance = L1_period * V_ground, clipped below
    L1_min: float = 140.0        # minimum L1 distance [m]
    L1_max: float = 420.0        # maximum L1 distance [m]
    k_alt: float = 0.22          # altitude -> climb-rate gain [1/s]
    climb_rate_max: float = 4.0  # commanded |h_dot| cap [m/s]
    wp_capture_radius: float = 90.0   # waypoint switch radius [m]
    goal_radius: float = 80.0         # goal-reached radius [m]


@dataclass(frozen=True)
class AvoidanceParams:
    """3-D reciprocal-velocity-obstacle (ORCA) + CBF safety-filter settings."""

    R_min: float = 60.0          # required minimum pairwise separation [m]
    R_collision: float = 15.0    # separation below which we call it a collision [m]
    orca_radius: float = 35.0    # per-vehicle ORCA radius [m] (combined = 70 m)
    tau_horizon: float = 10.0    # ORCA time horizon [s]
    tau_escape: float = 2.5      # recovery horizon when already inside r_comb [s]
    sense_range: float = 700.0   # neighbour sensing range [m]
    max_neighbours: int = 10     # nearest-neighbour budget per vehicle
    cbf_alpha: float = 0.8       # CBF linear class-K gain [1/s]
    cbf_accel: float = 3.0       # CBF braking-bound acceleration [m/s^2]
    cbf_margin: float = 12.0     # CBF keeps h = d - (R_min + margin) >= 0
    orca_rho: float = 40.0       # base stiffness of the (soft) ORCA rows
    orca_rho_urgency: float = 60.0    # extra stiffness for close neighbours
    zone_margin: float = 45.0    # no-fly-zone barrier stand-off [m]
    zone_sense_range: float = 900.0   # no-fly-zone awareness range [m]
    qp_sweeps: int = 90          # Hildreth dual sweeps
    vertical_pref_gain: float = 0.75  # weight of the vertical axis in the ORCA metric


@dataclass(frozen=True)
class SimParams:
    dt: float = 0.05             # dynamics integration step [s]
    control_every: int = 2       # control (guidance+ORCA) runs every N dynamics steps
    log_every: int = 2           # state logging decimation
    t_max: float = 240.0         # simulation horizon [s]
    seed: int = 0


@dataclass(frozen=True)
class NoFlyZone:
    """Vertical cylinder that vehicles must stay out of."""

    x: float
    y: float
    radius: float
    z_low: float = 0.0
    z_high: float = 4000.0

    def contains(self, p: np.ndarray) -> np.ndarray:
        p = np.atleast_2d(p)
        horiz = np.hypot(p[:, 0] - self.x, p[:, 1] - self.y) < self.radius
        vert = (p[:, 2] > self.z_low) & (p[:, 2] < self.z_high)
        return horiz & vert

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DrydenParams:
    """MIL-F-8785C Dryden turbulence parameters (see swarmsim.wind)."""

    sigma_u: float = 1.5         # longitudinal gust RMS [m/s]
    sigma_v: float = 1.5         # lateral gust RMS [m/s]
    sigma_w: float = 1.0         # vertical gust RMS [m/s]
    L_u: float = 200.0           # longitudinal scale length [m]
    L_v: float = 200.0
    L_w: float = 100.0

    def scaled(self, k: float) -> "DrydenParams":
        return DrydenParams(self.sigma_u * k, self.sigma_v * k, self.sigma_w * k,
                            self.L_u, self.L_v, self.L_w)


@dataclass
class WindParams:
    """Steady wind (ENU components) plus Dryden turbulence."""

    mean: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    dryden: DrydenParams = field(default_factory=DrydenParams)
    turbulence_on: bool = False

    def to_dict(self) -> dict:
        return {"mean_enu_mps": list(self.mean),
                "turbulence_on": self.turbulence_on,
                "dryden": asdict(self.dryden)}
