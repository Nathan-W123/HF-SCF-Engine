"""High-level trajectory driver and post-processing."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .atmosphere import AtmosphereBase, USSA76
from .constants import G0, MU_EARTH, R_EARTH
from .dynamics import EntryModel, state_to_cartesian
from .heating import (
    heat_flux_dkr,
    heat_flux_exponent_sensitivity,
    heat_flux_sutton_graves,
    hot_wall_factor,
    radiative_equilibrium_temperature,
)
from .integrators import integrate_dop853
from .vehicle import Vehicle

__all__ = ["TrajectoryResult", "simulate"]


@dataclass
class TrajectoryResult:
    """Post-processed trajectory time histories (all arrays share ``t``)."""

    t: np.ndarray
    altitude: np.ndarray
    radius: np.ndarray
    velocity: np.ndarray
    gamma: np.ndarray
    psi: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    downrange: np.ndarray
    density: np.ndarray
    mach: np.ndarray
    q_dyn: np.ndarray
    q_dot: np.ndarray
    q_dot_dkr: np.ndarray
    q_dot_exp315: np.ndarray
    heat_load: np.ndarray
    g_load: np.ndarray
    wall_temperature: np.ndarray
    drag_work: np.ndarray
    cd: np.ndarray
    cl: np.ndarray
    termination: str = ""
    meta: dict = field(default_factory=dict)

    # -- convenience --------------------------------------------------------
    @property
    def peak_heat_flux(self) -> float:
        return float(np.max(self.q_dot))

    @property
    def peak_g(self) -> float:
        return float(np.max(self.g_load))

    @property
    def total_heat_load(self) -> float:
        return float(self.heat_load[-1])

    def at_peak_heating(self) -> int:
        return int(np.argmax(self.q_dot))

    def at_peak_g(self) -> int:
        return int(np.argmax(self.g_load))

    def cartesian(self, omega: float = 0.0):
        """Inertial Cartesian position and velocity histories, shape (N, 3)."""
        return state_to_cartesian(
            self.radius, self.lon, self.lat, self.velocity, self.gamma, self.psi,
            omega, self.t,
        )

    def summary(self) -> dict:
        ih = self.at_peak_heating()
        ig = self.at_peak_g()
        return {
            "termination": self.termination,
            "duration_s": float(self.t[-1]),
            "entry_altitude_km": float(self.altitude[0] / 1e3),
            "entry_velocity_m_s": float(self.velocity[0]),
            "entry_flight_path_angle_deg": float(np.degrees(self.gamma[0])),
            "final_altitude_km": float(self.altitude[-1] / 1e3),
            "final_velocity_m_s": float(self.velocity[-1]),
            "final_mach": float(self.mach[-1]),
            "downrange_km": float(self.downrange[-1] / 1e3),
            "peak_heat_flux_W_cm2": float(self.q_dot[ih] / 1e4),
            "peak_heat_flux_time_s": float(self.t[ih]),
            "peak_heat_flux_altitude_km": float(self.altitude[ih] / 1e3),
            "peak_heat_flux_velocity_m_s": float(self.velocity[ih]),
            "peak_wall_temperature_K": float(np.max(self.wall_temperature)),
            "total_heat_load_J_cm2": float(self.heat_load[-1] / 1e4),
            "total_heat_load_dkr_J_cm2": float(
                np.trapezoid(self.q_dot_dkr, self.t) / 1e4
            ),
            "total_heat_load_exp315_J_cm2": float(
                np.trapezoid(self.q_dot_exp315, self.t) / 1e4
            ),
            "peak_g_load": float(self.g_load[ig]),
            "peak_g_time_s": float(self.t[ig]),
            "peak_g_altitude_km": float(self.altitude[ig] / 1e3),
            "peak_dynamic_pressure_kPa": float(np.max(self.q_dyn) / 1e3),
            "peak_dynamic_pressure_altitude_km": float(
                self.altitude[int(np.argmax(self.q_dyn))] / 1e3
            ),
            "peak_mach": float(np.max(self.mach)),
        }


def _postprocess(model: EntryModel, t: np.ndarray, y: np.ndarray, termination: str,
                 meta: dict) -> TrajectoryResult:
    r, lon, lat, v, gam, psi, q_int, w_int, s = (y[i] for i in range(9))
    h = model.altitude(r)
    rho = np.asarray(model.atmosphere.density(h), dtype=float)
    a_snd = np.asarray(model.atmosphere.sound_speed(h), dtype=float)
    mach = v / a_snd
    q_dyn = 0.5 * rho * v * v
    if model.constant_cd is not None:
        cd = np.full_like(v, float(model.constant_cd))
        cl = np.zeros_like(v)
    else:
        cd, cl = model.vehicle.aero(mach)
        cd = np.broadcast_to(np.asarray(cd, float), v.shape).copy()
        cl = np.broadcast_to(np.asarray(cl, float), v.shape).copy()
    a_drag = q_dyn * model.vehicle.area_ref * cd / model.vehicle.mass
    a_lift = q_dyn * model.vehicle.area_ref * cl / model.vehicle.mass
    g_load = np.hypot(a_drag, a_lift) / G0

    rn = model.vehicle.nose_radius
    qd = heat_flux_sutton_graves(rho, v, rn)
    qd_dkr = heat_flux_dkr(rho, v, rn)
    qd_315 = heat_flux_exponent_sensitivity(rho, v, rn)
    t_wall = radiative_equilibrium_temperature(qd, model.vehicle.emissivity)

    return TrajectoryResult(
        t=t, altitude=h, radius=r, velocity=v, gamma=gam, psi=psi, lat=lat, lon=lon,
        downrange=s, density=rho, mach=mach, q_dyn=q_dyn, q_dot=qd, q_dot_dkr=qd_dkr,
        q_dot_exp315=qd_315, heat_load=q_int, g_load=g_load, wall_temperature=t_wall,
        drag_work=w_int, cd=cd, cl=cl, termination=termination, meta=meta,
    )


def simulate(
    vehicle: Vehicle | None = None,
    atmosphere: AtmosphereBase | None = None,
    *,
    altitude0: float = 120.0e3,
    velocity0: float = 7800.0,
    gamma0_deg: float = -5.5,
    psi0_deg: float = 90.0,
    lat0_deg: float = 0.0,
    lon0_deg: float = 0.0,
    omega: float = 0.0,
    t_max: float = 4000.0,
    rtol: float = 1e-10,
    atol: float = 1e-10,
    terminal_altitude: float = 10.0e3,
    exit_altitude: float = 200.0e3,
    n_output: int = 4001,
    include_gravity: bool = True,
    curvature: bool = True,
    freeze_gamma: bool = False,
    constant_cd: float | None = None,
    r_planet: float = R_EARTH,
    mu: float = MU_EARTH,
) -> TrajectoryResult:
    """Integrate an entry trajectory and return the post-processed history.

    Termination is by event: descent through ``terminal_altitude`` (normal), or
    ascent through ``exit_altitude`` (skip-out), or ``t_max`` (timeout).
    """
    vehicle = vehicle if vehicle is not None else Vehicle()
    atmosphere = atmosphere if atmosphere is not None else USSA76()
    model = EntryModel(
        vehicle=vehicle,
        atmosphere=atmosphere,
        omega=omega,
        mu=mu,
        r_planet=r_planet,
        include_gravity=include_gravity,
        curvature=curvature,
        freeze_gamma=freeze_gamma,
        constant_cd=constant_cd,
    )

    y0 = np.array(
        [
            r_planet + altitude0,
            np.radians(lon0_deg),
            np.radians(lat0_deg),
            velocity0,
            np.radians(gamma0_deg),
            np.radians(psi0_deg),
            0.0,
            0.0,
            0.0,
        ]
    )

    def ev_land(t, y):
        return y[0] - (r_planet + terminal_altitude)

    ev_land.terminal = True
    ev_land.direction = -1.0

    def ev_skip(t, y):
        return y[0] - (r_planet + exit_altitude)

    ev_skip.terminal = True
    ev_skip.direction = 1.0

    def ev_stop(t, y):
        return y[3] - 1.0

    ev_stop.terminal = True
    ev_stop.direction = -1.0

    sol = integrate_dop853(
        model.rhs, (0.0, t_max), y0, events=[ev_land, ev_skip, ev_stop], rtol=rtol, atol=atol
    )
    if not sol.success:
        raise RuntimeError(f"integration failed: {sol.message}")

    if sol.t_events[0].size:
        termination = "terminal_altitude"
        t_end = float(sol.t_events[0][0])
    elif sol.t_events[1].size:
        termination = "skip_out"
        t_end = float(sol.t_events[1][0])
    elif sol.t_events[2].size:
        termination = "velocity_floor"
        t_end = float(sol.t_events[2][0])
    else:
        termination = "t_max"
        t_end = float(sol.t[-1])

    t = np.linspace(0.0, t_end, n_output)
    y = sol.sol(t)
    meta = {
        "rtol": rtol,
        "atol": atol,
        "omega": omega,
        "atmosphere": atmosphere.name,
        "vehicle": vehicle.describe(),
        "n_solver_steps": int(sol.t.size),
        "n_rhs_evals": int(sol.nfev),
        "initial_conditions": {
            "altitude_km": altitude0 / 1e3,
            "velocity_m_s": velocity0,
            "gamma_deg": gamma0_deg,
            "psi_deg": psi0_deg,
            "lat_deg": lat0_deg,
            "lon_deg": lon0_deg,
        },
        "options": {
            "include_gravity": include_gravity,
            "curvature": curvature,
            "freeze_gamma": freeze_gamma,
            "constant_cd": constant_cd,
        },
    }
    res = _postprocess(model, t, y, termination, meta)
    res.meta["raw_solution"] = sol
    return res


def hot_wall_temperature(result: TrajectoryResult, emissivity: float | None = None):
    """Radiative-equilibrium wall temperature with the hot-wall enthalpy
    correction applied (fixed-point iteration, 20 sweeps)."""
    from .constants import SIGMA_SB

    eps = emissivity if emissivity is not None else result.meta["vehicle"]["emissivity"]
    tw = result.wall_temperature.copy()
    for _ in range(20):
        f = hot_wall_factor(result.velocity, tw)
        tw = (result.q_dot * f / (eps * SIGMA_SB)) ** 0.25
    return tw
