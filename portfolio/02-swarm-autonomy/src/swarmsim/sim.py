"""Swarm simulation engine: ties dynamics, wind, guidance and avoidance together.

Control architecture (per vehicle, all decentralised)
----------------------------------------------------

    global plan  ->  L1 guidance  ->  ORCA + CBF  ->  kinematic projection
                                                        |
                                                        v
                                        coordinated-turn autopilot commands
                                          (V_cmd, phi_cmd, gamma_cmd)
                                                        |
                                                        v
                                             RK4 3-DOF fixed-wing model

* The **global plan** is computed once per vehicle by the tangent-visibility-graph
  planner and is a fixed polyline of horizontal waypoints.
* **L1 guidance** turns the plan into a preferred ground-velocity vector.
* **ORCA + CBF** deflect that preferred velocity using only sensed neighbours.
* The **kinematic projection** clips the result to what the airframe can reach
  within the manoeuvre horizon.
* The **autopilot** converts the commanded velocity to bank / flight-path /
  airspeed commands, which the 3-DOF model follows through its lags.

Heading-loop bandwidth
----------------------
While the avoidance layer is inactive the heading rate is commanded by the L1
law itself, ``psi_dot = 2 V_g sin(eta) / L1``.  When the avoidance layer has
deflected the command by more than ``DEFLECT_TRIGGER`` the loop switches to a
higher-bandwidth tracker (an effective ``L1 = 2 V_g / k_avoid``), because ORCA's
half-space guarantee is predicated on the commanded velocity being achieved
promptly rather than over a 12-second path-following time constant.  The switch
is logged, and validation 2 exercises the pure-L1 branch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from .avoidance import CollisionAvoidance, project_to_kinematic_set
from .config import (AvoidanceParams, GuidanceParams, NoFlyZone, SimParams,
                     VehicleParams, WindParams)
from .dynamics import (IGAM, IPHI, IPSI, IV, IX, IY, IZ, air_velocity,
                       commands_from_velocity, ground_velocity, rk4_step,
                       wrap_pi)
from .guidance import PolylinePath, l1_distance, preferred_velocity
from .planner import VisibilityGraphPlanner, path_length
from .wind import WindField

DEFLECT_TRIGGER = 0.25     # [m/s] avoidance deflection that switches the loop
K_AVOID = 1.2              # [1/s] heading-loop bandwidth while avoiding


@dataclass
class SwarmSpec:
    """Everything needed to define a scenario."""

    name: str
    starts: np.ndarray                  # (N, 3) ENU start positions
    goals: np.ndarray                   # (N, 3) ENU goal positions
    zones: List[NoFlyZone] = field(default_factory=list)
    wind: WindParams = field(default_factory=WindParams)
    sim: SimParams = field(default_factory=SimParams)
    vehicle: VehicleParams = field(default_factory=VehicleParams)
    guidance: GuidanceParams = field(default_factory=GuidanceParams)
    avoid: AvoidanceParams = field(default_factory=AvoidanceParams)
    clearance: float = 90.0
    # Failure injection
    fail_id: Optional[int] = None       # vehicle that goes non-cooperative
    fail_time: float = 1e9              # [s] when it does
    fail_bank: float = 0.0              # [rad] bank it locks into (0 = straight)
    avoidance_on: bool = True
    initial_heading_to_goal: bool = True


@dataclass
class SwarmResult:
    name: str
    t: np.ndarray                     # (T,)
    states: np.ndarray                # (T, N, 7)
    wind: np.ndarray                  # (T, N, 3)
    v_ground: np.ndarray              # (T, N, 3)
    deflection: np.ndarray            # (T, N) avoidance deflection magnitude
    reached: np.ndarray               # (T, N) bool goal reached
    cooperative: np.ndarray           # (T, N) bool
    reach_time: np.ndarray            # (N,) first time goal reached (nan if never)
    plan_length: np.ndarray           # (N,) planned horizontal path length
    direct_length: np.ndarray         # (N,) straight-line start->goal length
    flown_length: np.ndarray          # (N,) actual 3-D flown length
    control_effort: np.ndarray        # (N,) integral of |phi| dt  [rad s]
    n_qp_infeasible: int
    n_fallback: int
    plan_failures: int
    spec: SwarmSpec
    paths: List[np.ndarray]           # planned waypoint polylines (M_i, 2)


def build_paths(spec: SwarmSpec):
    planner = VisibilityGraphPlanner(spec.zones, clearance=spec.clearance)
    paths, lengths, fails = [], [], 0
    for i in range(len(spec.starts)):
        res = planner.plan(spec.starts[i, :2], spec.goals[i, :2])
        if not res.ok:
            fails += 1
        paths.append(res.waypoints)
        lengths.append(res.length)
    return paths, np.array(lengths), fails


def run(spec: SwarmSpec, progress: bool = False) -> SwarmResult:
    vp, gp, ap, sp = spec.vehicle, spec.guidance, spec.avoid, spec.sim
    n = len(spec.starts)
    rng = np.random.default_rng(sp.seed)

    raw_paths, plan_len, plan_failures = build_paths(spec)
    paths = [PolylinePath(w) for w in raw_paths]

    # ---- initial state ---------------------------------------------------
    state = np.zeros((n, 7))
    state[:, :3] = spec.starts
    state[:, IV] = vp.V_nom
    if spec.initial_heading_to_goal:
        d0 = np.array([p.wp[1] - p.wp[0] for p in paths])
        state[:, IPSI] = np.arctan2(d0[:, 1], d0[:, 0])
    state[:, IGAM] = 0.0
    state[:, IPHI] = 0.0

    wind_field = WindField(spec.wind, n, vp.V_nom, sp.dt, rng)
    ca = CollisionAvoidance(ap, vp, zones=spec.zones)

    dt = sp.dt
    dt_ctrl = dt * sp.control_every
    n_steps = int(round(sp.t_max / dt))

    h_ref = spec.goals[:, 2].copy()
    cooperative = np.ones(n, bool)

    cmd = np.zeros((n, 3))
    cmd[:, 0] = vp.V_nom
    deflect = np.zeros(n)

    log_t, log_s, log_w, log_vg, log_def, log_reached, log_coop = [], [], [], [], [], [], []
    reach_time = np.full(n, np.nan)
    flown = np.zeros(n)
    effort = np.zeros(n)
    n_infeas = 0
    n_fallback = 0
    prev_pos = state[:, :3].copy()

    it = range(n_steps)
    if progress:
        from tqdm import tqdm
        it = tqdm(it, desc=spec.name, ncols=78)

    for k in it:
        t = k * dt

        # -- failure injection --------------------------------------------
        if spec.fail_id is not None and t >= spec.fail_time and cooperative[spec.fail_id]:
            cooperative[spec.fail_id] = False

        wind_field.step(state[:, IPSI])
        wind = wind_field.wind()

        if k % sp.control_every == 0:
            vg = ground_velocity(state, wind)
            reached = np.linalg.norm(state[:, :2] - spec.goals[:, :2], axis=1) < gp.goal_radius
            newly = reached & np.isnan(reach_time)
            reach_time[newly] = t

            # ---- L1 guidance -> preferred ground velocity ----------------
            v_pref = np.zeros((n, 3))
            L1 = np.zeros(n)
            for i in range(n):
                v_pref[i], L1[i] = preferred_velocity(
                    state[i, :3], vg[i], paths[i], h_ref[i], vp.V_nom, gp)
            # Vehicles that have arrived hold a loiter heading (keep flying,
            # they do not vanish -- they still have to be avoided).
            if reached.any():
                idx = np.where(reached)[0]
                v_pref[idx] = vg[idx] / np.maximum(
                    np.linalg.norm(vg[idx], axis=1, keepdims=True), 1e-6) * vp.V_nom

            # A vehicle with a locked control failure ignores guidance entirely.
            if spec.fail_id is not None and not cooperative[spec.fail_id]:
                fi = spec.fail_id
                v_pref[fi] = air_velocity(state[fi:fi + 1])[0] + wind[fi]

            # ---- pre-project onto the reachable set ----------------------
            v_pref = project_to_kinematic_set(state, v_pref, wind, vp)

            # ---- ORCA + CBF ---------------------------------------------
            if spec.avoidance_on:
                v_safe, diag = ca.filter_fast(state[:, :3], vg, v_pref, dt_ctrl,
                                              cooperative=cooperative)
                n_infeas += int(diag.qp_infeasible.sum())
                n_fallback += int(diag.fallback_used.sum())
                deflect = diag.deflection
            else:
                v_safe = v_pref
                deflect = np.zeros(n)

            # ---- project the safe velocity onto the reachable set --------
            v_cmd_ground = project_to_kinematic_set(state, v_safe, wind, vp)
            v_air_des = v_cmd_ground - wind

            # ---- heading-loop bandwidth ---------------------------------
            sp_g = np.linalg.norm(v_safe[:, :2], axis=1)
            L1_eff = np.where(deflect > DEFLECT_TRIGGER,
                              2.0 * np.maximum(sp_g, 1.0) / K_AVOID, L1)
            cmd = commands_from_velocity(state, v_air_des, vp, L1_eff, dt_ctrl)

            # A failed vehicle holds a fixed bank and airspeed: no authority.
            if spec.fail_id is not None and not cooperative[spec.fail_id]:
                cmd[spec.fail_id] = (vp.V_nom, spec.fail_bank, 0.0)

        state = rk4_step(state, cmd, dt, vp, wind=wind)

        # Path length and control effort are accumulated only while a vehicle is
        # still en route; once it has reached its goal it keeps flying (so it
        # must still be avoided) but its mission is over and further distance
        # would corrupt the efficiency metric.
        en_route = np.isnan(reach_time)
        step_d = np.linalg.norm(state[:, :3] - prev_pos, axis=1)
        flown += step_d * en_route
        prev_pos = state[:, :3].copy()
        effort += np.abs(state[:, IPHI]) * dt * en_route

        if k % sp.log_every == 0:
            log_t.append(t)
            log_s.append(state.copy())
            log_w.append(wind.copy())
            log_vg.append(ground_velocity(state, wind))
            log_def.append(deflect.copy())
            log_reached.append(~np.isnan(reach_time))
            log_coop.append(cooperative.copy())

    direct = np.linalg.norm(spec.goals - spec.starts, axis=1)
    return SwarmResult(
        name=spec.name,
        t=np.array(log_t),
        states=np.array(log_s),
        wind=np.array(log_w),
        v_ground=np.array(log_vg),
        deflection=np.array(log_def),
        reached=np.array(log_reached),
        cooperative=np.array(log_coop),
        reach_time=reach_time,
        plan_length=plan_len,
        direct_length=direct,
        flown_length=flown,
        control_effort=effort,
        n_qp_infeasible=n_infeas,
        n_fallback=n_fallback,
        plan_failures=plan_failures,
        spec=spec,
        paths=raw_paths,
    )
