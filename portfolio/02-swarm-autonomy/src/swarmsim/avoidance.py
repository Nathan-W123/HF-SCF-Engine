"""Decentralised 3-D collision avoidance: ORCA + a CBF safety filter.

Each vehicle runs this on its own, using only what it can sense locally: the
positions and ground velocities of neighbours inside ``sense_range`` (capped at
the ``max_neighbours`` nearest).  There is no central allocator, no shared plan
and no inter-vehicle negotiation.

Layer 1 -- ORCA (optimal reciprocal collision avoidance), 3-D
------------------------------------------------------------
For a pair (A, B) with relative position ``x = p_B - p_A``, relative velocity
``v = v_A - v_B`` and combined radius ``r = r_A + r_B``, the velocity obstacle
over a time horizon ``tau`` is a truncated cone.  ``u`` is the smallest vector
that takes ``v`` to the boundary of that cone; ORCA then gives A the half-space

    (v_new - (v_A + f * u)) . n  >=  0,

where ``n = u / |u|`` and ``f`` is A's share of the avoidance effort.  With two
cooperative vehicles ``f = 1/2`` each, so together they undo the whole conflict
and neither has to know what the other will do -- that is the reciprocity
property, and it is tested directly in ``tests/test_avoidance.py``.  A vehicle
flagged non-cooperative (a failure case) is given ``f = 0`` by itself and the
cooperative vehicles take ``f = 1`` for it, i.e. full responsibility.

Layer 2 -- velocity control barrier function
--------------------------------------------
ORCA's guarantee is only as good as its model of the neighbour's future.  A
rogue vehicle, a heavy gust or a kinematically unreachable ORCA solution can all
break it.  The second layer therefore adds, for every neighbour, the linear
constraint

    n_ij . (v_i - v_j)  >=  -k(h_ij),     h_ij = |p_ij| - (R_min + margin),

with ``n_ij`` the unit line-of-sight from j to i.  This is the velocity-level
CBF condition ``h_dot >= -k(h)`` for the barrier ``h = |p_ij| - R_safe``: while
``h > 0`` it permits closure but at a rate that decays to zero as the boundary
is approached, so ``h >= 0`` is forward invariant for the kinematic
(single-integrator) model.  It is *not* a guarantee for the full fixed-wing
model, because the commanded velocity is only reached through the bank/airspeed
lags -- which is exactly why we measure the realised separation rather than
assert the theory.

Layer 3 -- no-fly zones
-----------------------
Each nearby no-fly cylinder contributes the same kind of barrier row with
``h = |p_xy - c_xy| - (R_zone + margin)`` and a purely horizontal normal, so a
vehicle pushed off its planned route by traffic still cannot be pushed into a
no-fly zone.

Barrier profile
---------------
``h_dot >= -k(h)`` uses the *tighter* of a linear class-K function ``alpha h``
and a braking bound ``sqrt(2 a_cbf h)``.  The braking form is the right one for
a vehicle that cannot stop: it caps the closing speed at exactly what can be
bled off over the remaining distance with the available lateral acceleration.

Assembly
--------
All three layers go into one QP,

    min |W(v - v_pref)|^2 + sum_k rho_k s_k^2   s.t.  A v >= b - s,  s >= 0,

with the barrier rows **hard** (``rho = inf``) and the ORCA rows **soft**, their
stiffness scaled by how close the neighbour is.  Soft ORCA rows matter: in a
dense swarm the intersection of twenty ORCA half-spaces is often empty, and the
usual remedy (discard them all) throws away the easy constraints along with the
impossible ones.  Only if the *hard* rows are jointly infeasible do we fall back
to a max-margin velocity over a fixed candidate fan, and that event is counted.

Kinematic projection
--------------------
The QP works on *ground* velocity.  Both its input and its output are projected
onto the set of velocities the airframe can reach within a manoeuvre horizon
(see :func:`project_to_kinematic_set`), so the safety layer never reasons about
-- or commands -- a velocity the aircraft cannot fly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._kernels import HAVE_NUMBA, avoid_kernel
from .config import AvoidanceParams, G0, VehicleParams
from .dynamics import wrap_pi
from .qp import max_margin_velocity, solve_qp


@dataclass
class AvoidanceDiagnostics:
    n_constraints: np.ndarray      # (N,) active ORCA+CBF rows per vehicle
    deflection: np.ndarray         # (N,) |v_safe - v_pref| [m/s]
    qp_infeasible: np.ndarray      # (N,) bool: ORCA+CBF set was infeasible
    fallback_used: np.ndarray      # (N,) bool: max-margin fallback was used


HEAD_ON_TOL = 0.08     # |ww| / |v_rel| below which an encounter counts as head-on
                       # (asin(0.08) ~ 4.6 deg of relative-bearing spread)


def _right_of_way_normal(p_rel, r_comb):
    """ORCA plane normal for an exactly head-on encounter.

    When the relative velocity lies on the axis of the velocity-obstacle cone,
    ``ww = v_rel - t p_rel`` vanishes and the usual normal ``ww/|ww|`` is
    undefined -- every escape direction is equally good, and a naive choice
    lands *inside* the obstacle.  Writing ``v_new = t (p + r w)`` for a unit
    ``w``, the escaped velocity is on the true cone boundary
    ``|p x v_new| / |v_new| = r`` exactly when

        cos(beta) = -r / |p|,        beta = angle(p, w),

    so the normal is built as ``w = cos(beta) p_hat + sin(beta) e`` with ``e``
    the unit horizontal vector to the **right** of the line of sight, and the
    escape is ``u = t (p + r w) - v_rel``.  Picking
    "right" is the rules-of-the-air head-on convention, and because vehicle A
    sees ``p_rel`` while B sees ``-p_rel`` (so ``e`` flips too), ``w`` is exactly
    antisymmetric and the ORCA reciprocity property ``u_A = -u_B`` survives.
    """
    d = float(np.linalg.norm(p_rel))
    h = np.array([p_rel[1], -p_rel[0], 0.0])
    nh = float(np.linalg.norm(h))
    if nh < 1e-9:
        # Line of sight is vertical: any horizontal direction will do.
        e = np.array([1.0, 0.0, 0.0])
    else:
        e = h / nh
    cb = -r_comb / max(d, 1e-9)
    cb = float(np.clip(cb, -1.0, 1.0))
    sb = float(np.sqrt(max(1.0 - cb * cb, 0.0)))
    w = cb * (p_rel / max(d, 1e-9)) + sb * e
    nw = float(np.linalg.norm(w))
    return w / max(nw, 1e-12)


def orca_halfspace(p_rel, v_rel, r_comb, tau, tau_escape):
    """ORCA half-space for one neighbour pair.

    Parameters
    ----------
    p_rel : neighbour position minus own position  (3,)
    v_rel : own velocity minus neighbour velocity  (3,)
    r_comb : combined radius [m]
    tau : time horizon [s]
    tau_escape : recovery horizon [s] used when the pair is already inside the
        combined radius.  The textbook RVO2 implementation uses the control
        interval here, which for a 0.1 s tick demands ``r_comb / dt`` = several
        hundred m/s and makes the constraint set trivially infeasible for an
        aircraft.  A physically reachable recovery horizon is used instead.

    Returns
    -------
    (n, u) with ``n`` the unit outward normal and ``u`` the minimum change to
    the relative velocity that escapes the velocity obstacle.
    """
    p_rel = np.asarray(p_rel, float)
    v_rel = np.asarray(v_rel, float)
    dist_sq = float(p_rel @ p_rel)
    r2 = r_comb * r_comb

    if dist_sq > r2:
        w = v_rel - p_rel / tau
        w_len_sq = float(w @ w)
        dot = float(w @ p_rel)
        if dot < 0.0 and dot * dot > r2 * w_len_sq:
            # Closest point is on the spherical cut-off cap.
            w_len = np.sqrt(max(w_len_sq, 1e-18))
            n = w / w_len
            u = (r_comb / tau - w_len) * n
        else:
            # Closest point is on the lateral surface of the cone.
            a = dist_sq
            bq = float(p_rel @ v_rel)
            cross = np.cross(p_rel, v_rel)
            c = float(v_rel @ v_rel) - float(cross @ cross) / max(dist_sq - r2, 1e-9)
            disc = max(bq * bq - a * c, 0.0)
            t = (bq + np.sqrt(disc)) / max(a, 1e-12)
            ww = v_rel - t * p_rel
            ww_len = float(np.linalg.norm(ww))
            v_rel_len = float(np.linalg.norm(v_rel))
            if ww_len < HEAD_ON_TOL * max(v_rel_len, 1e-6):
                # Head-on degeneracy: the relative velocity lies on the cone
                # axis, so every perpendicular direction is an equally good
                # escape and the cone normal is numerically meaningless.  Break
                # the tie the way the rules of the air do -- both aircraft turn
                # right.  Because vehicle A sees p_rel and vehicle B sees
                # -p_rel, the two tie-break normals are exact negatives, so the
                # reciprocity property u_A = -u_B is preserved.
                n = _right_of_way_normal(p_rel, r_comb)
                # ``n`` is chosen rather than derived from ``ww``, so the escape
                # must be written in its general form: the cone point is
                # t (p + r n), exactly as in the non-degenerate branch where
                # v + u = t p + r t ww_hat.
                u = t * (p_rel + r_comb * n) - v_rel
            else:
                n = ww / ww_len
                u = (r_comb * t - ww_len) * n
    else:
        # Already inside the combined radius: escape over ``tau_escape``.
        w = v_rel - p_rel / tau_escape
        w_len = float(np.linalg.norm(w))
        if w_len < 1e-12:
            n = np.array([0.0, 0.0, 1.0])
            u = np.zeros(3)
        else:
            n = w / w_len
            u = (r_comb / tau_escape - w_len) * n
    return n, u


_FALLBACK_DIRS = None


def _fallback_candidates(v_pref, speed_lo, speed_hi):  # noqa: ARG001
    """A fixed spherical fan of candidate velocities for the infeasible case."""
    global _FALLBACK_DIRS
    if _FALLBACK_DIRS is None:
        # Fibonacci sphere, 96 directions.
        k = np.arange(96) + 0.5
        phi = np.arccos(1.0 - 2.0 * k / 96)
        theta = np.pi * (1.0 + 5.0 ** 0.5) * k
        _FALLBACK_DIRS = np.stack([np.sin(phi) * np.cos(theta),
                                   np.sin(phi) * np.sin(theta),
                                   np.cos(phi)], axis=1)
    sp = np.array([speed_lo, 0.5 * (speed_lo + speed_hi), speed_hi])
    return (_FALLBACK_DIRS[:, None, :] * sp[None, :, None]).reshape(-1, 3)


class CollisionAvoidance:
    """Per-vehicle ORCA + CBF velocity filter (decentralised)."""

    def __init__(self, ap: AvoidanceParams, vp: VehicleParams, zones=None):
        self.ap = ap
        self.vp = vp
        self.zones = list(zones) if zones else []
        if self.zones:
            self.zone_c = np.array([[z.x, z.y] for z in self.zones], float)
            self.zone_r = np.array([z.radius for z in self.zones], float)
            self.zone_lo = np.array([z.z_low for z in self.zones], float)
            self.zone_hi = np.array([z.z_high for z in self.zones], float)

    # -- barrier profile ------------------------------------------------
    def _max_closure(self, h):
        """Largest permitted closing speed at barrier value ``h``.

        The tighter of a linear class-K function ``alpha h`` and the
        braking-distance bound ``sqrt(2 a h)``.  The braking form is what a
        vehicle with bounded lateral acceleration actually needs: at range it
        limits closure far more than a linear gain would, which is precisely
        where a fixed-wing aircraft must start reacting because it cannot stop.
        """
        h = max(h, 0.0)
        return min(self.ap.cbf_alpha * h,
                   float(np.sqrt(2.0 * self.ap.cbf_accel * h)))

    # -- compiled fast path --------------------------------------------
    def filter_fast(self, pos, vel, v_pref, dt, cooperative=None, active=None):
        """Numba-compiled equivalent of :meth:`filter` (same arithmetic).

        ``tests/test_kernel_equivalence.py`` asserts the two agree to 1e-9 on
        randomised swarm states, so this is purely an optimisation.
        """
        ap = self.ap
        n = pos.shape[0]
        pos = np.ascontiguousarray(pos, dtype=np.float64)
        vel = np.ascontiguousarray(vel, dtype=np.float64)
        v_pref = np.ascontiguousarray(v_pref, dtype=np.float64)
        coop = (np.ones(n, np.bool_) if cooperative is None
                else np.ascontiguousarray(cooperative, dtype=np.bool_))
        act = (np.ones(n, np.bool_) if active is None
               else np.ascontiguousarray(active, dtype=np.bool_))
        if self.zones:
            zc, zr, zlo, zhi = self.zone_c, self.zone_r, self.zone_lo, self.zone_hi
        else:
            zc = np.zeros((0, 2))
            zr = np.zeros(0)
            zlo = np.zeros(0)
            zhi = np.zeros(0)

        v_out = np.empty((n, 3))
        n_con = np.zeros(n, np.int64)
        infeas = np.zeros(n, np.bool_)
        fallback = np.zeros(n, np.bool_)
        fb = _fallback_candidates(None, self.vp.V_min, self.vp.V_max)

        avoid_kernel(pos, vel, v_pref, coop, act,
                     np.ascontiguousarray(zc, dtype=np.float64),
                     np.ascontiguousarray(zr, dtype=np.float64),
                     np.ascontiguousarray(zlo, dtype=np.float64),
                     np.ascontiguousarray(zhi, dtype=np.float64),
                     2.0 * ap.orca_radius, ap.tau_horizon, ap.tau_escape,
                     ap.sense_range, int(ap.max_neighbours), HEAD_ON_TOL,
                     ap.R_min + ap.cbf_margin, ap.cbf_alpha, ap.cbf_accel,
                     ap.zone_margin, ap.zone_sense_range,
                     ap.orca_rho, ap.orca_rho_urgency,
                     ap.vertical_pref_gain, int(ap.qp_sweeps), 1e-7,
                     3.0 * self.vp.V_max,
                     np.ascontiguousarray(fb, dtype=np.float64),
                     v_out, n_con, infeas, fallback)

        diag = AvoidanceDiagnostics(n_con, np.linalg.norm(v_out - v_pref, axis=1),
                                    infeas, fallback)
        return v_out, diag

    def filter(self,
               pos: np.ndarray,
               vel: np.ndarray,
               v_pref: np.ndarray,
               dt: float,
               cooperative: np.ndarray | None = None,
               active: np.ndarray | None = None):
        """Compute safe ground velocities for all vehicles.

        Although this is written as a loop over vehicles for clarity, each pass
        touches only vehicle ``i`` and its sensed neighbours -- no global state
        is shared between vehicles within a step, so it is a faithful model of a
        decentralised implementation.
        """
        ap = self.ap
        n = pos.shape[0]
        pos = np.asarray(pos, float)
        vel = np.asarray(vel, float)
        v_pref = np.asarray(v_pref, float)
        if cooperative is None:
            cooperative = np.ones(n, bool)
        if active is None:
            active = np.ones(n, bool)

        v_out = v_pref.copy()
        n_con = np.zeros(n, int)
        infeas = np.zeros(n, bool)
        fallback = np.zeros(n, bool)

        d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)

        r_comb = 2.0 * ap.orca_radius
        v_clamp = 3.0 * self.vp.V_max
        r_safe = ap.R_min + ap.cbf_margin
        wz = ap.vertical_pref_gain
        w = np.array([1.0, 1.0, wz])

        for i in range(n):
            if not active[i]:
                continue
            if not cooperative[i]:
                # A non-cooperative (failed / rogue) vehicle runs no avoidance at
                # all: it flies its own command and the others must work around it.
                continue
            cand = np.where(d[i] < ap.sense_range)[0]
            if len(cand) > ap.max_neighbours:
                cand = cand[np.argsort(d[i, cand])[:ap.max_neighbours]]

            rows, rhs, rho = [], [], []

            # --- layer 1: ORCA half-spaces (soft, urgency-weighted) -------
            for j in cand:
                p_rel = pos[j] - pos[i]
                v_rel = vel[i] - vel[j]
                nrm, u = orca_halfspace(p_rel, v_rel, r_comb, ap.tau_horizon,
                                        ap.tau_escape)
                # Cooperative pair -> split the effort; non-cooperative
                # neighbour -> this vehicle takes the whole burden.
                share = 0.5 if cooperative[j] else 1.0
                rows.append(nrm)
                rhs.append(float(nrm @ (vel[i] + share * u)))
                # Urgency: a neighbour at the combined radius is ORCA_RHO_MAX
                # times stiffer than one at the edge of the sensing range.
                urg = np.clip(r_comb / max(d[i, j], 1e-6), 0.0, 1.0)
                rho.append(ap.orca_rho * (1.0 + ap.orca_rho_urgency * urg ** 2))

            # --- layer 2: pairwise CBF (hard) -----------------------------
            for j in cand:
                p_ij = pos[i] - pos[j]
                dij = float(np.linalg.norm(p_ij))
                if dij < 1e-6:
                    continue
                nij = p_ij / dij
                rows.append(nij)
                rhs.append(float(nij @ vel[j]) - self._max_closure(dij - r_safe))
                rho.append(np.inf)

            # --- layer 3: no-fly-zone CBF (hard) --------------------------
            for k in range(len(self.zones)):
                if not (self.zone_lo[k] - 60.0 < pos[i, 2] < self.zone_hi[k] + 60.0):
                    continue
                rad = pos[i, :2] - self.zone_c[k]
                dz = float(np.linalg.norm(rad))
                if dz > ap.zone_sense_range + self.zone_r[k]:
                    continue
                if dz < 1e-6:
                    continue
                nz = np.array([rad[0] / dz, rad[1] / dz, 0.0])
                h = dz - (self.zone_r[k] + ap.zone_margin)
                rows.append(nz)
                rhs.append(-self._max_closure(h))
                rho.append(np.inf)

            if not rows:
                continue
            A = np.array(rows)
            b = np.array(rhs)
            rr = np.array(rho)
            n_con[i] = A.shape[0]

            v, ok, viol = solve_qp(v_pref[i], A, b, weights=w, rho=rr,
                                   sweeps=ap.qp_sweeps, tol=1e-7)
            if not ok:
                # The hard rows alone are infeasible: take the least-unsafe
                # velocity over a fixed candidate fan and record the event.
                infeas[i] = True
                hard = ~np.isfinite(rr)
                cands = _fallback_candidates(v_pref[i], self.vp.V_min,
                                             self.vp.V_max)
                v, _ = max_margin_velocity(A[hard], b[hard], v_pref[i], cands)
                fallback[i] = True
            sp_v = float(np.linalg.norm(v))
            if sp_v > v_clamp:
                v = v * (v_clamp / sp_v)
            v_out[i] = v

        diag = AvoidanceDiagnostics(n_con, np.linalg.norm(v_out - v_pref, axis=1),
                                    infeas, fallback)
        return v_out, diag


MANOEUVRE_HORIZON = 2.0     # [s] window over which a command must be reachable


def project_to_kinematic_set(state: np.ndarray,
                             v_ground_des: np.ndarray,
                             wind: np.ndarray,
                             vp: VehicleParams,
                             horizon: float = MANOEUVRE_HORIZON,
                             return_ground: bool = True) -> np.ndarray:
    """Project a desired ground velocity onto the kinematically reachable set.

    The reachable set is evaluated over a *manoeuvre horizon* ``T_m`` (roughly
    the roll-in time plus a margin) rather than over a single control tick,
    because that is the time scale on which the airframe actually redirects its
    velocity vector.  Over ``T_m`` the vehicle can

    * change heading by at most ``g tan(phi_max) / V * T_m``,
    * change flight-path angle by at most ``gamma_dot_max * T_m`` and in any
      case stay inside ``+/- gamma_max``,
    * change airspeed by at most ``accel_max * T_m`` and stay in
      ``[V_min, V_max]``.

    Applying this to the *preferred* velocity before the QP keeps ORCA from
    reasoning about velocities the aircraft could never fly; applying it again
    to the QP output gives the velocity actually commanded.  The residual
    ``|v_qp - v_projected|`` is logged as the achievability residual.
    """
    from .dynamics import IGAM, IPSI, IV

    s = np.atleast_2d(state)
    w = np.atleast_2d(wind)
    v_air = np.asarray(v_ground_des, float) - w

    speed = np.maximum(np.linalg.norm(v_air, axis=1), 1e-6)
    V = s[:, IV]
    V_new = np.clip(speed,
                    np.maximum(V - vp.accel_max * horizon, vp.V_min),
                    np.minimum(V + vp.accel_max * horizon, vp.V_max))

    gam_des = np.arcsin(np.clip(v_air[:, 2] / speed, -1.0, 1.0))
    gam = s[:, IGAM]
    gam_new = np.clip(gam_des,
                      np.maximum(gam - vp.gamma_dot_max * horizon, -vp.gamma_max),
                      np.minimum(gam + vp.gamma_dot_max * horizon, vp.gamma_max))

    psi_des = np.arctan2(v_air[:, 1], v_air[:, 0])
    dpsi = wrap_pi(psi_des - s[:, IPSI])
    dpsi_max = G0 * np.tan(vp.phi_limit) / np.maximum(V, 1.0) * horizon
    psi_new = s[:, IPSI] + np.clip(dpsi, -dpsi_max, dpsi_max)

    cg = np.cos(gam_new)
    v_air_new = np.stack([V_new * cg * np.cos(psi_new),
                          V_new * cg * np.sin(psi_new),
                          V_new * np.sin(gam_new)], axis=1)
    return v_air_new + w if return_ground else v_air_new
