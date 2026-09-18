"""Trajectory tracking: L1 nonlinear guidance plus altitude and airspeed loops.

L1 nonlinear guidance (Park, Deyst & How, AIAA GNC 2004)
-------------------------------------------------------
Pick a reference point on the path a distance ``L1`` ahead of the vehicle.  Let
``eta`` be the angle between the vehicle's ground-velocity vector and the vector
to that point.  The law commands the lateral acceleration

    a_s = 2 V_g^2 sin(eta) / L1                                    (1)

which, for a circular arc through the vehicle and the reference point, is
exactly the centripetal acceleration needed to fly that arc.  Two properties
make it a good fit here:

* On a straight path and for small ``eta`` it linearises to a second-order
  cross-track response with natural frequency ``omega_n = 2 V_g / L1`` and
  damping ``zeta = 1/sqrt(2)``, i.e. a well-damped approach with **no tuning
  per airspeed** -- the gain adapts automatically with speed.
* On a curved path it retains an anticipatory term, so steady-state cross-track
  error on a circular arc is small instead of the standing offset a pure
  proportional law leaves.

Equivalently, in heading-rate form, ``psi_dot = a_s / V_g = (2 V_g / L1) sin(eta)``,
which is the form used here (:func:`swarmsim.dynamics.commands_from_velocity`),
so that the identical law can track either the path reference direction or a
collision-avoidance-deflected velocity.

``L1`` is scheduled as ``L1 = clip(T_L1 * V_g, L1_min, L1_max)``.

Altitude and airspeed
---------------------
The altitude loop is a proportional climb-rate command,
``h_dot_cmd = clip(k_alt (h_ref - h), +/- h_dot_max)``, which is then folded into
the preferred velocity's vertical component; the flight-path-angle command
follows from ``gamma_cmd = asin(v_z / |v|)`` and is subject to the airframe's
``gamma`` and ``gamma_dot`` limits.  The airspeed loop simply commands the
nominal cruise speed unless the avoidance layer asks for something else.
"""

from __future__ import annotations

import numpy as np

from .config import GuidanceParams


def _closest_point_on_segment(p, a, b):
    d = b - a
    dd = float(d @ d)
    if dd < 1e-12:
        return a.copy(), 0.0
    t = float(np.clip(((p - a) @ d) / dd, 0.0, 1.0))
    return a + t * d, t


class PolylinePath:
    """A 2-D polyline with an L1 reference-point query and cross-track error."""

    def __init__(self, waypoints: np.ndarray):
        wp = np.asarray(waypoints, float)[:, :2]
        # drop duplicate consecutive points
        keep = [0]
        for i in range(1, len(wp)):
            if np.linalg.norm(wp[i] - wp[keep[-1]]) > 1e-6:
                keep.append(i)
        self.wp = wp[keep]
        if len(self.wp) < 2:
            self.wp = np.vstack([self.wp, self.wp[-1] + np.array([1.0, 0.0])])
        seg = np.diff(self.wp, axis=0)
        self.seg_len = np.linalg.norm(seg, axis=1)
        self.cum = np.concatenate([[0.0], np.cumsum(self.seg_len)])
        self.total = float(self.cum[-1])

    # -- arclength parameterisation ---------------------------------------
    def point_at(self, s: float) -> np.ndarray:
        s = float(np.clip(s, 0.0, self.total))
        k = int(np.searchsorted(self.cum, s, side="right") - 1)
        k = min(max(k, 0), len(self.seg_len) - 1)
        t = (s - self.cum[k]) / max(self.seg_len[k], 1e-9)
        return self.wp[k] + t * (self.wp[k + 1] - self.wp[k])

    def project(self, p: np.ndarray):
        """Return (arclength, closest point, signed cross-track error).

        Vectorised over all segments -- this runs once per vehicle per control
        step, so the numpy form matters.
        """
        p = np.asarray(p, float)[:2]
        a = self.wp[:-1]
        d = self.wp[1:] - a
        dd = np.maximum(np.einsum("ij,ij->i", d, d), 1e-12)
        t = np.clip(((p[None, :] - a) * d).sum(axis=1) / dd, 0.0, 1.0)
        q = a + t[:, None] * d
        dist = np.linalg.norm(p[None, :] - q, axis=1)
        k = int(np.argmin(dist))
        s = float(self.cum[k] + t[k] * self.seg_len[k])
        qk = q[k]
        tan = d[k] / max(np.sqrt(dd[k]), 1e-9)
        nrm = np.array([-tan[1], tan[0]])
        xte = float((p - qk) @ nrm)
        return s, qk, xte

    def l1_reference(self, p: np.ndarray, L1: float) -> np.ndarray:
        """The L1 reference point: on the path, Euclidean distance ``L1`` ahead.

        For a straight segment the along-path offset from the foot of the
        perpendicular is exactly ``sqrt(L1^2 - xte^2)``; that expression is used
        here and is therefore exact on the straight legs the planner produces,
        and a good approximation on the gently curved densified arcs.  When
        ``|xte| >= L1`` no such point exists and the law degenerates to steering
        straight at the closest point on the path, which is the standard L1
        fallback.
        """
        s, q, xte = self.project(p)
        if abs(xte) >= L1:
            return q
        return self.point_at(s + np.sqrt(max(L1 ** 2 - xte ** 2, 0.0)))

    def remaining(self, p: np.ndarray) -> float:
        s, _, _ = self.project(p)
        return self.total - s


class CirclePath:
    """A circular reference path (used by the tracking-convergence validation)."""

    def __init__(self, centre, radius, direction=+1):
        self.c = np.asarray(centre, float)[:2]
        self.R = float(radius)
        self.dir = int(np.sign(direction)) or 1
        self.total = 2 * np.pi * self.R

    def project(self, p):
        p = np.asarray(p, float)[:2]
        v = p - self.c
        r = float(np.linalg.norm(v))
        ang = float(np.arctan2(v[1], v[0]))
        q = self.c + self.R * np.array([np.cos(ang), np.sin(ang)])
        s = (ang % (2 * np.pi)) * self.R
        xte = (r - self.R) * self.dir      # positive = outside for CCW
        return s, q, float(xte)

    def point_at(self, s):
        ang = s / self.R
        return self.c + self.R * np.array([np.cos(ang), np.sin(ang)])

    def l1_reference(self, p, L1):
        """Exact intersection of the L1 circle with the reference circle."""
        p = np.asarray(p, float)[:2]
        v = p - self.c
        r = float(np.linalg.norm(v))
        th = float(np.arctan2(v[1], v[0]))
        if r < 1e-6:
            return self.c + self.R * np.array([1.0, 0.0])
        cosd = (self.R ** 2 + r ** 2 - L1 ** 2) / (2.0 * self.R * r)
        if abs(cosd) > 1.0:
            # No intersection -> steer at the closest point on the circle.
            return self.c + self.R * v / r
        delta = float(np.arccos(cosd))
        ang = th + self.dir * delta
        return self.c + self.R * np.array([np.cos(ang), np.sin(ang)])

    def remaining(self, p):
        return np.inf


def l1_distance(v_ground_speed, gp: GuidanceParams):
    return float(np.clip(gp.L1_period * v_ground_speed, gp.L1_min, gp.L1_max))


def lateral_acceleration(v_ground: np.ndarray, to_ref: np.ndarray, L1: float) -> float:
    """Park/Deyst/How lateral acceleration command, eq. (1)."""
    vg = np.asarray(v_ground, float)[:2]
    sp = float(np.linalg.norm(vg))
    r = np.asarray(to_ref, float)[:2]
    rn = float(np.linalg.norm(r))
    if sp < 1e-6 or rn < 1e-6:
        return 0.0
    # signed eta: positive when the reference lies to the left of the velocity
    a = vg / sp
    c = r / rn
    sin_eta = float(a[0] * c[1] - a[1] * c[0])
    return 2.0 * sp * sp * sin_eta / L1


def preferred_velocity(pos3: np.ndarray,
                       v_ground: np.ndarray,
                       path,
                       h_ref: float,
                       V_cruise: float,
                       gp: GuidanceParams):
    """Guidance output: the preferred *ground* velocity vector (3,).

    Horizontal direction points at the L1 reference point on the path; vertical
    component comes from the proportional altitude loop.
    """
    p2 = np.asarray(pos3, float)[:2]
    sp = float(np.linalg.norm(np.asarray(v_ground, float)[:2]))
    L1 = l1_distance(max(sp, 1.0), gp)
    ref = path.l1_reference(p2, L1)
    d = ref - p2
    dn = float(np.linalg.norm(d))
    if dn < 1e-6:
        dirn = np.asarray(v_ground, float)[:2]
        dn = max(float(np.linalg.norm(dirn)), 1e-6)
        d = dirn
    hdir = d / dn

    hdot = float(np.clip(gp.k_alt * (h_ref - pos3[2]),
                         -gp.climb_rate_max, gp.climb_rate_max))
    vh = np.sqrt(max(V_cruise ** 2 - hdot ** 2, (0.5 * V_cruise) ** 2))
    return np.array([hdir[0] * vh, hdir[1] * vh, hdot]), L1
