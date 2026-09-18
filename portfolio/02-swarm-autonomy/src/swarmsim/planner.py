"""Global path planning around circular no-fly zones: tangent visibility graph + A*.

Why a tangent graph
-------------------
For a workspace whose obstacles are discs, the shortest obstacle-free path
between two points is a sequence of straight segments that are *tangent* to the
discs, joined by arcs that hug the disc boundaries.  The set of candidate
segments is therefore finite and small: the tangent lines from each terminal
point to each disc, plus the internal and external bitangents between every
pair of discs.  Building that graph and running A* on it gives paths that are
optimal within this family, at a fraction of the node count of a grid search
and with no discretisation artefacts on the turns.

Graph construction
------------------
* Every no-fly disc is inflated by ``clearance`` (vehicle radius + separation
  buffer + a turn-radius allowance) before planning.
* Nodes: the start point, the goal point, and every tangent point on every
  inflated disc.
* Straight edges: a segment is admissible if it does not pass through the
  interior of any inflated disc (analytic segment/disc intersection test).
* Arc edges: two tangent points on the same disc are joined by the boundary
  arc.  Both directions are offered; each is rejected if a sampled set of
  points along the arc falls inside another inflated disc.
* Heuristic: straight-line distance, which is admissible because every edge
  length is at least the Euclidean distance between its endpoints.

Planning is done in the horizontal plane only.  Vertical separation is handled
by assigning each vehicle a cruise altitude and by the 3-D collision-avoidance
layer; the no-fly cylinders span the whole operating altitude band, so climbing
over them is not an option.

Fallback
--------
If the graph search fails (e.g. the goal is inside an inflated disc because the
clearance is large), the planner falls back to a direct start->goal path and
flags ``ok=False`` so the caller can report it rather than silently pretending
the route is clean.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from .config import NoFlyZone

_EPS = 1e-9
# Tangent edges touch the inflated disc exactly; allow 1 mm of float slop so a
# genuinely tangent segment is not reported as blocked.
TANGENT_EPS = 1e-2


@dataclass
class PlanResult:
    waypoints: np.ndarray    # (M, 2) horizontal waypoints, start .. goal
    length: float            # path length [m]
    ok: bool                 # False if the graph search failed and we fell back
    n_nodes: int
    n_edges: int


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------
def segment_circle_blocked(p0, p1, centres, radii, eps=TANGENT_EPS) -> np.ndarray:
    """True where segment p0->p1 penetrates the interior of each disc.

    Uses the exact point-to-segment distance; a segment merely *touching* a
    tangent circle (distance == radius) is not blocked, hence ``eps``.
    """
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    c = np.atleast_2d(np.asarray(centres, float))
    r = np.atleast_1d(np.asarray(radii, float))
    d = p1 - p0
    dd = float(d @ d)
    if dd < _EPS:
        dist = np.linalg.norm(c - p0, axis=1)
    else:
        t = np.clip(((c - p0) @ d) / dd, 0.0, 1.0)
        proj = p0[None, :] + t[:, None] * d[None, :]
        dist = np.linalg.norm(c - proj, axis=1)
    return dist < (r - eps)


def _tangent_points_from_point(p, c, r):
    """The two tangent points on circle (c, r) from an external point p."""
    v = c - p
    d = float(np.linalg.norm(v))
    if d <= r + 1e-9:
        return []
    # angle between p->c and p->tangent point
    alpha = np.arcsin(np.clip(r / d, -1.0, 1.0))
    base = np.arctan2(v[1], v[0])
    lt = np.sqrt(max(d * d - r * r, 0.0))
    out = []
    for sgn in (+1.0, -1.0):
        ang = base + sgn * alpha
        q = p + lt * np.array([np.cos(ang), np.sin(ang)])
        out.append(q)
    return out


def _bitangents(c0, r0, c1, r1):
    """External and internal bitangent point pairs between two circles."""
    d_vec = c1 - c0
    d = float(np.linalg.norm(d_vec))
    if d < 1e-9:
        return []
    base = np.arctan2(d_vec[1], d_vec[0])
    pairs = []
    # External tangents: signs (+,+) with cos(alpha) = (r0 - r1)/d
    ca = (r0 - r1) / d
    if abs(ca) <= 1.0:
        alpha = np.arccos(ca)
        for sgn in (+1.0, -1.0):
            a = base + sgn * alpha
            n = np.array([np.cos(a), np.sin(a)])
            pairs.append((c0 + r0 * n, c1 + r1 * n))
    # Internal tangents: require the circles to be separated
    ca = (r0 + r1) / d
    if abs(ca) <= 1.0:
        alpha = np.arccos(ca)
        for sgn in (+1.0, -1.0):
            a = base + sgn * alpha
            n = np.array([np.cos(a), np.sin(a)])
            pairs.append((c0 + r0 * n, c1 - r1 * n))
    return pairs


def _arc_free(c, r, a0, a1, direction, centres, radii, self_idx, n_samp=14):
    """Is the arc from angle a0 to a1 on circle ``self_idx`` obstacle-free?"""
    sweep = (a1 - a0) % (2 * np.pi) if direction > 0 else -((a0 - a1) % (2 * np.pi))
    if abs(sweep) < 1e-9:
        return True, 0.0
    ts = np.linspace(0.0, 1.0, n_samp)
    angs = a0 + sweep * ts
    pts = c[None, :] + r * np.stack([np.cos(angs), np.sin(angs)], axis=1)
    mask = np.ones(len(centres), bool)
    mask[self_idx] = False
    if mask.any():
        cc = centres[mask]
        rr = radii[mask]
        d = np.linalg.norm(pts[:, None, :] - cc[None, :, :], axis=2)
        if np.any(d < rr[None, :] - TANGENT_EPS):
            return False, 0.0
    return True, abs(sweep) * r


# --------------------------------------------------------------------------
# Planner
# --------------------------------------------------------------------------
class VisibilityGraphPlanner:
    """Tangent-visibility-graph planner over inflated circular no-fly zones."""

    def __init__(self, zones: Sequence[NoFlyZone], clearance: float = 80.0):
        self.zones = list(zones)
        self.clearance = float(clearance)
        if self.zones:
            self.centres = np.array([[z.x, z.y] for z in self.zones], float)
            self.radii = np.array([z.radius for z in self.zones], float) + clearance
        else:
            self.centres = np.zeros((0, 2))
            self.radii = np.zeros(0)

    # ---- node bookkeeping -------------------------------------------------
    def _build_nodes(self, start, goal):
        pts = [np.asarray(start, float), np.asarray(goal, float)]
        owner = [-1, -1]           # circle index owning the node (-1 = terminal)
        for i, (c, r) in enumerate(zip(self.centres, self.radii)):
            for p in (start, goal):
                for q in _tangent_points_from_point(np.asarray(p, float), c, r):
                    pts.append(q)
                    owner.append(i)
            for j in range(i + 1, len(self.centres)):
                for (qa, qb) in _bitangents(c, r, self.centres[j], self.radii[j]):
                    pts.append(qa)
                    owner.append(i)
                    pts.append(qb)
                    owner.append(j)
        return np.array(pts), np.array(owner)

    def plan(self, start, goal) -> PlanResult:
        start = np.asarray(start, float)[:2]
        goal = np.asarray(goal, float)[:2]

        # Trivial case: direct line is clear.
        if len(self.centres) == 0 or not segment_circle_blocked(
                start, goal, self.centres, self.radii).any():
            wp = np.stack([start, goal])
            return PlanResult(wp, float(np.linalg.norm(goal - start)), True, 2, 1)

        pts, owner = self._build_nodes(start, goal)
        n = len(pts)

        # --- edges ---------------------------------------------------------
        adj: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
        n_edges = 0

        # straight (tangent) edges -- vectorised over all node pairs and discs
        ia, ib = np.triu_indices(n, k=1)
        same = (owner[ia] != -1) & (owner[ia] == owner[ib])
        ia, ib = ia[~same], ib[~same]
        p0 = pts[ia]                                   # (P, 2)
        p1 = pts[ib]
        d = p1 - p0
        dd = np.einsum("ij,ij->i", d, d)
        dd_safe = np.maximum(dd, _EPS)
        # point-to-segment distance from every disc centre to every segment
        rel = self.centres[None, :, :] - p0[:, None, :]          # (P, Z, 2)
        t = np.clip(np.einsum("pzi,pi->pz", rel, d) / dd_safe[:, None], 0.0, 1.0)
        proj = p0[:, None, :] + t[..., None] * d[:, None, :]
        dist = np.linalg.norm(self.centres[None, :, :] - proj, axis=2)
        blocked = np.any(dist < (self.radii[None, :] - TANGENT_EPS), axis=1)
        keep = ~blocked
        ia, ib = ia[keep], ib[keep]
        w_all = np.sqrt(dd[keep])
        for a, b, w in zip(ia.tolist(), ib.tolist(), w_all.tolist()):
            adj[a].append((b, w))
            adj[b].append((a, w))
            n_edges += 1

        # arc edges, grouped per circle
        for i in range(len(self.centres)):
            idx = np.where(owner == i)[0]
            if len(idx) < 2:
                continue
            c, r = self.centres[i], self.radii[i]
            angs = np.arctan2(pts[idx, 1] - c[1], pts[idx, 0] - c[0])
            for ii in range(len(idx)):
                for jj in range(ii + 1, len(idx)):
                    for direction in (+1, -1):
                        ok, w = _arc_free(c, r, angs[ii], angs[jj], direction,
                                          self.centres, self.radii, i)
                        if ok and w > 0:
                            adj[idx[ii]].append((idx[jj], w))
                            adj[idx[jj]].append((idx[ii], w))
                            n_edges += 1
                            break

        # --- A* ------------------------------------------------------------
        h = np.linalg.norm(pts - goal[None, :], axis=1)
        g = np.full(n, np.inf)
        g[0] = 0.0
        prev = np.full(n, -1, int)
        pq = [(float(h[0]), 0)]
        closed = np.zeros(n, bool)
        found = False
        while pq:
            _, u = heapq.heappop(pq)
            if closed[u]:
                continue
            closed[u] = True
            if u == 1:
                found = True
                break
            for v, w in adj[u]:
                ng = g[u] + w
                if ng < g[v] - 1e-9:
                    g[v] = ng
                    prev[v] = u
                    heapq.heappush(pq, (float(ng + h[v]), int(v)))

        if not found:
            wp = np.stack([start, goal])
            return PlanResult(wp, float(np.linalg.norm(goal - start)), False, n, n_edges)

        chain = [1]
        while chain[-1] != 0:
            chain.append(int(prev[chain[-1]]))
        chain.reverse()
        raw = pts[chain]

        wp = self._densify_arcs(raw, np.asarray(owner)[chain])
        length = float(np.sum(np.linalg.norm(np.diff(wp, axis=0), axis=1)))
        return PlanResult(wp, length, True, n, n_edges)

    def _densify_arcs(self, raw, owners, step_deg=9.0):
        """Replace same-circle hops by a polyline along the boundary arc."""
        out = [raw[0]]
        for k in range(1, len(raw)):
            oa, ob = owners[k - 1], owners[k]
            if oa != -1 and oa == ob:
                c, r = self.centres[oa], self.radii[oa]
                a0 = np.arctan2(raw[k - 1][1] - c[1], raw[k - 1][0] - c[0])
                a1 = np.arctan2(raw[k][1] - c[1], raw[k][0] - c[0])
                d1 = (a1 - a0) % (2 * np.pi)
                d2 = d1 - 2 * np.pi
                # choose the direction that is obstacle-free (prefer shorter)
                cands = sorted([d1, d2], key=abs)
                sweep = cands[0]
                for cand in cands:
                    ok, _ = _arc_free(c, r, a0, a1, +1 if cand > 0 else -1,
                                      self.centres, self.radii, oa)
                    if ok:
                        sweep = cand
                        break
                m = max(2, int(abs(np.rad2deg(sweep)) / step_deg) + 1)
                # Place the polyline vertices on a slightly larger radius so the
                # chords between them stay *outside* the inflated disc rather
                # than cutting the corner by the chord sagitta.
                r_arc = r / np.cos(0.5 * abs(sweep) / m)
                for t in np.linspace(0, 1, m + 1):
                    a = a0 + sweep * t
                    out.append(c + r_arc * np.array([np.cos(a), np.sin(a)]))
            else:
                out.append(raw[k])
        return np.array(out)


def path_length(wp: np.ndarray) -> float:
    wp = np.asarray(wp, float)
    if len(wp) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(wp[:, :2], axis=0), axis=1)))
