"""Barnes-Hut octree gravity, JIT-compiled with numba.

Design notes
------------
*Build*: top-down, iterative (explicit task stack, no recursion).  At each node
the particle index range is partitioned into the eight octants with a counting
sort, which is a cache-friendly O(N) pass per level.  Only non-empty octants
become nodes, and they are allocated contiguously, so a child always has a
higher node index than its parent.  That lets the centre-of-mass pass be a
single reverse sweep over the node array.

*Traversal*: stackless, using the classic Barnes-Hut ``more``/``next`` pointer
pair.  ``more[n]`` is the first child, ``next[n]`` is the node to visit after
skipping n's entire subtree.  Accepting a node means jumping to ``next``;
opening it means jumping to ``more``.  No per-particle stack means the walk
parallelises over particles with ``prange`` with no allocation in the hot loop.

*Opening criterion*: a node is accepted when

    (size + delta)^2 < theta^2 * d^2

where ``size`` is the full cube width, ``d`` the distance from the sink
particle to the node centre of mass and ``delta`` the offset between the node's
geometric centre and its centre of mass.  The ``delta`` term is Barnes' (1994)
correction; it prevents the pathological case of a node whose centre of mass
sits near the sink particle, which the plain s/d criterion accepts.

*Softening*: Plummer, with a per-particle softening length.  A pair
interaction uses ``h^2 = eps_i^2 + eps_j^2``; a particle-node interaction uses
the node's mass-weighted mean ``eps^2``.  The pairwise form is symmetric, so
the direct-summation limit conserves momentum exactly.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange, get_num_threads

from .units import G

LEAF_SIZE = 32
MAX_DEPTH = 40


# --------------------------------------------------------------------------
# Tree construction
# --------------------------------------------------------------------------
@njit(cache=True)
def _build_tree(pos, mass, eps2, order, scratch, octbuf,
                node_more, node_next, node_start, node_count,
                node_cx, node_cy, node_cz, node_size,
                node_mass, node_mx, node_my, node_mz, node_eps2, node_delta,
                leaf_size, root_cx, root_cy, root_cz, root_half):
    n = pos.shape[0]
    max_nodes = node_more.shape[0]

    # task stack: (node, start, end, cx, cy, cz, half, depth)
    cap = max_nodes + 64
    st_node = np.empty(cap, np.int64)
    st_s = np.empty(cap, np.int64)
    st_e = np.empty(cap, np.int64)
    st_cx = np.empty(cap, np.float64)
    st_cy = np.empty(cap, np.float64)
    st_cz = np.empty(cap, np.float64)
    st_h = np.empty(cap, np.float64)
    st_d = np.empty(cap, np.int64)

    cnt = np.empty(8, np.int64)
    off = np.empty(8, np.int64)
    ptr = np.empty(8, np.int64)
    kid = np.empty(8, np.int64)

    nnodes = 1
    node_next[0] = -1
    sp = 0
    st_node[0] = 0; st_s[0] = 0; st_e[0] = n
    st_cx[0] = root_cx; st_cy[0] = root_cy; st_cz[0] = root_cz
    st_h[0] = root_half; st_d[0] = 0
    sp = 1

    while sp > 0:
        sp -= 1
        nd = st_node[sp]; s = st_s[sp]; e = st_e[sp]
        cx = st_cx[sp]; cy = st_cy[sp]; cz = st_cz[sp]
        h = st_h[sp]; depth = st_d[sp]

        node_cx[nd] = cx; node_cy[nd] = cy; node_cz[nd] = cz
        node_size[nd] = 2.0 * h
        node_start[nd] = s
        node_count[nd] = e - s

        if (e - s) <= leaf_size or depth >= MAX_DEPTH:
            node_more[nd] = -1
            continue

        for k in range(8):
            cnt[k] = 0
        for k in range(s, e):
            i = order[k]
            o = 0
            if pos[i, 0] > cx:
                o += 1
            if pos[i, 1] > cy:
                o += 2
            if pos[i, 2] > cz:
                o += 4
            octbuf[k] = o
            cnt[o] += 1

        acc = s
        for k in range(8):
            off[k] = acc
            ptr[k] = acc
            acc += cnt[k]
        for k in range(s, e):
            i = order[k]
            o = octbuf[k]
            scratch[ptr[o]] = i
            ptr[o] += 1
        for k in range(s, e):
            order[k] = scratch[k]

        # allocate non-empty children contiguously
        nkids = 0
        for k in range(8):
            if cnt[k] > 0:
                kid[nkids] = k
                nkids += 1
        if nnodes + nkids > max_nodes:      # should not happen; degrade to leaf
            node_more[nd] = -1
            continue

        first = nnodes
        nnodes += nkids
        node_more[nd] = first
        parent_next = node_next[nd]
        hh = 0.5 * h
        for j in range(nkids):
            c = first + j
            node_next[c] = (first + j + 1) if j < nkids - 1 else parent_next
            o = kid[j]
            ccx = cx + (hh if (o & 1) else -hh)
            ccy = cy + (hh if (o & 2) else -hh)
            ccz = cz + (hh if (o & 4) else -hh)
            st_node[sp] = c
            st_s[sp] = off[o]
            st_e[sp] = off[o] + cnt[o]
            st_cx[sp] = ccx; st_cy[sp] = ccy; st_cz[sp] = ccz
            st_h[sp] = hh; st_d[sp] = depth + 1
            sp += 1

    # ---- centre of mass, reverse sweep (children have higher indices) ----
    for nd in range(nnodes - 1, -1, -1):
        m = 0.0; mx = 0.0; my = 0.0; mz = 0.0; me = 0.0
        if node_more[nd] == -1:
            s = node_start[nd]
            e = s + node_count[nd]
            for k in range(s, e):
                i = order[k]
                mi = mass[i]
                m += mi
                mx += mi * pos[i, 0]
                my += mi * pos[i, 1]
                mz += mi * pos[i, 2]
                me += mi * eps2[i]
        else:
            c = node_more[nd]
            stop = node_next[nd]
            while c != stop and c != -1:
                mc = node_mass[c]
                m += mc
                mx += mc * node_mx[c]
                my += mc * node_my[c]
                mz += mc * node_mz[c]
                me += mc * node_eps2[c]
                c = node_next[c]
        node_mass[nd] = m
        if m > 0.0:
            inv = 1.0 / m
            node_mx[nd] = mx * inv
            node_my[nd] = my * inv
            node_mz[nd] = mz * inv
            node_eps2[nd] = me * inv
        else:
            node_mx[nd] = node_cx[nd]
            node_my[nd] = node_cy[nd]
            node_mz[nd] = node_cz[nd]
            node_eps2[nd] = 0.0
        dx = node_mx[nd] - node_cx[nd]
        dy = node_my[nd] - node_cy[nd]
        dz = node_mz[nd] - node_cz[nd]
        node_delta[nd] = np.sqrt(dx * dx + dy * dy + dz * dz)

    return nnodes


class Octree:
    """Container for the flattened octree arrays."""

    __slots__ = ("order", "rank", "more", "next", "start", "count",
                 "cx", "cy", "cz", "size", "mass", "mx", "my", "mz",
                 "eps2", "delta", "nnodes", "groups")


def build_tree(pos, mass, eps2, leaf_size=LEAF_SIZE):
    """Build a Barnes-Hut octree over ``pos`` (N,3) float64."""
    n = pos.shape[0]
    lo = pos.min(axis=0)
    hi = pos.max(axis=0)
    centre = 0.5 * (lo + hi)
    half = 0.5 * float(np.max(hi - lo))
    half = max(half * 1.0000001, 1e-6)

    max_nodes = max(64, int(2.2 * n / max(1, leaf_size)) * 8 + 64)
    t = Octree()
    t.order = np.arange(n, dtype=np.int64)
    scratch = np.empty(n, np.int64)
    octbuf = np.empty(n, np.int64)
    t.more = np.empty(max_nodes, np.int64)
    t.next = np.empty(max_nodes, np.int64)
    t.start = np.zeros(max_nodes, np.int64)
    t.count = np.zeros(max_nodes, np.int64)
    t.cx = np.zeros(max_nodes); t.cy = np.zeros(max_nodes); t.cz = np.zeros(max_nodes)
    t.size = np.zeros(max_nodes)
    t.mass = np.zeros(max_nodes)
    t.mx = np.zeros(max_nodes); t.my = np.zeros(max_nodes); t.mz = np.zeros(max_nodes)
    t.eps2 = np.zeros(max_nodes)
    t.delta = np.zeros(max_nodes)

    t.nnodes = _build_tree(
        pos, mass, eps2, t.order, scratch, octbuf,
        t.more, t.next, t.start, t.count,
        t.cx, t.cy, t.cz, t.size,
        t.mass, t.mx, t.my, t.mz, t.eps2, t.delta,
        leaf_size, centre[0], centre[1], centre[2], half)

    t.rank = np.empty(n, np.int64)
    t.rank[t.order] = np.arange(n, dtype=np.int64)
    nn = t.nnodes
    t.groups = np.flatnonzero(t.more[:nn] == -1).astype(np.int64)
    return t


# --------------------------------------------------------------------------
# Force evaluation
# --------------------------------------------------------------------------
@njit(parallel=True, fastmath=True, cache=True)
def _walk(pos, mass, eps2, order, rank,
          node_more, node_next, node_start, node_count,
          node_size, node_mass, node_mx, node_my, node_mz,
          node_eps2, node_delta, theta2, acc, pot, want_pot):
    n = pos.shape[0]
    for i in prange(n):
        xi = pos[i, 0]; yi = pos[i, 1]; zi = pos[i, 2]
        ei = eps2[i]
        ri = rank[i]
        ax = 0.0; ay = 0.0; az = 0.0; ph = 0.0
        nd = 0
        while nd != -1:
            dx = node_mx[nd] - xi
            dy = node_my[nd] - yi
            dz = node_mz[nd] - zi
            r2 = dx * dx + dy * dy + dz * dz
            sz = node_size[nd] + node_delta[nd]
            open_it = sz * sz >= theta2 * r2
            if node_more[nd] == -1:
                s = node_start[nd]
                c = node_count[nd]
                contains = (ri >= s) and (ri < s + c)
                if open_it or contains:
                    for k in range(s, s + c):
                        j = order[k]
                        if j == i:
                            continue
                        ddx = pos[j, 0] - xi
                        ddy = pos[j, 1] - yi
                        ddz = pos[j, 2] - zi
                        rr2 = ddx * ddx + ddy * ddy + ddz * ddz + ei + eps2[j]
                        inv = 1.0 / np.sqrt(rr2)
                        mj = mass[j]
                        w = mj * inv * inv * inv
                        ax += w * ddx; ay += w * ddy; az += w * ddz
                        ph -= mj * inv
                else:
                    rr2 = r2 + ei + node_eps2[nd]
                    inv = 1.0 / np.sqrt(rr2)
                    mn = node_mass[nd]
                    w = mn * inv * inv * inv
                    ax += w * dx; ay += w * dy; az += w * dz
                    ph -= mn * inv
                nd = node_next[nd]
            else:
                if open_it:
                    nd = node_more[nd]
                else:
                    rr2 = r2 + ei + node_eps2[nd]
                    inv = 1.0 / np.sqrt(rr2)
                    mn = node_mass[nd]
                    w = mn * inv * inv * inv
                    ax += w * dx; ay += w * dy; az += w * dz
                    ph -= mn * inv
                    nd = node_next[nd]
        acc[i, 0] = G * ax
        acc[i, 1] = G * ay
        acc[i, 2] = G * az
        if want_pot:
            pot[i] = G * ph




# --------------------------------------------------------------------------
# Group (cell-based) traversal -- the production kernel
# --------------------------------------------------------------------------
# Walking the tree once per *particle* spends most of its time chasing pointers
# and mispredicting branches.  Walking once per *leaf cell* and reusing the
# resulting interaction list for every particle in that cell amortises the
# traversal over ~LEAF_SIZE sinks and turns the force sum into two flat,
# branch-free loops that the compiler can vectorise.  The opening criterion is
# applied to the group's bounding sphere,
#
#     (size + delta)^2 < theta^2 * (d - r_group)^2 ,
#
# which is strictly more conservative than the per-particle test, so the
# group walk is at least as accurate as the per-particle walk at the same
# theta (verified in validation/val_tree_accuracy.py).

@njit(parallel=True, fastmath=True, cache=True)
def _walk_groups(pos, mass, eps2, order, groups,
                 node_more, node_next, node_start, node_count,
                 node_size, node_mx, node_my, node_mz,
                 node_mass, node_eps2, node_delta,
                 theta2, acc, pot, want_pot, nchunks, max_nodes, max_direct):
    ngroups = groups.shape[0]
    for ci in prange(nchunks):
        lm = np.empty(max_nodes, np.float64)
        lx = np.empty(max_nodes, np.float64)
        ly = np.empty(max_nodes, np.float64)
        lz = np.empty(max_nodes, np.float64)
        le = np.empty(max_nodes, np.float64)
        dlist = np.empty(max_direct, np.int64)

        for gi in range(ci, ngroups, nchunks):
            g = groups[gi]
            gs = node_start[g]
            gc = node_count[g]

            # bounding sphere of the group
            xlo = pos[order[gs], 0]; xhi = xlo
            ylo = pos[order[gs], 1]; yhi = ylo
            zlo = pos[order[gs], 2]; zhi = zlo
            for k in range(gs + 1, gs + gc):
                i = order[k]
                v = pos[i, 0]
                if v < xlo: xlo = v
                if v > xhi: xhi = v
                v = pos[i, 1]
                if v < ylo: ylo = v
                if v > yhi: yhi = v
                v = pos[i, 2]
                if v < zlo: zlo = v
                if v > zhi: zhi = v
            gx = 0.5 * (xlo + xhi); gy = 0.5 * (ylo + yhi); gz = 0.5 * (zlo + zhi)
            rg = 0.5 * np.sqrt((xhi - xlo) ** 2 + (yhi - ylo) ** 2 + (zhi - zlo) ** 2)

            nn = 0
            nd_cnt = 0
            nd = 0
            while nd != -1:
                dx = node_mx[nd] - gx
                dy = node_my[nd] - gy
                dz = node_mz[nd] - gz
                d = np.sqrt(dx * dx + dy * dy + dz * dz) - rg
                sz = node_size[nd] + node_delta[nd]
                accept = (d > 0.0) and (sz * sz < theta2 * d * d)
                if node_more[nd] == -1:
                    if accept and nn < max_nodes:
                        lm[nn] = node_mass[nd]
                        lx[nn] = node_mx[nd]
                        ly[nn] = node_my[nd]
                        lz[nn] = node_mz[nd]
                        le[nn] = node_eps2[nd]
                        nn += 1
                    else:
                        s = node_start[nd]
                        for k in range(s, s + node_count[nd]):
                            if nd_cnt < max_direct:
                                dlist[nd_cnt] = order[k]
                                nd_cnt += 1
                    nd = node_next[nd]
                else:
                    if accept:
                        if nn < max_nodes:
                            lm[nn] = node_mass[nd]
                            lx[nn] = node_mx[nd]
                            ly[nn] = node_my[nd]
                            lz[nn] = node_mz[nd]
                            le[nn] = node_eps2[nd]
                            nn += 1
                        nd = node_next[nd]
                    else:
                        nd = node_more[nd]

            for k in range(gs, gs + gc):
                i = order[k]
                xi = pos[i, 0]; yi = pos[i, 1]; zi = pos[i, 2]
                ei = eps2[i]
                ax = 0.0; ay = 0.0; az = 0.0; ph = 0.0
                for t in range(nn):
                    dx = lx[t] - xi
                    dy = ly[t] - yi
                    dz = lz[t] - zi
                    r2 = dx * dx + dy * dy + dz * dz + ei + le[t]
                    inv = 1.0 / np.sqrt(r2)
                    w = lm[t] * inv * inv * inv
                    ax += w * dx; ay += w * dy; az += w * dz
                    ph -= lm[t] * inv
                for t in range(nd_cnt):
                    j = dlist[t]
                    dx = pos[j, 0] - xi
                    dy = pos[j, 1] - yi
                    dz = pos[j, 2] - zi
                    r2 = dx * dx + dy * dy + dz * dz + ei + eps2[j]
                    inv = 1.0 / np.sqrt(r2)
                    mj = mass[j]
                    w = mj * inv * inv * inv
                    ax += w * dx; ay += w * dy; az += w * dz
                    ph -= mj * inv
                    if j == i:
                        # remove the self term added above (branch hoisted out
                        # of the arithmetic so the loop still vectorises)
                        inv0 = 1.0 / np.sqrt(2.0 * ei)
                        ph += mj * inv0
                acc[i, 0] = G * ax
                acc[i, 1] = G * ay
                acc[i, 2] = G * az
                if want_pot:
                    pot[i] = G * ph


def accel_tree(pos, mass, eps2, theta=0.6, leaf_size=LEAF_SIZE,
               want_potential=False, tree=None, out=None, groups=True):
    """Barnes-Hut acceleration (and optionally potential per unit mass).

    Returns ``(acc, pot)``; ``pot`` is ``None`` when ``want_potential`` is
    False.  ``pot[i]`` is the specific potential Phi(x_i) excluding the
    particle's self-term, so the total potential energy is
    ``0.5 * sum(m_i * pot_i)``.
    """
    n = pos.shape[0]
    if tree is None:
        tree = build_tree(pos, mass, eps2, leaf_size=leaf_size)
    acc = np.empty((n, 3)) if out is None else out
    pot = np.empty(n) if want_potential else np.empty(1)
    if groups:
        nchunks = max(1, int(get_num_threads()))
        _walk_groups(pos, mass, eps2, tree.order, tree.groups,
                     tree.more, tree.next, tree.start, tree.count,
                     tree.size, tree.mx, tree.my, tree.mz,
                     tree.mass, tree.eps2, tree.delta,
                     theta * theta, acc, pot, bool(want_potential),
                     nchunks, int(tree.nnodes), int(n))
        return acc, (pot if want_potential else None)
    _walk(pos, mass, eps2, tree.order, tree.rank,
          tree.more, tree.next, tree.start, tree.count,
          tree.size, tree.mass, tree.mx, tree.my, tree.mz,
          tree.eps2, tree.delta, theta * theta, acc, pot,
          bool(want_potential))
    return acc, (pot if want_potential else None)


# --------------------------------------------------------------------------
# Reference direct summation (validation / small N)
# --------------------------------------------------------------------------
@njit(parallel=True, fastmath=True, cache=True)
def _direct(pos, mass, eps2, acc, pot, want_pot):
    n = pos.shape[0]
    for i in prange(n):
        xi = pos[i, 0]; yi = pos[i, 1]; zi = pos[i, 2]
        ei = eps2[i]
        ax = 0.0; ay = 0.0; az = 0.0; ph = 0.0
        for j in range(n):
            if j == i:
                continue
            dx = pos[j, 0] - xi
            dy = pos[j, 1] - yi
            dz = pos[j, 2] - zi
            r2 = dx * dx + dy * dy + dz * dz + ei + eps2[j]
            inv = 1.0 / np.sqrt(r2)
            mj = mass[j]
            w = mj * inv * inv * inv
            ax += w * dx; ay += w * dy; az += w * dz
            ph -= mj * inv
        acc[i, 0] = G * ax; acc[i, 1] = G * ay; acc[i, 2] = G * az
        if want_pot:
            pot[i] = G * ph


def accel_direct(pos, mass, eps2, want_potential=False):
    """Exact O(N^2) softened acceleration; the Barnes-Hut reference."""
    n = pos.shape[0]
    acc = np.empty((n, 3))
    pot = np.empty(n) if want_potential else np.empty(1)
    _direct(pos, mass, eps2, acc, pot, bool(want_potential))
    return acc, (pot if want_potential else None)


@njit(parallel=True, fastmath=True, cache=True)
def _direct_subset(pos, mass, eps2, idx, acc):
    """Exact acceleration for a subset of sinks against all sources."""
    n = pos.shape[0]
    ns = idx.shape[0]
    for t in prange(ns):
        i = idx[t]
        xi = pos[i, 0]; yi = pos[i, 1]; zi = pos[i, 2]
        ei = eps2[i]
        ax = 0.0; ay = 0.0; az = 0.0
        for j in range(n):
            if j == i:
                continue
            dx = pos[j, 0] - xi
            dy = pos[j, 1] - yi
            dz = pos[j, 2] - zi
            r2 = dx * dx + dy * dy + dz * dz + ei + eps2[j]
            inv = 1.0 / np.sqrt(r2)
            w = mass[j] * inv * inv * inv
            ax += w * dx; ay += w * dy; az += w * dz
        acc[t, 0] = G * ax; acc[t, 1] = G * ay; acc[t, 2] = G * az


def accel_direct_subset(pos, mass, eps2, idx):
    acc = np.empty((idx.shape[0], 3))
    _direct_subset(pos, mass, eps2, np.asarray(idx, np.int64), acc)
    return acc
