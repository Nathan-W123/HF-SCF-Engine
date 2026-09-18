"""Numba kernel for the collision-avoidance hot loop.

The readable reference implementation lives in :mod:`swarmsim.avoidance` and
:mod:`swarmsim.qp`; this module re-expresses exactly the same arithmetic as
scalar loops so numba can compile it.  The two are checked against each other in
``tests/test_kernel_equivalence.py`` (agreement to 1e-9 on randomised swarm
states), so the kernel is an optimisation, not a second algorithm.

It matters: the QP carries up to ``2 * max_neighbours + n_zones`` rows and runs
once per vehicle per control step, which is tens of millions of inner iterations
over a full scenario sweep.  In pure Python/numpy that dominates the runtime.
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit
    HAVE_NUMBA = True
except Exception:                                    # pragma: no cover
    HAVE_NUMBA = False

    def njit(*a, **k):                               # type: ignore
        def deco(f):
            return f
        return deco if not a else a[0]


@njit(cache=True, fastmath=False)
def _orca_halfspace(px, py, pz, vx, vy, vz, r_comb, tau, tau_escape,
                    head_on_tol, out):
    """Write (nx, ny, nz, ux, uy, uz) for one pair into ``out`` (6,)."""
    dist_sq = px * px + py * py + pz * pz
    r2 = r_comb * r_comb

    if dist_sq > r2:
        wx = vx - px / tau
        wy = vy - py / tau
        wz = vz - pz / tau
        w_len_sq = wx * wx + wy * wy + wz * wz
        dot = wx * px + wy * py + wz * pz
        if dot < 0.0 and dot * dot > r2 * w_len_sq:
            w_len = np.sqrt(w_len_sq) if w_len_sq > 1e-18 else 1e-9
            nx = wx / w_len
            ny = wy / w_len
            nz = wz / w_len
            s = r_comb / tau - w_len
            out[0] = nx; out[1] = ny; out[2] = nz
            out[3] = s * nx; out[4] = s * ny; out[5] = s * nz
            return
        a = dist_sq
        bq = px * vx + py * vy + pz * vz
        cx = py * vz - pz * vy
        cy = pz * vx - px * vz
        cz = px * vy - py * vx
        cr2 = cx * cx + cy * cy + cz * cz
        den = dist_sq - r2
        if den < 1e-9:
            den = 1e-9
        c = (vx * vx + vy * vy + vz * vz) - cr2 / den
        disc = bq * bq - a * c
        if disc < 0.0:
            disc = 0.0
        aa = a if a > 1e-12 else 1e-12
        t = (bq + np.sqrt(disc)) / aa
        wwx = vx - t * px
        wwy = vy - t * py
        wwz = vz - t * pz
        ww_len = np.sqrt(wwx * wwx + wwy * wwy + wwz * wwz)
        v_len = np.sqrt(vx * vx + vy * vy + vz * vz)
        thr = head_on_tol * (v_len if v_len > 1e-6 else 1e-6)
        if ww_len < thr:
            # Head-on degeneracy -> rules-of-the-air tie-break (turn right),
            # tilted so the escaped velocity lands on the true cone boundary.
            # See avoidance._right_of_way_normal for the derivation.
            dlen = np.sqrt(dist_sq)
            hx = py
            hy = -px
            nh = np.sqrt(hx * hx + hy * hy)
            if nh < 1e-9:
                ex = 1.0; ey = 0.0; ez = 0.0
            else:
                ex = hx / nh; ey = hy / nh; ez = 0.0
            cb = -r_comb / (dlen if dlen > 1e-9 else 1e-9)
            if cb < -1.0:
                cb = -1.0
            sb = np.sqrt(1.0 - cb * cb) if cb * cb < 1.0 else 0.0
            nx = cb * (px / dlen) + sb * ex
            ny = cb * (py / dlen) + sb * ey
            nz = cb * (pz / dlen) + sb * ez
            nn = np.sqrt(nx * nx + ny * ny + nz * nz)
            if nn < 1e-12:
                nn = 1e-12
            nx /= nn; ny /= nn; nz /= nn
            # n is chosen, not derived from ww, so use the general escape form.
            out[0] = nx; out[1] = ny; out[2] = nz
            out[3] = t * (px + r_comb * nx) - vx
            out[4] = t * (py + r_comb * ny) - vy
            out[5] = t * (pz + r_comb * nz) - vz
            return
        nx = wwx / ww_len
        ny = wwy / ww_len
        nz = wwz / ww_len
        s = r_comb * t - ww_len
        out[0] = nx; out[1] = ny; out[2] = nz
        out[3] = s * nx; out[4] = s * ny; out[5] = s * nz
        return

    # already inside the combined radius
    wx = vx - px / tau_escape
    wy = vy - py / tau_escape
    wz = vz - pz / tau_escape
    w_len = np.sqrt(wx * wx + wy * wy + wz * wz)
    if w_len < 1e-12:
        out[0] = 0.0; out[1] = 0.0; out[2] = 1.0
        out[3] = 0.0; out[4] = 0.0; out[5] = 0.0
        return
    nx = wx / w_len; ny = wy / w_len; nz = wz / w_len
    s = r_comb / tau_escape - w_len
    out[0] = nx; out[1] = ny; out[2] = nz
    out[3] = s * nx; out[4] = s * ny; out[5] = s * nz


@njit(cache=True, fastmath=False)
def _max_closure(h, alpha, accel):
    if h < 0.0:
        h = 0.0
    a = alpha * h
    b = np.sqrt(2.0 * accel * h)
    return a if a < b else b


@njit(cache=True, fastmath=False)
def _hildreth(A, b, rho_inv, hard, Minv, v_pref, m, sweeps, tol, v_out):
    """Solve min |W(v-v_pref)|^2 + sum rho_k s_k^2  s.t.  A v >= b - s."""
    lam = np.zeros(m)
    x = np.zeros(3)
    AM = np.empty((m, 3))
    denom = np.empty(m)
    c = np.empty(m)
    for i in range(m):
        s = 0.0
        for d in range(3):
            AM[i, d] = A[i, d] * Minv[d]
            s += A[i, d] * A[i, d] * Minv[d]
        denom[i] = s + rho_inv[i]
        if denom[i] < 1e-12:
            denom[i] = 1e-12
        c[i] = b[i] - (A[i, 0] * v_pref[0] + A[i, 1] * v_pref[1]
                       + A[i, 2] * v_pref[2])

    for _ in range(sweeps):
        delta_max = 0.0
        for i in range(m):
            ax = A[i, 0] * x[0] + A[i, 1] * x[1] + A[i, 2] * x[2]
            g = c[i] - ax - lam[i] * rho_inv[i]
            new = lam[i] + g / denom[i]
            if new < 0.0:
                new = 0.0
            d = new - lam[i]
            if d != 0.0:
                x[0] += d * AM[i, 0]
                x[1] += d * AM[i, 1]
                x[2] += d * AM[i, 2]
                lam[i] = new
                ad = d if d > 0.0 else -d
                if ad > delta_max:
                    delta_max = ad
        if delta_max < tol:
            break

    for d in range(3):
        v_out[d] = v_pref[d] + x[d]

    viol = -1e30
    for i in range(m):
        if hard[i]:
            r = b[i] - (A[i, 0] * v_out[0] + A[i, 1] * v_out[1]
                        + A[i, 2] * v_out[2])
            if r > viol:
                viol = r
    if viol < 0.0:
        viol = 0.0
    return viol


@njit(cache=True, fastmath=False)
def avoid_kernel(pos, vel, v_pref, coop, active,
                 zone_c, zone_r, zone_lo, zone_hi,
                 r_comb, tau, tau_escape, sense_range, max_nb, head_on_tol,
                 r_safe, cbf_alpha, cbf_accel, zone_margin, zone_sense,
                 orca_rho, orca_rho_urg, wz, sweeps, tol, v_clamp,
                 fb_cands,
                 v_out, n_con, infeas, fallback):
    n = pos.shape[0]
    nz = zone_c.shape[0]
    max_rows = 2 * max_nb + nz
    A = np.empty((max_rows, 3))
    b = np.empty(max_rows)
    rho_inv = np.empty(max_rows)
    hard = np.empty(max_rows, np.bool_)
    hs = np.empty(6)
    Minv = np.empty(3)
    Minv[0] = 1.0
    Minv[1] = 1.0
    Minv[2] = 1.0 / (wz * wz)
    nb_d = np.empty(n)
    nb_j = np.empty(n, np.int64)

    for i in range(n):
        for d in range(3):
            v_out[i, d] = v_pref[i, d]
        if not active[i]:
            continue
        if not coop[i]:
            continue

        # ---- nearest neighbours within the sensing range ----------------
        cnt = 0
        for j in range(n):
            if j == i:
                continue
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dz = pos[i, 2] - pos[j, 2]
            dd = np.sqrt(dx * dx + dy * dy + dz * dz)
            if dd < sense_range:
                nb_d[cnt] = dd
                nb_j[cnt] = j
                cnt += 1
        k_use = cnt if cnt < max_nb else max_nb
        # partial selection sort for the k smallest
        for a_ in range(k_use):
            best = a_
            for b_ in range(a_ + 1, cnt):
                if nb_d[b_] < nb_d[best]:
                    best = b_
            if best != a_:
                td = nb_d[a_]; nb_d[a_] = nb_d[best]; nb_d[best] = td
                tj = nb_j[a_]; nb_j[a_] = nb_j[best]; nb_j[best] = tj

        m = 0
        # ---- layer 1: ORCA (soft) ---------------------------------------
        for a_ in range(k_use):
            j = nb_j[a_]
            _orca_halfspace(pos[j, 0] - pos[i, 0], pos[j, 1] - pos[i, 1],
                            pos[j, 2] - pos[i, 2],
                            vel[i, 0] - vel[j, 0], vel[i, 1] - vel[j, 1],
                            vel[i, 2] - vel[j, 2],
                            r_comb, tau, tau_escape, head_on_tol, hs)
            share = 0.5 if coop[j] else 1.0
            rhs = 0.0
            for d in range(3):
                A[m, d] = hs[d]
                rhs += hs[d] * (vel[i, d] + share * hs[3 + d])
            b[m] = rhs
            urg = r_comb / (nb_d[a_] if nb_d[a_] > 1e-6 else 1e-6)
            if urg > 1.0:
                urg = 1.0
            rho_inv[m] = 1.0 / (orca_rho * (1.0 + orca_rho_urg * urg * urg))
            hard[m] = False
            m += 1

        # ---- layer 2: pairwise CBF (hard) -------------------------------
        for a_ in range(k_use):
            j = nb_j[a_]
            dij = nb_d[a_]
            if dij < 1e-6:
                continue
            rhs = 0.0
            for d in range(3):
                nij = (pos[i, d] - pos[j, d]) / dij
                A[m, d] = nij
                rhs += nij * vel[j, d]
            b[m] = rhs - _max_closure(dij - r_safe, cbf_alpha, cbf_accel)
            rho_inv[m] = 0.0
            hard[m] = True
            m += 1

        # ---- layer 3: no-fly zones (hard) -------------------------------
        for z in range(nz):
            if pos[i, 2] <= zone_lo[z] - 60.0 or pos[i, 2] >= zone_hi[z] + 60.0:
                continue
            rx = pos[i, 0] - zone_c[z, 0]
            ry = pos[i, 1] - zone_c[z, 1]
            dzn = np.sqrt(rx * rx + ry * ry)
            if dzn > zone_sense + zone_r[z] or dzn < 1e-6:
                continue
            A[m, 0] = rx / dzn
            A[m, 1] = ry / dzn
            A[m, 2] = 0.0
            b[m] = -_max_closure(dzn - (zone_r[z] + zone_margin),
                                 cbf_alpha, cbf_accel)
            rho_inv[m] = 0.0
            hard[m] = True
            m += 1

        n_con[i] = m
        if m == 0:
            continue

        viol = _hildreth(A, b, rho_inv, hard, Minv, v_pref[i], m, sweeps, tol,
                         v_out[i])
        if viol > 1e-6:
            infeas[i] = True
            fallback[i] = True
            # max-margin over the candidate fan, restricted to the hard rows
            best_w = -1e30
            best_k = 0
            best_dist = 1e30
            for cidx in range(fb_cands.shape[0]):
                worst = 1e30
                for r_ in range(m):
                    if not hard[r_]:
                        continue
                    mm = (A[r_, 0] * fb_cands[cidx, 0]
                          + A[r_, 1] * fb_cands[cidx, 1]
                          + A[r_, 2] * fb_cands[cidx, 2]) - b[r_]
                    if mm < worst:
                        worst = mm
                if worst > best_w + 1e-9:
                    best_w = worst
                    best_k = cidx
                    dx = fb_cands[cidx, 0] - v_pref[i, 0]
                    dy = fb_cands[cidx, 1] - v_pref[i, 1]
                    dz = fb_cands[cidx, 2] - v_pref[i, 2]
                    best_dist = np.sqrt(dx * dx + dy * dy + dz * dz)
                elif worst > best_w - 1e-9:
                    dx = fb_cands[cidx, 0] - v_pref[i, 0]
                    dy = fb_cands[cidx, 1] - v_pref[i, 1]
                    dz = fb_cands[cidx, 2] - v_pref[i, 2]
                    dd = np.sqrt(dx * dx + dy * dy + dz * dz)
                    if dd < best_dist:
                        best_k = cidx
                        best_dist = dd
            for d in range(3):
                v_out[i, d] = fb_cands[best_k, d]

        sp = np.sqrt(v_out[i, 0] ** 2 + v_out[i, 1] ** 2 + v_out[i, 2] ** 2)
        if sp > v_clamp:
            sc = v_clamp / sp
            for d in range(3):
                v_out[i, d] *= sc
