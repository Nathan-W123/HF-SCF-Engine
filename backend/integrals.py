"""
Gaussian-type orbital integral evaluation using the McMurchie-Davidson algorithm.

References:
  McMurchie & Davidson, J. Comput. Phys. 26, 218 (1978)
  Helgaker, Jørgensen, Olsen - Molecular Electronic-Structure Theory (2000)

Performance:
  When numba is available the hot path (_eri_contracted, _eri_prim, Boys, E/R
  tables) is JIT-compiled to native code.  Pure-Python fallback is used otherwise.

  Hot-path improvements over the original implementation:
    1. boys():  erf + upward recurrence / Taylor series  (replaces hyp1f1, ~30x faster)
    2. E table: fully iterative, shared across t/u/v in one call  (no dict overhead)
    3. R table: fully iterative, no recursion  (no dict / call-stack overhead)
    4. Numba JIT: native code for the 4-deep primitive contraction loop
"""
from __future__ import annotations

import numpy as np
from math import pi, sqrt, exp, erf

# ── Numba JIT setup ────────────────────────────────────────────────────────────

try:
    from numba import njit as _njit_base
    def _njit(fn):
        return _njit_base(fn, cache=True)   # compile-once, cache to disk
    _HAS_NUMBA = True
except ImportError:                         # pragma: no cover
    def _njit(fn):                          # identity decorator
        return fn
    _HAS_NUMBA = False


# ── Boys function (scalar — used by one-electron integrals) ───────────────────

def boys(n: int, x: float) -> float:
    """
    Boys function F_n(x) = ∫₀¹ t^{2n} exp(−x t²) dt.

    Taylor series for x < 1 (avoids cancellation in upward recurrence);
    erf-based F_0 + upward recurrence for x ≥ 1.
    """
    if x < 1.0:
        total = 0.0
        term = 1.0 / (2 * n + 1)
        for k in range(1, 25):
            total += term
            term *= -x / k / (2 * n + 2 * k + 1)
            if abs(term) < 1e-15:
                break
        return total + term
    f = sqrt(pi) / (2.0 * sqrt(x)) * erf(sqrt(x))
    if n == 0:
        return f
    ex = exp(-x)
    inv2x = 0.5 / x
    for m in range(n):
        f = ((2 * m + 1) * f - ex) * inv2x
    return f


# ── Numba-compiled core ───────────────────────────────────────────────────────

@_njit
def _boys_array(n_max: int, x: float):
    """Compute F_0(x) … F_{n_max}(x) in one pass."""
    fn = np.zeros(n_max + 1)
    if x < 1.0:
        for n in range(n_max + 1):
            total = 0.0
            term = 1.0 / (2 * n + 1)
            for k in range(1, 25):
                total += term
                term *= -x / k / (2 * n + 2 * k + 1)
                if abs(term) < 1e-15:
                    break
            fn[n] = total + term
    else:
        fn[0] = sqrt(pi) / (2.0 * sqrt(x)) * erf(sqrt(x))
        if n_max > 0:
            ex = exp(-x)
            inv2x = 0.5 / x
            for m in range(n_max):
                fn[m + 1] = ((2 * m + 1) * fn[m] - ex) * inv2x
    return fn


@_njit
def _E_table(l1: int, l2: int, Qx: float, a: float, b: float):
    """
    Iterative McMurchie-Davidson E coefficient table.
    Returns E[i, j, t] for 0≤i≤l1, 0≤j≤l2, 0≤t≤i+j.
    """
    p = a + b
    q_ab = a * b / p
    XPA = -b * Qx / p
    XPB =  a * Qx / p
    inv2p = 0.5 / p

    max_t = l1 + l2
    E = np.zeros((l1 + 1, l2 + 1, max_t + 1))
    E[0, 0, 0] = exp(-q_ab * Qx * Qx)

    for i in range(l1 + 1):
        for j in range(l2 + 1):
            if i == 0 and j == 0:
                continue
            for t in range(i + j + 1):
                val = 0.0
                if i > 0:
                    if t > 0:
                        val += inv2p * E[i - 1, j, t - 1]
                    val += XPA * E[i - 1, j, t]
                    if t + 1 <= i - 1 + j:
                        val += (t + 1) * E[i - 1, j, t + 1]
                else:
                    if t > 0:
                        val += inv2p * E[i, j - 1, t - 1]
                    val += XPB * E[i, j - 1, t]
                    if t + 1 <= i + j - 1:
                        val += (t + 1) * E[i, j - 1, t + 1]
                E[i, j, t] = val
    return E


@_njit
def _R_table(n_max: int, t_max: int, u_max: int, v_max: int,
             p: float, PQx: float, PQy: float, PQz: float, fn):
    """
    Iterative Hermite Coulomb integral table R[n, t, u, v].
    Fills from high n downward so each entry only reads from n+1 (already done).
    """
    # Shape: (n_max+2) to allow access at n+1 when n==n_max (returns 0)
    R = np.zeros((n_max + 2, t_max + 2, u_max + 2, v_max + 2))

    # Base cases R[n, 0, 0, 0] = (-2p)^n * F_n
    pow_m2p = 1.0
    for n in range(n_max + 1):
        R[n, 0, 0, 0] = pow_m2p * fn[n]
        pow_m2p *= -2.0 * p

    # Build higher (t, u, v) using level n+1 (already computed / zero)
    for n in range(n_max, -1, -1):
        for t in range(t_max + 1):
            for u in range(u_max + 1):
                for v in range(v_max + 1):
                    if t == 0 and u == 0 and v == 0:
                        continue
                    if t > 0:
                        val = PQx * R[n + 1, t - 1, u, v]
                        if t >= 2:
                            val += (t - 1) * R[n + 1, t - 2, u, v]
                        R[n, t, u, v] = val
                    elif u > 0:
                        val = PQy * R[n + 1, t, u - 1, v]
                        if u >= 2:
                            val += (u - 1) * R[n + 1, t, u - 2, v]
                        R[n, t, u, v] = val
                    else:
                        val = PQz * R[n + 1, t, u, v - 1]
                        if v >= 2:
                            val += (v - 1) * R[n + 1, t, u, v - 2]
                        R[n, t, u, v] = val
    return R


@_njit
def _eri_prim_core(
    a1: float, l1: int, m1: int, n1: int, Ax: float, Ay: float, Az: float,
    a2: float, l2: int, m2: int, n2: int, Bx: float, By: float, Bz: float,
    a3: float, l3: int, m3: int, n3: int, Cx: float, Cy: float, Cz: float,
    a4: float, l4: int, m4: int, n4: int, Dx: float, Dy: float, Dz: float,
) -> float:
    """Primitive ERI (φ₁φ₂|φ₃φ₄) — fully iterative, JIT-compiled."""
    p = a1 + a2
    q = a3 + a4
    alpha = p * q / (p + q)
    Px = (a1 * Ax + a2 * Bx) / p
    Py = (a1 * Ay + a2 * By) / p
    Pz = (a1 * Az + a2 * Bz) / p
    Qx = (a3 * Cx + a4 * Dx) / q
    Qy = (a3 * Cy + a4 * Dy) / q
    Qz = (a3 * Cz + a4 * Dz) / q
    PQx = Px - Qx
    PQy = Py - Qy
    PQz = Pz - Qz

    n_max = l1 + l2 + l3 + l4 + m1 + m2 + m3 + m4 + n1 + n2 + n3 + n4
    x = alpha * (PQx * PQx + PQy * PQy + PQz * PQz)
    fn = _boys_array(n_max, x)

    Ex12 = _E_table(l1, l2, Ax - Bx, a1, a2)
    Ey12 = _E_table(m1, m2, Ay - By, a1, a2)
    Ez12 = _E_table(n1, n2, Az - Bz, a1, a2)
    Ex34 = _E_table(l3, l4, Cx - Dx, a3, a4)
    Ey34 = _E_table(m3, m4, Cy - Dy, a3, a4)
    Ez34 = _E_table(n3, n4, Cz - Dz, a3, a4)

    t_max = l1 + l2 + l3 + l4
    u_max = m1 + m2 + m3 + m4
    v_max = n1 + n2 + n3 + n4
    R = _R_table(n_max, t_max, u_max, v_max, alpha, PQx, PQy, PQz, fn)

    total = 0.0
    for t in range(l1 + l2 + 1):
        et = Ex12[l1, l2, t]
        for u in range(m1 + m2 + 1):
            eu = Ey12[m1, m2, u]
            for v in range(n1 + n2 + 1):
                ev = Ez12[n1, n2, v]
                if et * eu * ev == 0.0:
                    continue
                for tau in range(l3 + l4 + 1):
                    etau = Ex34[l3, l4, tau]
                    for nu in range(m3 + m4 + 1):
                        enu = Ey34[m3, m4, nu]
                        for phi in range(n3 + n4 + 1):
                            ephi = Ez34[n3, n4, phi]
                            sign = 1.0 if (tau + nu + phi) % 2 == 0 else -1.0
                            r = R[0, t + tau, u + nu, v + phi]
                            total += et * eu * ev * sign * etau * enu * ephi * r

    return 2.0 * pi ** 2.5 / (p * q * sqrt(p + q)) * total


@_njit
def _eri_contracted(
    exps1, coeffs1, norms1, l1: int, m1: int, n1: int, Ax: float, Ay: float, Az: float,
    exps2, coeffs2, norms2, l2: int, m2: int, n2: int, Bx: float, By: float, Bz: float,
    exps3, coeffs3, norms3, l3: int, m3: int, n3: int, Cx: float, Cy: float, Cz: float,
    exps4, coeffs4, norms4, l4: int, m4: int, n4: int, Dx: float, Dy: float, Dz: float,
) -> float:
    """
    Contracted ERI — four nested loops over primitives, JIT-compiled.

    E tables for pairs (1,2) and (3,4) are precomputed for all np1*np2 and
    np3*np4 exponent combinations before the inner loops, avoiding 6x redundant
    recomputation that would otherwise occur inside the quartet loop.
    """
    np1 = len(exps1); np2 = len(exps2)
    np3 = len(exps3); np4 = len(exps4)

    n_max = l1+l2+l3+l4 + m1+m2+m3+m4 + n1+n2+n3+n4
    t_max = l1+l2+l3+l4
    u_max = m1+m2+m3+m4
    v_max = n1+n2+n3+n4
    pi25 = 2.0 * pi ** 2.5

    # ── Precompute E tables for pair (1,2) ────────────────────────────────────
    # E12x[i1,i2,i,j,t] = _E_table(l1,l2,...)[i,j,t]
    E12x = np.zeros((np1, np2, l1+1, l2+1, l1+l2+1))
    E12y = np.zeros((np1, np2, m1+1, m2+1, m1+m2+1))
    E12z = np.zeros((np1, np2, n1+1, n2+1, n1+n2+1))
    Px_arr = np.zeros((np1, np2)); Py_arr = np.zeros((np1, np2)); Pz_arr = np.zeros((np1, np2))
    p_arr  = np.zeros((np1, np2))
    for i1 in range(np1):
        for i2 in range(np2):
            p = exps1[i1] + exps2[i2]
            p_arr[i1, i2] = p
            Px_arr[i1, i2] = (exps1[i1]*Ax + exps2[i2]*Bx) / p
            Py_arr[i1, i2] = (exps1[i1]*Ay + exps2[i2]*By) / p
            Pz_arr[i1, i2] = (exps1[i1]*Az + exps2[i2]*Bz) / p
            E12x[i1, i2] = _E_table(l1, l2, Ax-Bx, exps1[i1], exps2[i2])
            E12y[i1, i2] = _E_table(m1, m2, Ay-By, exps1[i1], exps2[i2])
            E12z[i1, i2] = _E_table(n1, n2, Az-Bz, exps1[i1], exps2[i2])

    # ── Precompute E tables for pair (3,4) ────────────────────────────────────
    E34x = np.zeros((np3, np4, l3+1, l4+1, l3+l4+1))
    E34y = np.zeros((np3, np4, m3+1, m4+1, m3+m4+1))
    E34z = np.zeros((np3, np4, n3+1, n4+1, n3+n4+1))
    Qx_arr = np.zeros((np3, np4)); Qy_arr = np.zeros((np3, np4)); Qz_arr = np.zeros((np3, np4))
    q_arr  = np.zeros((np3, np4))
    for i3 in range(np3):
        for i4 in range(np4):
            q = exps3[i3] + exps4[i4]
            q_arr[i3, i4] = q
            Qx_arr[i3, i4] = (exps3[i3]*Cx + exps4[i4]*Dx) / q
            Qy_arr[i3, i4] = (exps3[i3]*Cy + exps4[i4]*Dy) / q
            Qz_arr[i3, i4] = (exps3[i3]*Cz + exps4[i4]*Dz) / q
            E34x[i3, i4] = _E_table(l3, l4, Cx-Dx, exps3[i3], exps4[i4])
            E34y[i3, i4] = _E_table(m3, m4, Cy-Dy, exps3[i3], exps4[i4])
            E34z[i3, i4] = _E_table(n3, n4, Cz-Dz, exps3[i3], exps4[i4])

    # ── Four-index contraction ────────────────────────────────────────────────
    val = 0.0
    for i1 in range(np1):
        for i2 in range(np2):
            p  = p_arr[i1, i2]
            Px = Px_arr[i1, i2]; Py = Py_arr[i1, i2]; Pz = Pz_arr[i1, i2]
            pf12 = norms1[i1] * norms2[i2] * coeffs1[i1] * coeffs2[i2]

            for i3 in range(np3):
                for i4 in range(np4):
                    q     = q_arr[i3, i4]
                    alpha = p * q / (p + q)
                    PQx   = Px - Qx_arr[i3, i4]
                    PQy   = Py - Qy_arr[i3, i4]
                    PQz   = Pz - Qz_arr[i3, i4]

                    x  = alpha * (PQx*PQx + PQy*PQy + PQz*PQz)
                    fn = _boys_array(n_max, x)
                    R  = _R_table(n_max, t_max, u_max, v_max, alpha, PQx, PQy, PQz, fn)

                    pf = pf12 * norms3[i3] * norms4[i4] * coeffs3[i3] * coeffs4[i4]
                    scale = pf * pi25 / (p * q * sqrt(p + q))

                    inner = 0.0
                    for t in range(l1+l2+1):
                        et = E12x[i1, i2, l1, l2, t]
                        for u in range(m1+m2+1):
                            eu = E12y[i1, i2, m1, m2, u]
                            for v in range(n1+n2+1):
                                ev = E12z[i1, i2, n1, n2, v]
                                if et * eu * ev == 0.0:
                                    continue
                                for tau in range(l3+l4+1):
                                    etau = E34x[i3, i4, l3, l4, tau]
                                    for nu in range(m3+m4+1):
                                        enu = E34y[i3, i4, m3, m4, nu]
                                        for phi in range(n3+n4+1):
                                            ephi = E34z[i3, i4, n3, n4, phi]
                                            sign = 1.0 if (tau+nu+phi) % 2 == 0 else -1.0
                                            inner += (et * eu * ev * sign
                                                      * etau * enu * ephi
                                                      * R[0, t+tau, u+nu, v+phi])
                    val += scale * inner
    return val


# ── Helpers still used by one-electron integrals ──────────────────────────────

def _dfact(n: int) -> int:
    if n <= 0:
        return 1
    return n * _dfact(n - 2)


def norm_const(alpha: float, l: int, m: int, n: int) -> float:
    return sqrt(
        (4.0 * alpha) ** (l + m + n)
        * (2.0 * alpha / pi) ** 1.5
        / (_dfact(2 * l - 1) * _dfact(2 * m - 1) * _dfact(2 * n - 1))
    )


def _E(i, j, t, Qx, a, b, memo):
    """Recursive E (kept for one-electron integrals; uses shared memo dict)."""
    key = (i, j, t)
    if key in memo:
        return memo[key]
    if t < 0 or t > i + j:
        return 0.0
    p = a + b
    q = a * b / p
    if i == j == t == 0:
        v = exp(-q * Qx * Qx)
        memo[key] = v
        return v
    if i > 0:
        XPA = -b * Qx / p
        v = (
            (1.0 / (2 * p)) * _E(i - 1, j, t - 1, Qx, a, b, memo)
            + XPA            * _E(i - 1, j, t,     Qx, a, b, memo)
            + (t + 1)        * _E(i - 1, j, t + 1, Qx, a, b, memo)
        )
    else:
        XPB = a * Qx / p
        v = (
            (1.0 / (2 * p)) * _E(i, j - 1, t - 1, Qx, a, b, memo)
            + XPB            * _E(i, j - 1, t,     Qx, a, b, memo)
            + (t + 1)        * _E(i, j - 1, t + 1, Qx, a, b, memo)
        )
    memo[key] = v
    return v


def _R_scalar(n, t, u, v, p, PQ, memo, fn):
    """Recursive R (used by one-electron integrals only)."""
    key = (n, t, u, v)
    if key in memo:
        return memo[key]
    if t < 0 or u < 0 or v < 0:
        return 0.0
    if t == u == v == 0:
        val = (-2.0 * p) ** n * fn[n]
        memo[key] = val
        return val
    if t > 0:
        val = (t - 1) * _R_scalar(n+1, t-2, u, v, p, PQ, memo, fn) + PQ[0] * _R_scalar(n+1, t-1, u, v, p, PQ, memo, fn)
    elif u > 0:
        val = (u - 1) * _R_scalar(n+1, t, u-2, v, p, PQ, memo, fn) + PQ[1] * _R_scalar(n+1, t, u-1, v, p, PQ, memo, fn)
    else:
        val = (v - 1) * _R_scalar(n+1, t, u, v-2, p, PQ, memo, fn) + PQ[2] * _R_scalar(n+1, t, u, v-1, p, PQ, memo, fn)
    memo[key] = val
    return val


# ── Primitive one-electron integrals (Python, called N²×K² times — fast enough) ─

def _overlap_prim(a1, lmn1, A, a2, lmn2, B):
    l1, m1, n1 = lmn1; l2, m2, n2 = lmn2
    p = a1 + a2; Q = A - B
    Sx = _E(l1, l2, 0, Q[0], a1, a2, {}) * sqrt(pi / p)
    Sy = _E(m1, m2, 0, Q[1], a1, a2, {}) * sqrt(pi / p)
    Sz = _E(n1, n2, 0, Q[2], a1, a2, {}) * sqrt(pi / p)
    return Sx * Sy * Sz


def _kinetic_prim(a1, lmn1, A, a2, lmn2, B):
    l1, m1, n1 = lmn1; l2, m2, n2 = lmn2
    p = a1 + a2; rt_p = sqrt(pi / p); Q = A - B

    def S1d(i, j, dim):
        return _E(i, j, 0, Q[dim], a1, a2, {}) * rt_p

    def T1d(i, j, dim):
        Qd = Q[dim]
        return (
            -0.5 * j * (j - 1) * _E(i, j - 2, 0, Qd, a1, a2, {})
            + a2 * (2 * j + 1)  * _E(i, j,     0, Qd, a1, a2, {})
            - 2.0 * a2 ** 2     * _E(i, j + 2, 0, Qd, a1, a2, {})
        ) * rt_p

    Sx, Sy, Sz = S1d(l1,l2,0), S1d(m1,m2,1), S1d(n1,n2,2)
    Tx, Ty, Tz = T1d(l1,l2,0), T1d(m1,m2,1), T1d(n1,n2,2)
    return Tx*Sy*Sz + Sx*Ty*Sz + Sx*Sy*Tz


def _nuclear_prim(a1, lmn1, A, a2, lmn2, B, C):
    l1, m1, n1 = lmn1; l2, m2, n2 = lmn2
    p = a1 + a2
    P = (a1 * A + a2 * B) / p
    PC = P - C
    n_max = l1 + l2 + m1 + m2 + n1 + n2
    x = p * float(np.dot(PC, PC))
    fn_list = [boys(k, x) for k in range(n_max + 1)]
    total = 0.0
    rm = {}; mex = {}; mey = {}; mez = {}
    for t in range(l1 + l2 + 1):
        et = _E(l1, l2, t, A[0]-B[0], a1, a2, mex)
        for u in range(m1 + m2 + 1):
            eu = _E(m1, m2, u, A[1]-B[1], a1, a2, mey)
            for v in range(n1 + n2 + 1):
                ev = _E(n1, n2, v, A[2]-B[2], a1, a2, mez)
                total += et * eu * ev * _R_scalar(0, t, u, v, p, PC, rm, fn_list)
    return 2.0 * pi / p * total


# ── Contracted integrals (public API) ─────────────────────────────────────────

def overlap(bf1, bf2) -> float:
    val = 0.0
    for a, da, Na in zip(bf1.exponents, bf1.coefficients, bf1.norms):
        for b, db, Nb in zip(bf2.exponents, bf2.coefficients, bf2.norms):
            val += Na * Nb * da * db * _overlap_prim(a, bf1.lmn, bf1.center,
                                                      b, bf2.lmn, bf2.center)
    return val


def kinetic(bf1, bf2) -> float:
    val = 0.0
    for a, da, Na in zip(bf1.exponents, bf1.coefficients, bf1.norms):
        for b, db, Nb in zip(bf2.exponents, bf2.coefficients, bf2.norms):
            val += Na * Nb * da * db * _kinetic_prim(a, bf1.lmn, bf1.center,
                                                      b, bf2.lmn, bf2.center)
    return val


def nuclear(bf1, bf2, C: np.ndarray, Z: float) -> float:
    val = 0.0
    for a, da, Na in zip(bf1.exponents, bf1.coefficients, bf1.norms):
        for b, db, Nb in zip(bf2.exponents, bf2.coefficients, bf2.norms):
            val += Na * Nb * da * db * _nuclear_prim(a, bf1.lmn, bf1.center,
                                                      b, bf2.lmn, bf2.center, C)
    return -Z * val


def eri(bf1, bf2, bf3, bf4) -> float:
    """
    Two-electron repulsion integral (μν|λσ).
    Routes through the numba JIT-compiled _eri_contracted when numba is available.
    """
    l1, m1, n1 = bf1.lmn; Ax, Ay, Az = bf1.center
    l2, m2, n2 = bf2.lmn; Bx, By, Bz = bf2.center
    l3, m3, n3 = bf3.lmn; Cx, Cy, Cz = bf3.center
    l4, m4, n4 = bf4.lmn; Dx, Dy, Dz = bf4.center
    return _eri_contracted(
        bf1.exponents, bf1.coefficients, bf1.norms, l1, m1, n1, Ax, Ay, Az,
        bf2.exponents, bf2.coefficients, bf2.norms, l2, m2, n2, Bx, By, Bz,
        bf3.exponents, bf3.coefficients, bf3.norms, l3, m3, n3, Cx, Cy, Cz,
        bf4.exponents, bf4.coefficients, bf4.norms, l4, m4, n4, Dx, Dy, Dz,
    )


# ── 3-centre and 2-centre ERIs for RI-JK density fitting ──────────────────────

@_njit
def _eri_3c_contracted(
    exps1, coeffs1, norms1, l1: int, m1: int, n1: int, Ax: float, Ay: float, Az: float,
    exps2, coeffs2, norms2, l2: int, m2: int, n2: int, Bx: float, By: float, Bz: float,
    exps3, coeffs3, norms3, l3: int, m3: int, n3: int, Cx: float, Cy: float, Cz: float,
) -> float:
    """
    3-centre 2-electron integral (μν|P) — Numba JIT.

    Bra: contracted Gaussian pair (1,2) at A,B  (same McMurchie-Davidson bra as _eri_contracted).
    Ket: single contracted auxiliary function (3) at C.

    The ket E table is computed with b=0 and XAB=0 — this correctly expands a single
    Gaussian (r-C)^{l} exp(-α|r-C|²) into Hermite Gaussians at its own centre.
    No K_CD prefactor arises (no product of two ket Gaussians).
    """
    np1 = len(exps1); np2 = len(exps2); np3 = len(exps3)
    n_max = l1+l2+l3 + m1+m2+m3 + n1+n2+n3
    t_max = l1+l2+l3
    u_max = m1+m2+m3
    v_max = n1+n2+n3
    pi25  = 2.0 * pi ** 2.5

    # Precompute bra E tables (one per primitive pair i1,i2) — identical to _eri_contracted
    E12x   = np.zeros((np1, np2, l1+1, l2+1, l1+l2+1))
    E12y   = np.zeros((np1, np2, m1+1, m2+1, m1+m2+1))
    E12z   = np.zeros((np1, np2, n1+1, n2+1, n1+n2+1))
    Px_arr = np.zeros((np1, np2)); Py_arr = np.zeros((np1, np2)); Pz_arr = np.zeros((np1, np2))
    p_arr  = np.zeros((np1, np2))
    for i1 in range(np1):
        for i2 in range(np2):
            p = exps1[i1] + exps2[i2]
            p_arr[i1, i2]  = p
            Px_arr[i1, i2] = (exps1[i1]*Ax + exps2[i2]*Bx) / p
            Py_arr[i1, i2] = (exps1[i1]*Ay + exps2[i2]*By) / p
            Pz_arr[i1, i2] = (exps1[i1]*Az + exps2[i2]*Bz) / p
            E12x[i1, i2]   = _E_table(l1, l2, Ax-Bx, exps1[i1], exps2[i2])
            E12y[i1, i2]   = _E_table(m1, m2, Ay-By, exps1[i1], exps2[i2])
            E12z[i1, i2]   = _E_table(n1, n2, Az-Bz, exps1[i1], exps2[i2])

    # Precompute ket E tables — one per ket primitive i3.
    # _E_table(l, 0, 0.0, α, 0.0) expands a single-centre Gaussian of AM l
    # into Hermite Gaussians: E[l,0,t] for t=0…l.
    E3x_arr = np.zeros((np3, l3+1))
    E3y_arr = np.zeros((np3, m3+1))
    E3z_arr = np.zeros((np3, n3+1))
    for i3 in range(np3):
        a3 = exps3[i3]
        ex = _E_table(l3, 0, 0.0, a3, 0.0)
        for t in range(l3+1):
            E3x_arr[i3, t] = ex[l3, 0, t]
        ey = _E_table(m3, 0, 0.0, a3, 0.0)
        for t in range(m3+1):
            E3y_arr[i3, t] = ey[m3, 0, t]
        ez = _E_table(n3, 0, 0.0, a3, 0.0)
        for t in range(n3+1):
            E3z_arr[i3, t] = ez[n3, 0, t]

    val = 0.0
    for i1 in range(np1):
        for i2 in range(np2):
            p   = p_arr[i1, i2]
            Px  = Px_arr[i1, i2]; Py = Py_arr[i1, i2]; Pz = Pz_arr[i1, i2]
            pf12 = norms1[i1] * norms2[i2] * coeffs1[i1] * coeffs2[i2]
            for i3 in range(np3):
                q     = exps3[i3]          # ket is a single function — no q = a3+a4
                alpha = p * q / (p + q)
                PQx   = Px - Cx;  PQy = Py - Cy;  PQz = Pz - Cz
                x     = alpha * (PQx*PQx + PQy*PQy + PQz*PQz)
                fn    = _boys_array(n_max, x)
                R     = _R_table(n_max, t_max, u_max, v_max, alpha, PQx, PQy, PQz, fn)
                pf    = pf12 * norms3[i3] * coeffs3[i3]
                scale = pf * pi25 / (p * q * sqrt(p + q))
                inner = 0.0
                for t in range(l1+l2+1):
                    et = E12x[i1, i2, l1, l2, t]
                    for u in range(m1+m2+1):
                        eu = E12y[i1, i2, m1, m2, u]
                        for v in range(n1+n2+1):
                            ev = E12z[i1, i2, n1, n2, v]
                            if et * eu * ev == 0.0:
                                continue
                            for tau in range(l3+1):
                                etau = E3x_arr[i3, tau]
                                for nu in range(m3+1):
                                    enu = E3y_arr[i3, nu]
                                    for phi in range(n3+1):
                                        ephi = E3z_arr[i3, phi]
                                        sign = 1.0 if (tau+nu+phi) % 2 == 0 else -1.0
                                        inner += (et * eu * ev * sign
                                                  * etau * enu * ephi
                                                  * R[0, t+tau, u+nu, v+phi])
                val += scale * inner
    return val


@_njit
def _eri_2c_contracted(
    exps1, coeffs1, norms1, l1: int, m1: int, n1: int, Ax: float, Ay: float, Az: float,
    exps2, coeffs2, norms2, l2: int, m2: int, n2: int, Bx: float, By: float, Bz: float,
) -> float:
    """
    2-centre 2-electron Coulomb metric integral (P|Q) — Numba JIT.

    Both electrons carry a single auxiliary function (no Gaussian product pairs).
    The E table for each function is computed with b=0, XAB=0 (single-centre expansion).
    scale = 2π^{5/2} / (a1 × a2 × √(a1+a2)); no K_AB or K_CD prefactors.
    """
    np1 = len(exps1); np2 = len(exps2)
    n_max = l1+l2 + m1+m2 + n1+n2
    t_max = l1+l2
    u_max = m1+m2
    v_max = n1+n2
    pi25  = 2.0 * pi ** 2.5
    ABx = Ax - Bx;  ABy = Ay - By;  ABz = Az - Bz

    E1x_arr = np.zeros((np1, l1+1))
    E1y_arr = np.zeros((np1, m1+1))
    E1z_arr = np.zeros((np1, n1+1))
    for i1 in range(np1):
        a1 = exps1[i1]
        ex = _E_table(l1, 0, 0.0, a1, 0.0)
        for t in range(l1+1): E1x_arr[i1, t] = ex[l1, 0, t]
        ey = _E_table(m1, 0, 0.0, a1, 0.0)
        for t in range(m1+1): E1y_arr[i1, t] = ey[m1, 0, t]
        ez = _E_table(n1, 0, 0.0, a1, 0.0)
        for t in range(n1+1): E1z_arr[i1, t] = ez[n1, 0, t]

    E2x_arr = np.zeros((np2, l2+1))
    E2y_arr = np.zeros((np2, m2+1))
    E2z_arr = np.zeros((np2, n2+1))
    for i2 in range(np2):
        a2 = exps2[i2]
        ex = _E_table(l2, 0, 0.0, a2, 0.0)
        for t in range(l2+1): E2x_arr[i2, t] = ex[l2, 0, t]
        ey = _E_table(m2, 0, 0.0, a2, 0.0)
        for t in range(m2+1): E2y_arr[i2, t] = ey[m2, 0, t]
        ez = _E_table(n2, 0, 0.0, a2, 0.0)
        for t in range(n2+1): E2z_arr[i2, t] = ez[n2, 0, t]

    val = 0.0
    for i1 in range(np1):
        a1  = exps1[i1]
        pf1 = norms1[i1] * coeffs1[i1]
        for i2 in range(np2):
            a2    = exps2[i2]
            rho   = a1 * a2 / (a1 + a2)
            x     = rho * (ABx*ABx + ABy*ABy + ABz*ABz)
            fn    = _boys_array(n_max, x)
            R     = _R_table(n_max, t_max, u_max, v_max, rho, ABx, ABy, ABz, fn)
            pf    = pf1 * norms2[i2] * coeffs2[i2]
            scale = pf * pi25 / (a1 * a2 * sqrt(a1 + a2))
            inner = 0.0
            for t in range(l1+1):
                et = E1x_arr[i1, t]
                for u in range(m1+1):
                    eu = E1y_arr[i1, u]
                    for v in range(n1+1):
                        ev = E1z_arr[i1, v]
                        if et * eu * ev == 0.0:
                            continue
                        for tau in range(l2+1):
                            etau = E2x_arr[i2, tau]
                            for nu in range(m2+1):
                                enu = E2y_arr[i2, nu]
                                for phi in range(n2+1):
                                    ephi = E2z_arr[i2, phi]
                                    sign = 1.0 if (tau+nu+phi) % 2 == 0 else -1.0
                                    inner += (et * eu * ev * sign
                                              * etau * enu * ephi
                                              * R[0, t+tau, u+nu, v+phi])
            val += scale * inner
    return val


def eri_3c(bf1, bf2, bf_aux) -> float:
    """3-centre 2-electron integral (μν|P)."""
    l1, m1, n1 = bf1.lmn;    Ax, Ay, Az = bf1.center
    l2, m2, n2 = bf2.lmn;    Bx, By, Bz = bf2.center
    l3, m3, n3 = bf_aux.lmn; Cx, Cy, Cz = bf_aux.center
    return _eri_3c_contracted(
        bf1.exponents,   bf1.coefficients,   bf1.norms,   l1, m1, n1, Ax, Ay, Az,
        bf2.exponents,   bf2.coefficients,   bf2.norms,   l2, m2, n2, Bx, By, Bz,
        bf_aux.exponents,bf_aux.coefficients,bf_aux.norms,l3, m3, n3, Cx, Cy, Cz,
    )


def eri_2c(bf1, bf2) -> float:
    """2-centre 2-electron Coulomb metric integral (P|Q)."""
    l1, m1, n1 = bf1.lmn; Ax, Ay, Az = bf1.center
    l2, m2, n2 = bf2.lmn; Bx, By, Bz = bf2.center
    return _eri_2c_contracted(
        bf1.exponents, bf1.coefficients, bf1.norms, l1, m1, n1, Ax, Ay, Az,
        bf2.exponents, bf2.coefficients, bf2.norms, l2, m2, n2, Bx, By, Bz,
    )
