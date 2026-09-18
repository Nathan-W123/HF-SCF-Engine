"""Tiny dense QP used by the collision-avoidance layer.

Problem
-------
    minimise    1/2 ||W (v - v_pref)||^2  +  1/2 sum_k rho_k s_k^2
    subject to  A_k . v  >=  b_k - s_k ,      s_k >= 0

on a 3-vector ``v`` with a handful (< ~30) of linear inequality rows.  Rows with
``rho_k = inf`` are **hard** (no slack); rows with a finite ``rho_k`` are **soft**
and may be violated at a quadratic cost.  This is exactly what the avoidance
layer needs: the control-barrier rows and the no-fly-zone rows are hard, while
the ORCA rows are soft and weighted by how urgent each encounter is.

Why soft ORCA rows matter
-------------------------
In a dense swarm the intersection of ~20 ORCA half-spaces is frequently empty.
The textbook remedy -- throw all of them away and fall back to a much weaker
rule -- loses every constraint including the ones that were easy to satisfy.
With quadratic slacks the solver instead degrades gracefully: it keeps the
urgent half-spaces essentially tight and gives ground on the distant ones.

Method
------
Hildreth dual coordinate ascent.  With ``x = v - v_pref``, ``M = W^-2`` and
``c = b - A v_pref`` the stationarity conditions give ``x = M A^T lambda`` and
``s_k = lambda_k / rho_k``, and the per-coordinate Newton step is

    delta   = (c_k - A_k x - lambda_k / rho_k) / (A_k M A_k^T + 1 / rho_k)
    lambda_k <- max(0, lambda_k + delta)
    x        <- x + (delta_applied) * M A_k

which reduces to the classical hard-constraint Hildreth update as
``rho_k -> inf``.  Every sweep is monotone in the dual objective, it needs no
factorisation, and the soft rows make the dual bounded whenever the hard rows
alone are feasible.
"""

from __future__ import annotations

import numpy as np


def solve_qp(v_pref: np.ndarray,
             A: np.ndarray,
             b: np.ndarray,
             weights: np.ndarray | None = None,
             rho: np.ndarray | None = None,
             sweeps: int = 90,
             tol: float = 1e-7):
    """Return ``(v, hard_feasible, max_hard_violation)``.

    ``weights`` are the diagonal of ``W`` (default ones).  ``rho`` gives the
    per-row slack stiffness; ``np.inf`` (the default for every row) means hard.
    """
    v_pref = np.asarray(v_pref, float)
    n = v_pref.size
    if A is None or len(A) == 0:
        return v_pref.copy(), True, 0.0

    A = np.atleast_2d(np.asarray(A, float))
    b = np.atleast_1d(np.asarray(b, float))
    m = A.shape[0]

    if weights is None:
        Minv = np.ones(n)
    else:
        w = np.asarray(weights, float)
        Minv = 1.0 / np.maximum(w, 1e-9) ** 2

    if rho is None:
        inv_rho = np.zeros(m)
        hard = np.ones(m, bool)
    else:
        r = np.asarray(rho, float)
        inv_rho = np.where(np.isfinite(r), 1.0 / np.maximum(r, 1e-12), 0.0)
        hard = ~np.isfinite(r)

    c = b - A @ v_pref
    AM = A * Minv[None, :]
    denom = np.einsum("ij,ij->i", AM, A) + inv_rho
    denom = np.maximum(denom, 1e-12)

    lam = np.zeros(m)
    x = np.zeros(n)

    for _ in range(sweeps):
        delta_max = 0.0
        for i in range(m):
            g = c[i] - A[i] @ x - lam[i] * inv_rho[i]
            new = lam[i] + g / denom[i]
            if new < 0.0:
                new = 0.0
            d = new - lam[i]
            if d != 0.0:
                x += d * AM[i]
                lam[i] = new
                delta_max = max(delta_max, abs(d))
        if delta_max < tol:
            break

    v = v_pref + x
    if hard.any():
        viol = float(np.max(b[hard] - A[hard] @ v))
    else:
        viol = 0.0
    return v, bool(viol <= 1e-6), max(viol, 0.0)


def max_margin_velocity(A: np.ndarray, b: np.ndarray, v_pref: np.ndarray,
                        candidates: np.ndarray):
    """Fallback for an infeasible *hard* constraint set.

    Picks, from a fixed candidate set of velocities, the one that maximises the
    worst constraint margin ``min_k (A_k v - b_k)``, breaking ties towards the
    preferred velocity.  This is the "least unsafe" action rather than a
    solution, and the caller records when it was used.
    """
    A = np.atleast_2d(A)
    b = np.atleast_1d(b)
    margins = (candidates @ A.T) - b[None, :]
    worst = margins.min(axis=1)
    best = worst.max()
    near = np.where(worst >= best - 1e-9)[0]
    if len(near) > 1:
        d = np.linalg.norm(candidates[near] - v_pref[None, :], axis=1)
        k = near[int(np.argmin(d))]
    else:
        k = int(near[0])
    return candidates[k], float(best)
