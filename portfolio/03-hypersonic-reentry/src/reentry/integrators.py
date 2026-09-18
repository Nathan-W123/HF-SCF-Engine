"""Time integrators.

Two independent integrators are provided:

* :func:`integrate_dop853` -- ``scipy.integrate.solve_ivp`` with the
  Dormand-Prince 8(5,3) explicit Runge-Kutta pair, adaptive step size, dense
  output and terminal events.  This is the production integrator.
* :func:`integrate_rk4` -- a plain fixed-step classical RK4.  It exists so that
  the convergence-order study in ``validation/val_04_energy_convergence.py`` can
  demonstrate the expected 4th-order error scaling on the *same* right-hand
  side, and so that the adaptive solver has an independent cross-check.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

__all__ = ["integrate_dop853", "integrate_rk4"]


def integrate_dop853(rhs, t_span, y0, events=None, rtol=1e-10, atol=1e-10, max_step=np.inf):
    return solve_ivp(
        rhs,
        t_span,
        np.asarray(y0, dtype=float),
        method="DOP853",
        rtol=rtol,
        atol=atol,
        dense_output=True,
        events=events,
        max_step=max_step,
    )


def integrate_rk4(rhs, t_span, y0, n_steps, stop=None):
    """Classical fixed-step RK4.

    Parameters
    ----------
    stop : callable or None
        ``stop(t, y) -> bool``; integration halts at the first step for which it
        returns True.

    Returns ``(t, y)`` with ``y`` of shape ``(n+1, len(y0))``.
    """
    t0, t1 = t_span
    dt = (t1 - t0) / n_steps
    y = np.asarray(y0, dtype=float).copy()
    ts = np.empty(n_steps + 1)
    ys = np.empty((n_steps + 1, y.size))
    ts[0] = t0
    ys[0] = y
    t = t0
    for i in range(n_steps):
        k1 = rhs(t, y)
        k2 = rhs(t + 0.5 * dt, y + 0.5 * dt * k1)
        k3 = rhs(t + 0.5 * dt, y + 0.5 * dt * k2)
        k4 = rhs(t + dt, y + dt * k3)
        y = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t = t0 + (i + 1) * dt
        ts[i + 1] = t
        ys[i + 1] = y
        if stop is not None and stop(t, y):
            return ts[: i + 2], ys[: i + 2]
    return ts, ys
