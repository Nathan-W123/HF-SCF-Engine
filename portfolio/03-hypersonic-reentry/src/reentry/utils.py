"""Small numerical helpers shared by the validation scripts and figures."""

from __future__ import annotations

import numpy as np

__all__ = ["refine_peak", "value_at", "rel_error"]


def refine_peak(x: np.ndarray, y: np.ndarray, window: int = 6):
    """Sub-sample the maximum of ``y(x)``.

    A cubic spline is fitted through the ``2*window+1`` samples straddling
    ``argmax`` and its stationary point is solved for exactly; this removes the
    output-grid quantisation from "altitude of peak deceleration" style
    comparisons.  Falls back to a three-point parabola at the array ends.

    Returns ``(x_peak, y_peak, index)``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    i = int(np.argmax(y))
    if i == 0 or i == y.size - 1:
        return float(x[i]), float(y[i]), i

    lo = max(i - window, 0)
    hi = min(i + window + 1, y.size)
    if hi - lo >= 5:
        from scipy.interpolate import CubicSpline

        xs = x[lo:hi]
        ys = y[lo:hi]
        if xs[0] > xs[-1]:  # ensure ascending abscissa
            xs = xs[::-1]
            ys = ys[::-1]
        if np.all(np.diff(xs) > 0):
            spl = CubicSpline(xs, ys)
            roots = spl.derivative().roots(extrapolate=False)
            roots = roots[(roots >= xs[0]) & (roots <= xs[-1])]
            if roots.size:
                vals = spl(roots)
                k = int(np.argmax(vals))
                if vals[k] >= y[i]:
                    return float(roots[k]), float(vals[k]), i

    x0, x1, x2 = x[i - 1], x[i], x[i + 1]
    y0, y1, y2 = y[i - 1], y[i], y[i + 1]
    denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
    if denom == 0.0:
        return float(x[i]), float(y[i]), i
    a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom
    b = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / denom
    c = (
        x1 * x2 * (x1 - x2) * y0
        + x2 * x0 * (x2 - x0) * y1
        + x0 * x1 * (x0 - x1) * y2
    ) / denom
    if a >= 0.0:
        return float(x[i]), float(y[i]), i
    xp = -b / (2.0 * a)
    yp = a * xp * xp + b * xp + c
    return float(xp), float(yp), i


def value_at(x_query: float, x: np.ndarray, y: np.ndarray) -> float:
    """Linear interpolation of ``y`` at ``x_query`` (``x`` need not be sorted
    ascending; it is re-sorted internally)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x)
    return float(np.interp(x_query, x[order], y[order]))


def rel_error(value: float, reference: float) -> float:
    """Signed relative error ``(value - reference) / |reference|``."""
    return float((value - reference) / abs(reference))
