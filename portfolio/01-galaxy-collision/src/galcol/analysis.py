"""Structural diagnostics used by the validation scripts and the figures."""

from __future__ import annotations

import numpy as np


def robust_centre(pos, mask=None, n_iter=8, shrink=0.75):
    """Shrinking-sphere centre: robust to tidal debris and unbound material."""
    p = pos if mask is None else pos[mask]
    if p.shape[0] == 0:
        return np.zeros(3)
    c = np.median(p, axis=0)
    r = np.linalg.norm(p - c, axis=1)
    rad = np.percentile(r, 90)
    for _ in range(n_iter):
        sel = np.linalg.norm(p - c, axis=1) < rad
        if sel.sum() < 32:
            break
        c = p[sel].mean(axis=0)
        rad *= shrink
    return c


def mass_radius(pos, mass, centre=None, frac=0.5):
    """Spherical radius enclosing ``frac`` of the mass."""
    c = np.zeros(3) if centre is None else np.asarray(centre)
    r = np.linalg.norm(pos - c, axis=1)
    o = np.argsort(r)
    cw = np.cumsum(mass[o])
    cw /= cw[-1]
    return float(np.interp(frac, cw, r[o]))


def cyl_mass_radius(pos, mass, centre=None, frac=0.5):
    """Cylindrical (in-plane) radius enclosing ``frac`` of the mass."""
    c = np.zeros(3) if centre is None else np.asarray(centre)
    d = pos - c
    R = np.hypot(d[:, 0], d[:, 1])
    o = np.argsort(R)
    cw = np.cumsum(mass[o])
    cw /= cw[-1]
    return float(np.interp(frac, cw, R[o]))


def surface_density_profile(pos, mass, centre=None, r_max=15.0, n_bins=24):
    """Azimuthally averaged surface density Sigma(R) in code units."""
    c = np.zeros(3) if centre is None else np.asarray(centre)
    d = pos - c
    R = np.hypot(d[:, 0], d[:, 1])
    edges = np.linspace(0.0, r_max, n_bins + 1)
    m, _ = np.histogram(R, bins=edges, weights=mass)
    area = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    mid = 0.5 * (edges[1:] + edges[:-1])
    return mid, m / area


def fit_scale_length(pos, mass, centre=None, r_fit=(3.0, 10.5), n_bins=24):
    """Exponential scale length from a least-squares fit to ln Sigma(R).

    The fit window deliberately excludes the bulge-dominated centre and the
    truncation edge.
    """
    mid, sigma = surface_density_profile(pos, mass, centre=centre,
                                         r_max=max(r_fit[1] * 1.4, 15.0),
                                         n_bins=n_bins)
    sel = (mid >= r_fit[0]) & (mid <= r_fit[1]) & (sigma > 0)
    if sel.sum() < 3:
        return float("nan")
    slope, _ = np.polyfit(mid[sel], np.log(sigma[sel]), 1)
    return float(-1.0 / slope)


def vertical_thickness(pos, mass, centre=None, r_range=(2.0, 10.0)):
    """Median |z| and rms z of disk particles in an annulus.

    Median |z| is the more robust statistic: for an unheated sech^2 disk of
    scale height zd it equals zd * artanh(1/2) = 0.549 zd.
    """
    c = np.zeros(3) if centre is None else np.asarray(centre)
    d = pos - c
    R = np.hypot(d[:, 0], d[:, 1])
    sel = (R >= r_range[0]) & (R <= r_range[1])
    if sel.sum() < 16:
        return float("nan"), float("nan")
    z = d[sel, 2]
    return float(np.median(np.abs(z))), float(np.sqrt(np.mean(z ** 2)))


def rotation_curve(pos, vel, mass, centre=None, v_centre=None,
                   r_max=16.0, n_bins=16, z_max=1.5):
    """Mean azimuthal streaming speed of disk particles vs cylindrical R."""
    c = np.zeros(3) if centre is None else np.asarray(centre)
    vc = np.zeros(3) if v_centre is None else np.asarray(v_centre)
    d = pos - c
    v = vel - vc
    R = np.hypot(d[:, 0], d[:, 1])
    sel = (np.abs(d[:, 2]) < z_max) & (R < r_max) & (R > 1e-6)
    vphi = (d[sel, 0] * v[sel, 1] - d[sel, 1] * v[sel, 0]) / R[sel]
    edges = np.linspace(0.0, r_max, n_bins + 1)
    idx = np.digitize(R[sel], edges) - 1
    mid = 0.5 * (edges[1:] + edges[:-1])
    out = np.full(n_bins, np.nan)
    for k in range(n_bins):
        m = idx == k
        if m.sum() > 8:
            out[k] = vphi[m].mean()
    return mid, out


def surface_density_map(pos, weights=None, extent=60.0, n=512, axis=2,
                        centre=(0.0, 0.0, 0.0)):
    """2-D projected mass map (for the technical figures, not the hero)."""
    c = np.asarray(centre)
    d = pos - c
    ax = [a for a in (0, 1, 2) if a != axis]
    h, _, _ = np.histogram2d(d[:, ax[1]], d[:, ax[0]], bins=n,
                             range=[[-extent, extent], [-extent, extent]],
                             weights=weights)
    return h


def bound_fraction(pos, vel, mass, pot, centre, v_centre):
    """Fraction of mass bound to a centre, using the simulated potential."""
    d = pos - centre
    v = vel - v_centre
    e = 0.5 * (v * v).sum(axis=1) + pot
    b = e < 0
    return float(mass[b].sum() / mass.sum())
