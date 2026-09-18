"""Analytic Earth + atmosphere renderer (numba, parallel over pixels).

The background of the hero frames is not a matplotlib sphere: every pixel is a
ray traced against the real geometry.

* **Earth** is an exact sphere of radius ``R_EARTH``.  The surface is shaded with
  a Lambert term against the sun direction, a Blinn-Phong ocean glint, and a
  procedural albedo/cloud field sampled from a pre-computed 3-D noise volume
  (sampling a 3-D volume on the sphere avoids any texture seam).
* **Atmosphere** is a shell from the surface to 140 km.  For every pixel the
  renderer integrates along the view ray

      I  =  sum over samples of  beta_R * (rho(z)/rho_0) * T_view * T_sun * ds

  where ``rho(z)`` is the **actual USSA76 density** used by the trajectory
  simulation, passed in as a log-density lookup table, ``beta_R`` are the
  sea-level Rayleigh scattering coefficients for R/G/B, ``T_view`` is the
  accumulated transmittance back to the camera, and ``T_sun`` approximates the
  transmittance from the sample point to the sun.  That view-ray integral is
  what produces the limb glow and its reddening at the terminator: nothing about
  the glow is painted by hand.

Approximations, stated plainly (this is a renderer, not a radiative-transfer
solver): single scattering only; the sun-ward transmittance uses a secant
(plane-parallel) air-mass capped at grazing incidence rather than a full Chapman
function; Mie/aerosol scattering and ozone absorption are omitted; the shadow
test is a hard cylinder with a soft edge.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

__all__ = ["render_background", "build_noise_volumes", "RAYLEIGH_BETA"]

#: Sea-level Rayleigh scattering coefficients for R, G, B [1/m]
#: (the standard 680/550/440 nm values used in atmospheric rendering).
RAYLEIGH_BETA = np.array([5.8e-6, 13.5e-6, 33.1e-6], dtype=np.float64)


# ---------------------------------------------------------------------------
@njit(cache=True, inline="always")
def _sample_noise(vol, n, x, y, z):
    """Periodic trilinear sample of a cubic noise volume of side ``n``."""
    xi = x - np.floor(x / n) * n
    yi = y - np.floor(y / n) * n
    zi = z - np.floor(z / n) * n
    i0 = int(xi)
    j0 = int(yi)
    k0 = int(zi)
    fx = xi - i0
    fy = yi - j0
    fz = zi - k0
    i1 = i0 + 1 if i0 + 1 < n else 0
    j1 = j0 + 1 if j0 + 1 < n else 0
    k1 = k0 + 1 if k0 + 1 < n else 0
    c00 = vol[i0, j0, k0] * (1 - fx) + vol[i1, j0, k0] * fx
    c10 = vol[i0, j1, k0] * (1 - fx) + vol[i1, j1, k0] * fx
    c01 = vol[i0, j0, k1] * (1 - fx) + vol[i1, j0, k1] * fx
    c11 = vol[i0, j1, k1] * (1 - fx) + vol[i1, j1, k1] * fx
    c0 = c00 * (1 - fy) + c10 * fy
    c1 = c01 * (1 - fy) + c11 * fy
    return c0 * (1 - fz) + c1 * fz


@njit(cache=True, inline="always")
def _rho(table, dz, z):
    """Linear interpolation of the log-density table (returns rho/rho_sl)."""
    if z < 0.0:
        return 1.0
    fi = z / dz
    i = int(fi)
    if i >= table.shape[0] - 1:
        return 0.0
    f = fi - i
    return np.exp(table[i] * (1.0 - f) + table[i + 1] * f)


@njit(cache=True, inline="always")
def _sphere_hit(ox, oy, oz, dx, dy, dz_, rad):
    """Return (t_near, t_far) of a ray/sphere intersection, or (-1, -1)."""
    b = ox * dx + oy * dy + oz * dz_
    c = ox * ox + oy * oy + oz * oz - rad * rad
    disc = b * b - c
    if disc < 0.0:
        return -1.0, -1.0
    s = np.sqrt(disc)
    return -b - s, -b + s


@njit(parallel=True, cache=True, fastmath=True)
def render_background(
    width,
    height,
    eye,
    fwd,
    right,
    up,
    tan_half_fov,
    sun,
    r_earth,
    r_atmos,
    logrho_table,
    table_dz,
    beta,
    n_steps,
    vol_a,
    vol_b,
    sun_intensity,
    ground_gain,
    ambient_night,
    city_gain,
):
    """Render the Earth + atmosphere background into an HDR RGB buffer.

    ``out = ground_radiance * T_view + single_scattered_in_scatter``.
    """
    out = np.zeros((height, width, 3), dtype=np.float32)
    aspect = width / height
    na = vol_a.shape[0]
    nb = vol_b.shape[0]

    for iy in prange(height):
        ndc_y = 1.0 - 2.0 * (iy + 0.5) / height
        for ix in range(width):
            ndc_x = 2.0 * (ix + 0.5) / width - 1.0
            dx = fwd[0] + right[0] * ndc_x * tan_half_fov * aspect + up[0] * ndc_y * tan_half_fov
            dy = fwd[1] + right[1] * ndc_x * tan_half_fov * aspect + up[1] * ndc_y * tan_half_fov
            dz = fwd[2] + right[2] * ndc_x * tan_half_fov * aspect + up[2] * ndc_y * tan_half_fov
            inv = 1.0 / np.sqrt(dx * dx + dy * dy + dz * dz)
            dx *= inv
            dy *= inv
            dz *= inv

            tg0, tg1 = _sphere_hit(eye[0], eye[1], eye[2], dx, dy, dz, r_earth)
            ta0, ta1 = _sphere_hit(eye[0], eye[1], eye[2], dx, dy, dz, r_atmos)
            hit_ground = tg0 > 0.0

            gr = 0.0
            gg_ = 0.0
            gb = 0.0

            # ---------------- ground radiance ----------------
            if hit_ground:
                px = eye[0] + dx * tg0
                py = eye[1] + dy * tg0
                pz = eye[2] + dz * tg0
                nxx = px / r_earth
                nyy = py / r_earth
                nzz = pz / r_earth
                ndl = nxx * sun[0] + nyy * sun[1] + nzz * sun[2]
                lam = ndl if ndl > 0.0 else 0.0
                if lam < 1.0:
                    lam = lam * lam * (3.0 - 2.0 * lam)

                s = na * 0.5
                land = _sample_noise(vol_a, na, nxx * s + s, nyy * s + s, nzz * s + s)
                sb = nb * 0.5
                cloud = _sample_noise(
                    vol_b, nb, nxx * sb * 1.7 + sb, nyy * sb * 1.7 + sb,
                    nzz * sb * 1.7 + sb
                )
                cloud2 = _sample_noise(
                    vol_b, nb, nxx * sb * 4.3 + sb * 0.3, nyy * sb * 4.3 + sb * 0.7,
                    nzz * sb * 4.3 + sb * 0.1
                )
                cl = (cloud * 0.68 + cloud2 * 0.32 - 0.47) * 3.6
                if cl < 0.0:
                    cl = 0.0
                if cl > 1.0:
                    cl = 1.0

                is_land = 1.0 / (1.0 + np.exp(-(land - 0.54) * 26.0))
                ar = 0.024 + 0.080 * is_land
                ag = 0.038 + 0.085 * is_land
                ab = 0.082 + 0.045 * is_land
                ar = ar * (1.0 - cl) + 0.92 * cl
                ag = ag * (1.0 - cl) + 0.94 * cl
                ab = ab * (1.0 - cl) + 0.97 * cl

                hx = sun[0] - dx
                hy = sun[1] - dy
                hz = sun[2] - dz
                hn = 1.0 / np.sqrt(hx * hx + hy * hy + hz * hz)
                sp = nxx * hx * hn + nyy * hy * hn + nzz * hz * hn
                if sp < 0.0:
                    sp = 0.0
                sp2 = sp * sp
                sp4 = sp2 * sp2
                sp8 = sp4 * sp4
                sp16 = sp8 * sp8
                sp32 = sp16 * sp16
                sp64 = sp32 * sp32
                spec = sp64 * sp64 * (1.0 - is_land) * (1.0 - cl) * 7.0

                gsun = sun_intensity * ground_gain * lam
                gr = ar * gsun + spec * gsun * 0.65
                gg_ = ag * gsun + spec * gsun * 0.65
                gb = ab * gsun + spec * gsun * 0.75
                gr += ambient_night * (0.55 + 0.45 * cl)
                gg_ += ambient_night * (0.70 + 0.30 * cl)
                gb += ambient_night * (1.00 + 0.20 * cl)

                # Settlement lights on the night side: clustered on "land",
                # thresholded high-frequency noise, hidden by cloud and by day.
                if city_gain > 0.0 and ndl < 0.08:
                    dark = 1.0 - ndl / 0.08 if ndl > 0.0 else 1.0
                    spark = _sample_noise(
                        vol_b, nb, nxx * sb * 26.0 + sb * 0.9,
                        nyy * sb * 26.0 + sb * 0.2, nzz * sb * 26.0 + sb * 0.5
                    )
                    spark2 = _sample_noise(
                        vol_b, nb, nxx * sb * 9.0 + sb * 0.3,
                        nyy * sb * 9.0 + sb * 0.8, nzz * sb * 9.0 + sb * 0.6
                    )
                    lit_c = (spark - 0.62) * 6.0 * (spark2 - 0.40) * 2.2
                    if lit_c > 0.0:
                        glow = lit_c * lit_c * is_land * dark * (1.0 - cl) * city_gain
                        gr += glow * 1.00
                        gg_ += glow * 0.72
                        gb += glow * 0.38

            # ---------------- atmospheric single scattering ----------------
            ir = 0.0
            ig = 0.0
            ib = 0.0
            tvr = 1.0
            tvg = 1.0
            tvb = 1.0
            if ta1 > 0.0:
                t0 = ta0 if ta0 > 0.0 else 0.0
                t1 = tg0 if hit_ground else ta1
                if t1 > t0:
                    ds = (t1 - t0) / n_steps
                    for k in range(n_steps):
                        tt = t0 + (k + 0.5) * ds
                        sx = eye[0] + dx * tt
                        sy = eye[1] + dy * tt
                        sz = eye[2] + dz * tt
                        rad = np.sqrt(sx * sx + sy * sy + sz * sz)
                        dens = _rho(logrho_table, table_dz, rad - r_earth)
                        # Below this the sample cannot change the pixel at 8-bit
                        # output precision; skipping it avoids six exp() calls.
                        if dens < 2.0e-7:
                            continue
                        seg = dens * ds
                        odr = beta[0] * seg
                        odg = beta[1] * seg
                        odb = beta[2] * seg

                        proj = sx * sun[0] + sy * sun[1] + sz * sun[2]
                        lit = 1.0
                        if proj < 0.0:
                            perp2 = rad * rad - proj * proj
                            perp = np.sqrt(perp2) if perp2 > 0.0 else 0.0
                            e = (perp - r_earth) / 30000.0
                            if e < 0.0:
                                lit = 0.0
                            elif e < 1.0:
                                lit = e * e * (3.0 - 2.0 * e)
                        if lit > 0.0:
                            cos_s = proj / rad
                            if cos_s < 0.06:
                                cos_s = 0.06
                            air_sun = dens * 8500.0 / cos_s
                            w = sun_intensity * lit
                            ir += odr * tvr * np.exp(-beta[0] * air_sun) * w
                            ig += odg * tvg * np.exp(-beta[1] * air_sun) * w
                            ib += odb * tvb * np.exp(-beta[2] * air_sun) * w

                        tvr *= np.exp(-odr)
                        tvg *= np.exp(-odg)
                        tvb *= np.exp(-odb)

            out[iy, ix, 0] = gr * tvr + ir
            out[iy, ix, 1] = gg_ * tvg + ig
            out[iy, ix, 2] = gb * tvb + ib
    return out


@njit(parallel=True, cache=True, fastmath=True)
def optical_depth_to_points(eye, points, r_earth, r_atmos, logrho_table, table_dz,
                            n_steps):
    """Column density (in metres of sea-level-equivalent air) from ``eye`` to
    each point, used to extinguish and redden the reentry trail."""
    n = points.shape[0]
    out = np.zeros(n, dtype=np.float64)
    for i in prange(n):
        ox = eye[0]
        oy = eye[1]
        oz = eye[2]
        vx = points[i, 0] - ox
        vy = points[i, 1] - oy
        vz = points[i, 2] - oz
        length = np.sqrt(vx * vx + vy * vy + vz * vz)
        dx = vx / length
        dy = vy / length
        dz = vz / length
        ta0, ta1 = _sphere_hit(ox, oy, oz, dx, dy, dz, r_atmos)
        if ta1 <= 0.0:
            continue
        t0 = ta0 if ta0 > 0.0 else 0.0
        t1 = ta1 if ta1 < length else length
        if t1 <= t0:
            continue
        ds = (t1 - t0) / n_steps
        acc = 0.0
        for k in range(n_steps):
            tt = t0 + (k + 0.5) * ds
            sx = ox + dx * tt
            sy = oy + dy * tt
            sz = oz + dz * tt
            rad = np.sqrt(sx * sx + sy * sy + sz * sz)
            acc += _rho(logrho_table, table_dz, rad - r_earth) * ds
        out[i] = acc
    return out


def build_noise_volumes(seed=7, n_a=64, n_b=128, sigma_a=5.5, sigma_b=2.2):
    """Two periodic fBm-ish noise volumes: large-scale albedo and cloud deck."""
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed)

    def _vol(n, sigma, octaves):
        acc = np.zeros((n, n, n), dtype=np.float64)
        amp = 1.0
        norm = 0.0
        s = sigma
        for _ in range(octaves):
            w = gaussian_filter(rng.standard_normal((n, n, n)), sigma=s, mode="wrap")
            w = (w - w.mean()) / (w.std() + 1e-12)
            acc += amp * w
            norm += amp
            amp *= 0.55
            s *= 0.5
        acc /= norm
        acc = (acc - acc.min()) / (acc.max() - acc.min())
        return np.ascontiguousarray(acc)

    return _vol(n_a, sigma_a, 3), _vol(n_b, sigma_b, 4)
