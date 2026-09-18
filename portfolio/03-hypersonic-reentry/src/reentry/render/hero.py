"""Cinematic hero renderer for the reentry trajectory.

Everything on screen is derived from the simulation:

* the **trail geometry** is the integrated 3-DOF trajectory converted to
  Cartesian coordinates;
* the **trail brightness** is the simulated Sutton-Graves stagnation-point heat
  flux ``qdot(t)`` (times a documented wake-persistence factor);
* the **trail colour** is the Planck-locus colour of the radiative-equilibrium
  wall temperature ``T_w = (qdot / (eps sigma))^(1/4)`` computed from that same
  heat flux -- the trail is literally the colour of the glowing heat shield;
* the **limb glow** is a single-scattering integral of the *same USSA76 density
  profile* the trajectory was flown through, taken along every view ray.

The only invented quantities are photographic: exposure, bloom radii, the
wake-persistence time constant, and the procedural cloud field.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..atmosphere import USSA76
from ..blackbody import make_blackbody_lut
from ..constants import RHO_SL, R_EARTH, SIGMA_SB
from .compose import bloom, splat, tonemap, upsample_bilinear
from .raymarch import (
    RAYLEIGH_BETA,
    build_noise_volumes,
    optical_depth_to_points,
    render_background,
)

__all__ = ["HeroScene", "CameraPath"]

R_ATMOS = R_EARTH + 140.0e3


# ---------------------------------------------------------------------------
def _norm(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _ease(x):
    """Smooth, slow-in/slow-out easing used for all camera motion."""
    x = np.clip(x, 0.0, 1.0)
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)


@dataclass
class CameraPath:
    """Camera rig, expressed in the vehicle's local up/along/cross frame.

    The eye sits at fixed offsets from an anchor point that *lags* the vehicle,
    so the vehicle drifts through the composition instead of being pinned.  The
    aim is built from the geometry rather than from a hand-picked look-at point:
    the pitch is interpolated between the true horizon direction (``aim_blend``
    = 0, horizon centred) and the vehicle direction (``aim_blend`` = 1, vehicle
    centred), which keeps the limb in a stable place in frame while the camera
    moves.  Everything interpolates from ``start`` to ``end`` with a quintic
    ease.
    """

    start_offset: tuple = (2.6e5, 5.0e5, 1.85e6)
    end_offset: tuple = (1.5e5, 1.2e5, 1.30e6)
    start_fov_deg: float = 27.0
    end_fov_deg: float = 22.0
    start_roll_deg: float = -6.0
    end_roll_deg: float = 3.0
    start_aim_blend: float = 0.62
    end_aim_blend: float = 0.50
    start_yaw_deg: float = -5.0
    end_yaw_deg: float = 4.0
    anchor_lag_s: float = 16.0

    def at(self, s):
        e = _ease(s)

        def lerp(a, b):
            return a + (b - a) * e

        off = tuple(lerp(self.start_offset[i], self.end_offset[i]) for i in range(3))
        return (
            off,
            lerp(self.start_fov_deg, self.end_fov_deg),
            lerp(self.start_roll_deg, self.end_roll_deg),
            lerp(self.start_aim_blend, self.end_aim_blend),
            lerp(self.start_yaw_deg, self.end_yaw_deg),
        )


def _rotate_about(v, axis, angle):
    """Rodrigues rotation of ``v`` about the unit vector ``axis``."""
    c, s = np.cos(angle), np.sin(angle)
    return v * c + np.cross(axis, v) * s + axis * np.dot(axis, v) * (1.0 - c)


@dataclass
class HeroScene:
    """Holds every precomputed asset and renders one frame at a time."""

    positions: np.ndarray  # (N, 3) inertial positions along the trajectory
    times: np.ndarray  # (N,)
    q_dot: np.ndarray  # (N,) Sutton-Graves heat flux [W/m^2]
    altitude: np.ndarray  # (N,)
    emissivity: float = 0.85
    sun_azimuth_deg: float = -78.0
    sun_elevation_deg: float = -11.0
    sun_intensity: float = 22.0
    ground_gain: float = 1.0
    ambient_night: float = 0.0035
    city_gain: float = 0.10
    wake_halo_gain: float = 2.2
    head_gain: float = 55.0
    persistence_s: float = 55.0
    trail_samples: int = 26_000
    n_stars: int = 5200
    seed: int = 11
    camera: CameraPath = field(default_factory=CameraPath)

    # -- setup --------------------------------------------------------------
    def __post_init__(self):
        atm = USSA76()
        dz = 100.0
        z = np.arange(0.0, 140.0e3 + dz, dz)
        rho = np.asarray(atm.density(z)) / RHO_SL
        self._logrho = np.log(np.maximum(rho, 1e-300))
        self._table_dz = dz
        self._vol_a, self._vol_b = build_noise_volumes(seed=self.seed)
        self._make_trail()
        self._make_stars()
        self._make_sun()
        self._head_lut_t, self._head_lut_rgb = make_blackbody_lut(600.0, 3600.0, 256)
        self._bufcache = None

    def _make_sun(self):
        """Sun direction, defined in the trajectory's own frame.

        ``sun_elevation_deg`` is the solar elevation *at the mid-trajectory
        point*: a negative value puts that point in darkness, and because the
        local vertical tips forward along the ground track, the sunrise
        terminator then sits on the horizon ahead of the vehicle.  That is what
        produces the back-lit limb arc.  ``sun_azimuth_deg`` is measured from
        the along-track direction toward the cross-track direction.
        """
        mid = self.positions[len(self.positions) // 2]
        nxt = self.positions[min(len(self.positions) // 2 + 50,
                                 len(self.positions) - 1)]
        up_hat = _norm(mid)
        along = _norm(nxt - mid)
        along = _norm(along - up_hat * np.dot(along, up_hat))
        cross = np.cross(up_hat, along)
        az = np.radians(self.sun_azimuth_deg)
        el = np.radians(self.sun_elevation_deg)
        self.sun = _norm(
            np.cos(el) * (np.cos(az) * along + np.sin(az) * cross)
            + np.sin(el) * up_hat
        )

    def _make_trail(self):
        """Resample the trajectory uniformly in arc length.

        Uniform-in-time sampling would pile points up where the vehicle is slow
        and make the trail brightness depend on the speed; uniform-in-arc-length
        sampling makes the deposited energy per metre of path proportional to the
        physical quantity we are encoding.
        """
        p = self.positions
        seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        su = np.linspace(0.0, s[-1], self.trail_samples)
        self.trail_pos = np.stack([np.interp(su, s, p[:, k]) for k in range(3)], axis=1)
        self.trail_t = np.interp(su, s, self.times)
        self.trail_q = np.interp(su, s, self.q_dot)
        self.trail_alt = np.interp(su, s, self.altitude)
        self.q_max = float(np.max(self.q_dot))
        tw = (np.maximum(self.trail_q, 1.0) / (self.emissivity * SIGMA_SB)) ** 0.25
        self.trail_tw = tw
        lut_t, lut_rgb = make_blackbody_lut(600.0, 3200.0, 384)
        self.trail_rgb = np.stack(
            [np.interp(np.clip(tw, 600.0, 3200.0), lut_t, lut_rgb[:, k])
             for k in range(3)], axis=1
        )

    def _make_stars(self):
        rng = np.random.default_rng(self.seed + 3)
        v = rng.standard_normal((self.n_stars, 3))
        self.star_dir = v / np.linalg.norm(v, axis=1, keepdims=True)
        # A plausible magnitude distribution: many faint, few bright.
        u = rng.random(self.n_stars)
        self.star_flux = 0.055 * (u ** 3.0) * 60.0 + 0.010
        # Slight colour spread around white (hot blue-white to cool orange).
        tmp = rng.normal(0.0, 1.0, self.n_stars)
        self.star_rgb = np.stack(
            [1.0 + 0.10 * tmp, np.ones(self.n_stars), 1.0 - 0.10 * tmp], axis=1
        ).clip(0.55, 1.35)

    # -- camera -------------------------------------------------------------
    def camera_at(self, t_sim, s):
        off, fov, roll, aim_blend, yaw = self.camera.at(s)
        t_anchor = max(self.times[0], t_sim - self.camera.anchor_lag_s)
        anchor = self._pos_at(t_anchor)
        ahead = self._pos_at(min(t_anchor + 40.0, self.times[-1]))
        up_hat = _norm(anchor)
        along = _norm(ahead - anchor)
        along = _norm(along - up_hat * np.dot(along, up_hat))
        cross = np.cross(up_hat, along)

        eye = anchor + off[0] * up_hat + off[1] * along + off[2] * cross
        veh = self._pos_at(t_sim)

        n = _norm(eye)
        to_veh = veh - eye
        horiz = to_veh - n * np.dot(to_veh, n)
        h = _norm(horiz) if np.linalg.norm(horiz) > 1.0 else cross
        phi = np.arcsin(np.clip(R_EARTH / np.linalg.norm(eye), -1.0, 1.0))
        pitch_v = np.arccos(np.clip(np.dot(_norm(to_veh), -n), -1.0, 1.0))
        pitch = phi + aim_blend * (pitch_v - phi)
        fwd = -n * np.cos(pitch) + h * np.sin(pitch)
        fwd = _norm(_rotate_about(fwd, n, np.radians(yaw)))

        right = _norm(np.cross(fwd, n))
        up = np.cross(right, fwd)
        cr, sr = np.cos(np.radians(roll)), np.sin(np.radians(roll))
        right, up = right * cr + up * sr, -right * sr + up * cr
        return eye, fwd, right, up, np.tan(np.radians(fov) * 0.5)

    def _pos_at(self, t):
        return np.stack(
            [np.interp(t, self.times, self.positions[:, k]) for k in range(3)]
        )

    # -- projection ---------------------------------------------------------
    @staticmethod
    def _project(points, eye, fwd, right, up, tan_half, width, height):
        d = points - eye
        z = d @ fwd
        xs = d @ right
        ys = d @ up
        aspect = width / height
        with np.errstate(divide="ignore", invalid="ignore"):
            ndc_x = xs / (z * tan_half * aspect)
            ndc_y = ys / (z * tan_half)
        px = (ndc_x + 1.0) * 0.5 * width - 0.5
        py = (1.0 - ndc_y) * 0.5 * height - 0.5
        return px, py, z

    @staticmethod
    def _occluded(points, eye, radius):
        """True where the Earth sphere blocks the line of sight."""
        d = points - eye
        length = np.linalg.norm(d, axis=1)
        u = d / length[:, None]
        b = u @ eye
        c = eye @ eye - radius * radius
        disc = b * b - c
        hit = disc > 0.0
        t = np.where(hit, -b - np.sqrt(np.maximum(disc, 0.0)), np.inf)
        return hit & (t > 0.0) & (t < length)

    # -- frame --------------------------------------------------------------
    def _scratch(self, shape):
        """Reusable zeroed scratch buffers (frame rendering is called hundreds
        of times; re-allocating 25 MB float buffers per frame dominates)."""
        cache = getattr(self, "_bufcache", None)
        if cache is None or cache[0].shape != shape:
            cache = (np.zeros(shape, np.float32), np.zeros(shape, np.float32))
            self._bufcache = cache
        for b in cache:
            b.fill(0.0)
        return cache

    def render_frame(self, t_sim, s, width=1920, height=1080, n_steps=32,
                     star_gain=1.0, trail_gain=1.0):
        from scipy.ndimage import gaussian_filter

        eye, fwd, right, up, tan_half = self.camera_at(t_sim, s)

        buf = render_background(
            width, height,
            np.ascontiguousarray(eye), np.ascontiguousarray(fwd),
            np.ascontiguousarray(right), np.ascontiguousarray(up),
            float(tan_half), np.ascontiguousarray(self.sun),
            R_EARTH, R_ATMOS, self._logrho, self._table_dz,
            RAYLEIGH_BETA, int(n_steps), self._vol_a, self._vol_b,
            float(self.sun_intensity), float(self.ground_gain),
            float(self.ambient_night), float(self.city_gain),
        )

        # ---- stars (behind everything; hidden by the Earth) ----
        far = eye + self.star_dir * 6.0e9
        px, py, z = self._project(far, eye, fwd, right, up, tan_half, width, height)
        vis = (z > 0) & ~self._occluded(far, eye, R_EARTH)
        splat(buf, px[vis], py[vis], self.star_rgb[vis],
              self.star_flux[vis] * star_gain)

        core, _unused = self._scratch(buf.shape)
        hstep = 4
        hw, hh = width // hstep, height // hstep
        halo = np.zeros((hh, hw, 3), np.float32)

        # ---- reentry trail ----
        live = self.trail_t <= t_sim
        if np.any(live):
            pts = self.trail_pos[live]
            age = t_sim - self.trail_t[live]
            w = (self.trail_q[live] / self.q_max) * np.exp(-age / self.persistence_s)
            keep = w > 2e-4
            pts = pts[keep]
            w = w[keep]
            rgb = self.trail_rgb[live][keep]
            if pts.shape[0]:
                px, py, z = self._project(pts, eye, fwd, right, up, tan_half,
                                          width, height)
                vis = (z > 0) & ~self._occluded(pts, eye, R_EARTH)
                col = optical_depth_to_points(
                    np.ascontiguousarray(eye), np.ascontiguousarray(pts),
                    R_EARTH, R_ATMOS, self._logrho, self._table_dz, 20
                )
                ext = np.exp(-np.outer(col, RAYLEIGH_BETA))
                col_rgb = (rgb * ext)[vis]
                norm = trail_gain * (26_000.0 / self.trail_samples)
                # Core filament: thin and sharp.
                splat(core, px[vis], py[vis], col_rgb, w[vis] * norm)
                # Wake sheath: a broader, dimmer halo that grows super-linearly
                # with the heat flux, so it only appears where the flow is
                # genuinely hot.  Rendered at 1/4 resolution -- it carries no
                # detail finer than that.  A rendering layer, not new physics.
                splat(halo, px[vis] / hstep, py[vis] / hstep, col_rgb,
                      (w[vis] ** 1.7) * norm * self.wake_halo_gain)

        # ---- the vehicle itself ----
        veh = self._pos_at(t_sim)[None, :]
        q_now = float(np.interp(t_sim, self.times, self.q_dot))
        tw_now = (max(q_now, 1.0) / (self.emissivity * SIGMA_SB)) ** 0.25
        col_now = np.array(
            [[np.interp(np.clip(tw_now, 600.0, 3600.0), self._head_lut_t,
                        self._head_lut_rgb[:, k]) for k in range(3)]]
        )
        px, py, z = self._project(veh, eye, fwd, right, up, tan_half, width, height)
        if z[0] > 0 and not self._occluded(veh, eye, R_EARTH)[0]:
            head = self.head_gain * (0.06 + 0.94 * q_now / self.q_max) * trail_gain
            splat(core, px, py, 0.45 + 0.55 * col_now, np.array([head]))
            splat(halo, px / hstep, py / hstep, col_now,
                  np.array([head * 0.7 * self.wake_halo_gain]))

        sig = max(1.15 * width / 1920.0, 0.7)
        core = gaussian_filter(core, sigma=(sig, sig, 0), truncate=3.0)
        hsig = max(7.0 * width / (1920.0 * hstep), 1.0)
        halo = gaussian_filter(halo, sigma=(hsig, hsig, 0), truncate=3.0)
        buf += core
        buf += upsample_bilinear(halo, height, width)
        return buf, (eye, fwd, right, up, tan_half)

    def render_image(self, t_sim, s, width=1920, height=1080, n_steps=40,
                     supersample=1, exposure=1.0, star_gain=1.0, trail_gain=1.0,
                     tonemap_kwargs=None):
        ss = int(supersample)
        buf, _ = self.render_frame(
            t_sim, s, width * ss, height * ss, n_steps=n_steps,
            star_gain=star_gain * ss * ss, trail_gain=trail_gain * ss,
        )
        kw = dict(exposure=exposure)
        if tonemap_kwargs:
            kw.update(tonemap_kwargs)
        # Bloom radii are authored for a 1920-wide frame; scale them with the
        # render resolution so low-resolution previews look like the final frame.
        scale = width * ss / 1920.0
        sig = kw.get("bloom_sigmas", (2.0, 8.0, 28.0))
        kw["bloom_sigmas"] = tuple(max(x * scale, 0.8) for x in sig)
        img = tonemap(buf, **kw)
        if ss > 1:
            img = (
                img.reshape(height, ss, width, ss, 3)
                .astype(np.float32)
                .mean(axis=(1, 3))
                .round()
                .astype(np.uint8)
            )
        return img
