#!/usr/bin/env python3
"""Render media/hero.png and media/hero.mp4 from real simulation output.

Everything drawn comes from a logged ``SwarmResult``: aircraft positions and
attitudes, their flight history, the no-fly cylinders and the goal markers.
No matplotlib, no scatter plots, no overlays.

Run directly (``python3 make_hero.py``) or via ``make hero``.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

from swarmsim import scenarios, sim                                   # noqa: E402
from swarmsim.render import (Camera, activity_colour, attitude_matrix,  # noqa: E402
                             circle_points, splat, tonemap, _MESH_F, _MESH_V)

W, H = 1920, 1080
FPS = 30
CACHE = os.path.join(HERE, "results", "hero_run.pkl")
MEDIA = os.path.join(HERE, "media")

# ---- look ---------------------------------------------------------------
TRAIL_SECONDS = 46.0        # how much flight history the glow buffer holds
TRAIL_GAIN = 0.52           # energy per trail sample
TRAIL_SUBSTEPS = 4          # interpolation between logged samples (smooth lines)
TRAIL_SIGMA = 1.7          # px, widens 1-px threads into ribbons
GLYPH_SCALE_M = 74.0       # glyph span in metres (a readable exaggeration)
GLYPH_GAIN = 1.30
HALO_GAIN = 0.34
ZONE_GAIN = 0.30
FOG_SCALE = 2700.0          # atmospheric extinction e-folding distance [m]
SKY = np.array([0.0045, 0.0085, 0.020], np.float32)
EXPOSURE = 1.30


# =========================================================================
# Simulation (cached)
# =========================================================================
def get_run(force=False):
    if os.path.exists(CACHE) and not force:
        with open(CACHE, "rb") as fh:
            return pickle.load(fh)
    spec = scenarios.hero()
    res = sim.run(spec, progress=True)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "wb") as fh:
        pickle.dump(res, fh, protocol=4)
    return res


# =========================================================================
# Frame assembly
# =========================================================================
class HeroScene:
    """Holds the run and renders one frame at a given sample index."""

    def __init__(self, res, t_window=(0.0, 1e9)):
        self.res = res
        self.t = res.t
        self.S = res.states                      # (T, N, 7)
        self.dt = float(self.t[1] - self.t[0])
        self.N = self.S.shape[1]

        # ---- colour driver: manoeuvre activity ------------------------
        # Manoeuvre activity: bank angle relative to a 20-degree reference
        # (the working range through the obstacle field) plus the avoidance
        # deflection.  Both are simulated quantities, not palette indices.
        phi = np.abs(self.S[:, :, 6])
        phi_n = phi / np.deg2rad(12.0)
        defl = res.deflection / 3.0
        q = np.clip(0.80 * phi_n + 0.45 * np.clip(defl, 0.0, 2.0), 0.0, 1.0)
        # temporal smoothing so the colour does not flicker
        k = max(1, int(round(0.6 / self.dt)))
        ker = np.ones(k) / k
        qs = np.empty_like(q)
        for i in range(self.N):
            qs[:, i] = np.convolve(q[:, i], ker, mode="same")
        self.q = np.clip(qs, 0.0, 1.0)
        # Base hue from cruise altitude, normalised over the band the swarm
        # actually occupies.
        z = self.S[:, :, 2]
        z_lo, z_hi = float(z.min()), float(z.max())
        self.z_norm = (z - z_lo) / max(z_hi - z_lo, 1e-6)
        self.col = activity_colour(self.q, self.z_norm)      # (T, N, 3)

        # ---- densified trail samples ---------------------------------
        self.centre = np.array([self.S[:, :, 0].mean(), self.S[:, :, 1].mean(),
                                self.S[:, :, 2].mean()])
        self.zones = res.spec.zones
        self.goals = res.spec.goals

        # Pre-build the no-fly cylinder geometry (rims + a few verticals), drawn
        # over the altitude band the swarm actually occupies.
        z_lo = float(self.S[:, :, 2].min()) - 90.0
        z_hi = float(self.S[:, :, 2].max()) + 90.0
        self.z_band = (z_lo, z_hi)
        self.zone_pts = []
        self.zone_w = []
        for z in self.zones:
            # Sample densely enough that a rim is a continuous glowing line on
            # screen rather than a dotted ellipse.
            nring = 1400
            rings = [(circle_points(z.x, z.y, z_lo, z.radius, nring), 1.0),
                     (circle_points(z.x, z.y, z_hi, z.radius, nring), 0.62),
                     (circle_points(z.x, z.y, 0.5 * (z_lo + z_hi), z.radius, nring), 0.20)]
            nv = 150
            verts = []
            for a in np.linspace(0, 2 * np.pi, 16, endpoint=False):
                zs = np.linspace(z_lo, z_hi, nv)
                verts.append(np.stack([np.full(nv, z.x + z.radius * np.cos(a)),
                                       np.full(nv, z.y + z.radius * np.sin(a)),
                                       zs], axis=1))
            pts = np.vstack([r[0] for r in rings] + verts)
            wts = np.concatenate([np.full(len(r[0]), r[1]) for r in rings]
                                 + [np.full(nv, 0.13)] * 16)
            self.zone_pts.append(pts)
            self.zone_w.append(wts)

    # -- trails ---------------------------------------------------------
    def trail_samples(self, k):
        """Interpolated recent track points, their colours and their ages."""
        n_hist = int(TRAIL_SECONDS / self.dt)
        k0 = max(0, k - n_hist)
        seg = self.S[k0:k + 1, :, :3]                    # (M, N, 3)
        cseg = self.col[k0:k + 1]                        # (M, N, 3)
        M = seg.shape[0]
        if M < 2:
            return None
        # linear interpolation between logged samples
        u = np.linspace(0.0, 1.0, TRAIL_SUBSTEPS, endpoint=False)[None, :, None, None]
        a = seg[:-1][:, None, :, :]
        b = seg[1:][:, None, :, :]
        pts = (a + (b - a) * u).reshape(-1, self.N, 3)
        ca = cseg[:-1][:, None, :, :]
        cb = cseg[1:][:, None, :, :]
        cols = (ca + (cb - ca) * u).reshape(-1, self.N, 3)
        nsub = pts.shape[0]
        age = np.linspace(1.0, 0.0, nsub)[:, None]       # 1 = oldest
        return pts, cols, age

    # -- one frame ------------------------------------------------------
    def render(self, k, cam, exposure=EXPOSURE):
        buf = np.zeros((H, W, 3), np.float32)

        # ---------------- background: near-black with a faint haze ----
        yy = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None]
        buf += SKY[None, None, :] * (0.10 + 1.0 * yy ** 2.4)[:, :, None]

        # ---------------- no-fly cylinders ----------------------------
        for pts, wts in zip(self.zone_pts, self.zone_w):
            xy, z, vis = cam.project(pts)
            if not vis.any():
                continue
            fog = np.exp(-np.maximum(z, 0.0) / FOG_SCALE)
            rgb = np.tile(np.array([0.26, 0.50, 0.92], np.float32), (len(pts), 1))
            wgt = (wts * fog * ZONE_GAIN * vis).astype(np.float32)
            splat(buf, xy[:, 0], xy[:, 1], rgb, wgt)

        # ---------------- goal markers --------------------------------
        gp = self.goals
        xy, z, vis = cam.project(gp)
        fog = np.exp(-np.maximum(z, 0.0) / FOG_SCALE)
        rgb = np.tile(np.array([0.98, 0.74, 0.30], np.float32), (len(gp), 1))
        splat(buf, xy[:, 0], xy[:, 1], rgb, (1.1 * fog * vis).astype(np.float32))

        # ---------------- trails --------------------------------------
        tr = self.trail_samples(k)
        if tr is not None:
            pts, cols, age = tr
            flat = pts.reshape(-1, 3)
            xy, z, vis = cam.project(flat)
            fog = np.exp(-np.maximum(z, 0.0) / FOG_SCALE)
            fade = ((1.0 - age) ** 1.7).repeat(self.N, axis=1).reshape(-1)
            wgt = (TRAIL_GAIN * fade * fog * vis).astype(np.float32)
            trail = np.zeros_like(buf)
            splat(trail, xy[:, 0], xy[:, 1],
                  cols.reshape(-1, 3).astype(np.float32), wgt)
            # soften into ribbons rather than 1-pixel threads
            from scipy.ndimage import gaussian_filter
            buf += gaussian_filter(trail, sigma=(TRAIL_SIGMA, TRAIL_SIGMA, 0))

        # ---------------- aircraft glyphs -----------------------------
        self._draw_aircraft(buf, k, cam)

        return tonemap(buf, exposure=exposure)

    def _draw_aircraft(self, buf, k, cam):
        s = self.S[k]
        pos = s[:, :3]
        psi, gam, phi = s[:, 4], s[:, 5], s[:, 6]
        R = attitude_matrix(psi, gam, phi)                     # (N, 3, 3)
        verts = (_MESH_V[None, :, :] * GLYPH_SCALE_M) @ np.transpose(R, (0, 2, 1))
        verts = verts + pos[:, None, :]                        # (N, V, 3)

        flat = verts.reshape(-1, 3)
        xy, z, vis = cam.project(flat)
        xy = xy.reshape(self.N, -1, 2)
        zc = z.reshape(self.N, -1)
        visc = vis.reshape(self.N, -1)

        _, zctr, vctr = cam.project(pos)
        order = np.argsort(-zctr)                              # far to near

        img = Image.new("RGB", (W, H), (0, 0, 0))
        dr = ImageDraw.Draw(img)
        # a fixed key-light direction, so the bank angle reads as shading
        key = np.array([0.42, 0.36, 0.83])
        key = key / np.linalg.norm(key)

        halo_xy = []
        halo_rgb = []
        halo_w = []

        for i in order:
            if not vctr[i] or zctr[i] < 1.0:
                continue
            if not visc[i].all():
                continue
            px = xy[i]
            # cull tiny / off-screen glyphs early
            if (px[:, 0].max() < -60 or px[:, 0].min() > W + 60
                    or px[:, 1].max() < -60 or px[:, 1].min() > H + 60):
                continue
            fog = float(np.exp(-max(zctr[i], 0.0) / FOG_SCALE))
            base = self.col[k, i]
            for idx, shade, emissive in _MESH_F:
                p = verts[i, list(idx)]
                nrm = np.cross(p[1] - p[0], p[2] - p[0])
                nl = np.linalg.norm(nrm)
                if nl < 1e-9:
                    continue
                nrm = nrm / nl
                lam = abs(float(nrm @ key))
                if emissive:
                    c = np.clip(base * 1.0 + 0.22, 0, 1) * 255.0 * fog
                else:
                    c = base * (0.06 + 1.30 * lam ** 1.35) * shade * 255.0 * fog
                poly = [tuple(px[j]) for j in idx]
                dr.polygon(poly, fill=tuple(int(v) for v in np.clip(c, 0, 255)))
            # bright leading-edge outline: this is what makes the shape read
            outline = np.clip(base * 1.45 + 0.10, 0, 1) * 255.0 * fog
            oc = tuple(int(v) for v in np.clip(outline, 0, 255))
            for a, b in ((1, 0), (0, 3)):
                dr.line([tuple(px[a]), tuple(px[b])], fill=oc, width=2)

            # additive halo so bloom picks the vehicle out of the background
            span = float(np.linalg.norm(px[1] - px[3]))
            halo_xy.append(px[0])
            halo_rgb.append(np.clip(base * 1.15, 0, 1))
            halo_w.append(HALO_GAIN * fog * np.clip(span / 26.0, 0.20, 2.2))

        arr = np.asarray(img, dtype=np.float32) / 255.0
        buf += arr * GLYPH_GAIN

        if halo_xy:
            hxy = np.array(halo_xy)
            splat(buf, hxy[:, 0], hxy[:, 1],
                  np.array(halo_rgb, np.float32), np.array(halo_w, np.float32))


# =========================================================================
# Camera path
# =========================================================================
# Camera path parameters (baked-in defaults; overridable from the CLI while
# framing).  The shot is a slow orbit with a gentle dolly-in.
CAM = dict(az0=-126.0, az_sweep=58.0, elev0=9.5, elev_lift=7.0,
           fov0=40.0, fov_zoom=3.5, frame=0.98, dolly=0.10,
           rad_min=1100.0, rad_max=4200.0, focus_mix=0.55, focus_z=880.0)


def camera_at(scene, u, k, cam=None):
    """Slow orbit + dolly, framed on the swarm.

    The look-at point is a blend of the live swarm centroid and the centre of
    the obstacle field, so the camera tracks the action without drifting with
    every gust.  The range is derived from the swarm's own 88th-percentile
    radius, so the framing adapts as the formation contracts and expands
    instead of being hand-keyed to the clock.
    """
    c = dict(CAM)
    if cam:
        c.update(cam)
    S = scene.S[k]
    centroid = np.array([S[:, 0].mean(), S[:, 1].mean(), S[:, 2].mean()])
    focus = np.array([c["focus_mix"] * centroid[0], c["focus_mix"] * centroid[1],
                      0.45 * centroid[2] + 0.55 * c["focus_z"]])

    span = float(np.percentile(np.linalg.norm(S[:, :2] - focus[None, :2], axis=1), 88))
    span = float(np.clip(2.0 * span, 2100.0, 4300.0))

    az = np.deg2rad(c["az0"] + c["az_sweep"] * u)
    e = u * u * (3.0 - 2.0 * u)                       # smoothstep ease
    elev = np.deg2rad(c["elev0"] + c["elev_lift"] * np.sin(np.pi * u) - 2.0 * e)
    fov = c["fov0"] - c["fov_zoom"] * e

    fov_h = 2.0 * np.arctan(np.tan(np.deg2rad(fov) * 0.5) * W / H)
    rad = 0.5 * span / np.tan(0.5 * fov_h * c["frame"])
    rad = float(np.clip(rad, c["rad_min"], c["rad_max"])) * (1.0 - c["dolly"] * e)

    eye = focus + np.array([rad * np.cos(elev) * np.cos(az),
                            rad * np.cos(elev) * np.sin(az),
                            rad * np.sin(elev)])
    return Camera(eye, focus, up=(0, 0, 1), fov_deg=fov, width=W, height=H)


# =========================================================================
# Workers
# =========================================================================
_SCENE = None
_FRAMES = None


def _init(res, frames):
    global _SCENE, _FRAMES
    _SCENE = HeroScene(res)
    _FRAMES = frames


def _work(n):
    k, u = _FRAMES[n]
    cam = camera_at(_SCENE, u, k)
    return _SCENE.render(k, cam)


# =========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--still-only", action="store_true")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--speed", type=float, default=3.0,
                    help="simulated seconds per rendered second")
    ap.add_argument("--start", type=float, default=52.0,
                    help="simulation time at the first frame [s]")
    ap.add_argument("--still-u", type=float, default=0.66)
    ap.add_argument("--exposure", type=float, default=EXPOSURE)
    ap.add_argument("--force-sim", action="store_true")
    ap.add_argument("--cam", type=str, default="",
                    help="comma-separated key=value camera overrides")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    cam_over = {}
    for kv in filter(None, args.cam.split(",")):
        key, val = kv.split("=")
        cam_over[key.strip()] = float(val)
    CAM.update(cam_over)

    os.makedirs(MEDIA, exist_ok=True)
    res = get_run(force=args.force_sim)
    scene = HeroScene(res)

    n_frames = int(round(args.seconds * FPS))
    dt = float(res.t[1] - res.t[0])
    frames = []
    for n in range(n_frames):
        u = n / max(n_frames - 1, 1)
        t = args.start + n * args.speed / FPS
        k = int(np.clip(round((t - res.t[0]) / dt), 0, len(res.t) - 1))
        frames.append((k, u))

    # ---- still ---------------------------------------------------------
    n_still = int(np.clip(round(args.still_u * (n_frames - 1)), 0, n_frames - 1))
    k_s, u_s = frames[n_still]
    t0 = time.time()
    img = scene.render(k_s, camera_at(scene, u_s, k_s), exposure=args.exposure)
    Image.fromarray(img).save(os.path.join(MEDIA, "hero.png"))
    print(f"hero.png  ({img.shape[1]}x{img.shape[0]})  "
          f"sim t = {res.t[k_s]:.1f} s   [{time.time() - t0:.1f} s]")
    Image.fromarray(img).resize((400, 225), Image.LANCZOS).save(
        os.path.join(MEDIA, "hero_thumb.png"))
    if args.still_only:
        return

    # ---- movie ---------------------------------------------------------
    import imageio_ffmpeg
    path = os.path.join(MEDIA, "hero.mp4")
    w = imageio_ffmpeg.write_frames(
        path, (W, H), fps=FPS, codec="libx264", quality=None, bitrate=None,
        macro_block_size=1, ffmpeg_log_level="error",
        output_params=["-crf", "16", "-pix_fmt", "yuv420p", "-preset", "slow",
                       "-movflags", "+faststart"])
    w.send(None)
    t0 = time.time()
    from tqdm import tqdm
    with Pool(args.workers, initializer=_init, initargs=(res, frames)) as pool:
        for frame in tqdm(pool.imap(_work, range(n_frames), chunksize=2),
                          total=n_frames, ncols=78, desc="hero.mp4"):
            w.send(np.ascontiguousarray(frame))
    w.close()
    print(f"hero.mp4  {n_frames} frames  [{time.time() - t0:.1f} s]")


if __name__ == "__main__":
    main()
