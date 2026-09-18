#!/usr/bin/env python3
"""Render media/hero.png and media/hero.mp4 from the simulation snapshots.

Everything drawn here comes from real output of the N-body run; this script
never touches the physics.  See ``src/galcol/render.py`` for the splatting,
bloom and tone-mapping pipeline, and the README for the list of
rendering-only techniques.

    python3 render_hero.py                 # still + video
    python3 render_hero.py --still-only
    python3 render_hero.py --frames 180    # shorter video, for iterating
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from galcol.render import (Camera, FrameRenderer, luminosity_scatter,  # noqa: E402
                           open_writer, orbit_eye, population_colours)
from galcol.units import PTYPE_BULGE, PTYPE_DISK  # noqa: E402

# --------------------------------------------------------------------------
# Look controls.  These are the only "artistic" numbers in the project.
# --------------------------------------------------------------------------
WIDTH, HEIGHT = 1920, 1080
FPS = 30
SUBFRAMES = 3                # temporal motion-blur samples per output frame
SHUTTER = 0.55               # fraction of the frame interval the shutter is open
FOV_DEG = 32.0

EXPOSURE_TARGET = 7.5        # target HDR level for the 99.7th luminance pct
EXPOSURE_CAL_TIME = 0.50     # Gyr at which auto-exposure is calibrated
EXPOSURE_TRIM = 1.0          # manual multiplier on top of the calibration

HALO_GAIN = 0.20             # dark matter: a whisper of cold haze, nothing more
DOF_STRENGTH = 1.0
DEPTH_SCALE = 55.0           # kpc from the focus plane per DOF class step
# Splat kernel width in kpc.  The Plummer softening is 0.25 kpc for the disk;
# a Gaussian of sigma ~ 1.3 eps is the best match to the half-width of a
# *projected* Plummer sphere, which is the real shape of an N-body particle.
KERNEL_KPC = 0.33
BLOOM_GAINS = (0.82, 0.54, 0.42)
BLOOM_SIGMAS = (3.0, 11.0, 34.0)
SATURATION = 1.38
FRAME_BIAS_Y = 0.0           # centre the pair and the bridge
MAX_BLUR_GYR = 0.004
BULGE_BRIGHTNESS = 2.1       # surface-brightness weight for the old population
SHARP_FRAC = 0.42            # light routed through the near-pixel kernel

# Camera keyframes: (t_gyr, distance_kpc, azimuth_deg, elevation_deg, roll_deg)
# Distances are set from the measured extent of the stellar material at each
# epoch (results/morphology.json), so the subject fills the frame throughout.
CAM_KEYS = [
    (0.00, 176.0,  12.0, 44.0, 118.0),
    (0.35, 122.0,  30.0, 48.0, 110.0),
    (0.50, 110.0,  40.0, 50.0, 105.0),   # the hero frame
    (0.80, 128.0,  62.0, 52.0,  98.0),
    (1.10, 140.0,  86.0, 46.0,  92.0),
    (1.60, 206.0, 118.0, 54.0,  84.0),
    (2.20, 256.0, 152.0, 64.0,  76.0),
]

# Video time warp: (fraction through the video, simulation time in Gyr).
# Real time is spent where the morphology changes fastest.
TIME_WARP = [
    (0.00, 0.00),
    (0.22, 0.36),
    (0.42, 0.56),
    (0.62, 0.88),
    (0.80, 1.22),
    (1.00, 2.20),
]

HERO_TIME = 0.50             # Gyr: shortly after first pericentre


# --------------------------------------------------------------------------
class Snapshots:
    """Lazy, cached access to the snapshot series."""

    def __init__(self, data_dir):
        self.dir = Path(data_dir)
        h = np.load(self.dir / "header.npz", allow_pickle=True)
        self.ptype = h["ptype"]
        self.gid = h["gid"]
        self.star_idx = h["star_idx"]
        self.halo_idx = h["halo_idx"]
        self.mass = h["mass"]
        self.config = json.loads(str(h["config"]))
        self.files = sorted(self.dir.glob("snap_*.npz"))
        if not self.files:
            raise FileNotFoundError(f"no snapshots in {self.dir}")
        self.t_gyr = np.array([float(np.load(f)["t_gyr"]) for f in self.files])
        self.star_ptype = self.ptype[self.star_idx]
        self.star_gid = self.gid[self.star_idx]
        self._cache = {}

    def __len__(self):
        return len(self.files)

    def get(self, i):
        i = int(np.clip(i, 0, len(self.files) - 1))
        if i not in self._cache:
            if len(self._cache) > 6:
                self._cache.pop(next(iter(self._cache)))
            d = np.load(self.files[i])
            self._cache[i] = (d["star_pos"].astype(np.float64),
                              d["halo_pos"].astype(np.float64))
        return self._cache[i]

    def interp(self, t_gyr):
        """Star and halo positions linearly interpolated between snapshots."""
        t = float(np.clip(t_gyr, self.t_gyr[0], self.t_gyr[-1]))
        j = int(np.searchsorted(self.t_gyr, t, side="right") - 1)
        j = int(np.clip(j, 0, len(self.files) - 2))
        t0, t1 = self.t_gyr[j], self.t_gyr[j + 1]
        f = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
        s0, h0 = self.get(j)
        s1, h1 = self.get(j + 1)
        return s0 + (s1 - s0) * f, h0 + (h1 - h0) * f


def nucleus_track(snaps):
    """Midpoint of the two galactic nuclei at every snapshot.

    This is the compositional anchor: the camera aims at the baryonic centre
    of the encounter, so the two nuclei stay in frame while the tails are
    free to sweep asymmetrically out of it.  Before the merger it is the
    midpoint of the two bulges; afterwards the two medians coincide on the
    remnant, so it becomes the remnant centre with no special-casing.
    """
    mid = np.zeros((len(snaps), 3))
    core = [(snaps.star_gid == g) & (snaps.star_ptype == PTYPE_BULGE)
            for g in (0, 1)]
    for i in range(len(snaps)):
        sp, _ = snaps.get(i)
        mid[i] = 0.5 * (np.median(sp[core[0]], axis=0)
                        + np.median(sp[core[1]], axis=0))
    # light smoothing so the camera never twitches
    k = np.array([0.06, 0.24, 0.40, 0.24, 0.06])
    pad = np.pad(mid, ((2, 2), (0, 0)), mode="edge")
    out = np.stack([np.convolve(pad[:, j], k, mode="valid") for j in range(3)], 1)
    return out


def framing_track(snaps, lum):
    """Pre-solve the camera aim point at every snapshot, then smooth it.

    Solving the framing per frame at render time would let it twitch as
    individual particles cross a percentile boundary.  Solving it on the
    snapshot grid and low-pass filtering gives a camera move that is smooth by
    construction, and costs a few seconds once.
    """
    nuc = nucleus_track(snaps)
    out = np.zeros((len(snaps), 3))
    for i in range(len(snaps)):
        t = float(snaps.t_gyr[i])
        cam = camera_at(t, nuc[i])
        sp, _ = snaps.get(i)
        cam = auto_frame(cam, sp, lum, bias_y=FRAME_BIAS_Y)
        out[i] = cam.target
    k = np.array([0.05, 0.12, 0.20, 0.26, 0.20, 0.12, 0.05])
    pad = np.pad(out, ((3, 3), (0, 0)), mode="edge")
    return np.stack([np.convolve(pad[:, j], k, mode="valid") for j in range(3)], 1)


def star_colours(snaps, rng):
    """Colour and per-particle luminosity, fixed once from the initial state."""
    s0, _ = snaps.get(0)
    r_init = np.empty(s0.shape[0])
    for g in (0, 1):
        m = snaps.star_gid == g
        core = m & (snaps.star_ptype == PTYPE_BULGE)
        centre = np.median(s0[core], axis=0) if core.sum() else np.zeros(3)
        r_init[m] = np.linalg.norm(s0[m] - centre, axis=1)
    col = population_colours(snaps.star_ptype, r_init)
    lum = luminosity_scatter(s0.shape[0], rng)
    # the bulge is intrinsically the brightest surface-brightness component
    # The bulge is by far the highest-surface-brightness stellar component of
    # a real spiral; weighting it up is what makes the two nuclei read as the
    # anchors of the frame and drives their bloom haloes.  This is a rendering
    # weight, not a change to any simulated quantity.
    lum = lum * np.where(snaps.star_ptype == PTYPE_BULGE, BULGE_BRIGHTNESS, 1.0)
    return col, lum.astype(np.float32), r_init


def _smoothstep_interp(keys, x):
    """Monotone smooth interpolation through (x, y) keyframes."""
    ks = np.asarray(keys, dtype=float)
    xs, ys = ks[:, 0], ks[:, 1:]
    x = float(np.clip(x, xs[0], xs[-1]))
    j = int(np.clip(np.searchsorted(xs, x, side="right") - 1, 0, len(xs) - 2))
    u = 0.0 if xs[j + 1] <= xs[j] else (x - xs[j]) / (xs[j + 1] - xs[j])
    s = u * u * (3.0 - 2.0 * u)                    # smoothstep easing
    return ys[j] + (ys[j + 1] - ys[j]) * s


def auto_frame(cam, pos, weights, lo=6.0, hi=94.0, bias_y=0.06, n_iter=2):
    """Re-aim the camera so the stellar light is centred in the frame.

    Centring on the nuclei leaves the frame lopsided, because the tidal tails
    are wildly asymmetric; centring on the mean chases whichever tail is
    longest.  Instead the camera is aimed at the midpoint of a robust
    percentile box of the *projected, brightness-weighted* particles, which
    is what a person would call the centre of the picture.  ``bias_y`` sits
    the subject slightly above the geometric centre, which reads better than
    dead centre.
    """
    target = cam.target.copy()
    distance = float(getattr(cam, "distance",
                             np.linalg.norm(cam.eye - cam.target)))
    for _ in range(n_iter):
        sx, sy, depth = cam.project(pos)
        vis = depth > 1e-3
        if not np.any(vis):
            return cam
        w = weights[vis] * (float(np.median(depth[vis])) / depth[vis]) ** 2
        order_x = np.argsort(sx[vis])
        order_y = np.argsort(sy[vis])
        cw = np.cumsum(w[order_x]); cw /= cw[-1]
        x_lo = np.interp(lo / 100, cw, sx[vis][order_x])
        x_hi = np.interp(hi / 100, cw, sx[vis][order_x])
        cw = np.cumsum(w[order_y]); cw /= cw[-1]
        y_lo = np.interp(lo / 100, cw, sy[vis][order_y])
        y_hi = np.interp(hi / 100, cw, sy[vis][order_y])
        cx = 0.5 * (x_lo + x_hi)
        cy = 0.5 * (y_lo + y_hi) + bias_y * cam.height
        right, up = cam.basis[0], cam.basis[1]
        scale = distance / cam.focal
        target = target + (cx - 0.5 * cam.width) * scale * right \
                        - (cy - 0.5 * cam.height) * scale * up
        new = Camera(cam.eye + (target - cam.target), target=target,
                     fov_deg=np.degrees(cam.fov), width=cam.width,
                     height=cam.height)
        new.basis = np.stack([right, up, new.basis[2]])
        new.distance = distance
        cam = new
    return cam


def camera_at(t_gyr, target=(0.0, 0.0, 0.0)):
    d, az, el, roll = _smoothstep_interp(CAM_KEYS, t_gyr)
    target = np.asarray(target, dtype=float)
    cam = Camera(orbit_eye(target, d, az, el), target=target,
                 fov_deg=FOV_DEG, width=WIDTH, height=HEIGHT, roll_deg=roll)
    cam.distance = float(d)
    return cam


def video_time(u):
    """Video progress u in [0,1] -> simulation time in Gyr."""
    ks = np.asarray(TIME_WARP, dtype=float)
    return float(np.interp(np.clip(u, 0, 1), ks[:, 0], ks[:, 1]))


# --------------------------------------------------------------------------
def render_frame(snaps, fr, colours, lum, t_gyr, dt_blur=0.0, subframes=1,
                 exposure=1.0, track=None):
    fr.exposure = exposure
    fr.clear()
    ts = ([t_gyr] if subframes <= 1 else
          t_gyr + np.linspace(-0.5, 0.5, subframes) * dt_blur)
    cam = camera_at(t_gyr, _track_at(snaps, track, t_gyr))
    focus = float(np.linalg.norm(cam.eye))
    fr.set_kernel_from_camera(cam, KERNEL_KPC, cam.distance)
    w = 1.0 / len(ts)
    for tt in ts:
        spos, hpos = snaps.interp(tt)
        fr.add_stars(cam, spos, colours, lum, sub_weight=w,
                     focus_depth=focus, depth_scale=DEPTH_SCALE,
                     ref_depth=focus)
    spos, hpos = snaps.interp(t_gyr)
    fr.add_halo(cam, hpos, 1.0)
    return fr.resolve()


def _track_at(snaps, track, t_gyr):
    if track is None:
        return np.zeros(3)
    return np.array([np.interp(t_gyr, snaps.t_gyr, track[:, j]) for j in range(3)])


def calibrate_exposure(snaps, fr, colours, lum, t_gyr, track=None):
    """Pick the exposure so the bright cores land just below clipping."""
    fr.exposure = 1.0
    fr.clear()
    cam = camera_at(t_gyr, _track_at(snaps, track, t_gyr))
    focus = float(np.linalg.norm(cam.eye))
    fr.set_kernel_from_camera(cam, KERNEL_KPC, cam.distance)
    spos, hpos = snaps.interp(t_gyr)
    fr.add_stars(cam, spos, colours, lum, focus_depth=focus,
                 depth_scale=DEPTH_SCALE, ref_depth=focus)
    fr.add_halo(cam, hpos, 1.0)
    from scipy.ndimage import gaussian_filter
    from galcol.render import _upsample
    s0 = fr.base_sigma
    acc = fr.bufs[0].copy()
    for buf, fac, mult in zip(fr.bufs[1:], fr.DOF_FACTORS[1:],
                              fr.DOF_SIGMA_MULT[1:]):
        sig_low = np.sqrt(max((s0 * mult) ** 2 - s0 ** 2, 0.0)) / fac
        b = gaussian_filter(buf, sigma=(sig_low, sig_low, 0))
        acc += _upsample(b, fac)[:HEIGHT, :WIDTH, :] * np.float32(1.0 / (fac * fac))
    acc = gaussian_filter(acc, sigma=(s0, s0, 0), truncate=3.0)
    if fr.sharp_frac > 0.0:
        acc = acc + gaussian_filter(fr.sharp_buf,
                                    sigma=(fr.sharp_sigma, fr.sharp_sigma, 0),
                                    truncate=3.0)
    lumin = acc @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    hi = float(np.percentile(lumin[lumin > 0], 99.7)) if np.any(lumin > 0) else 1.0
    return EXPOSURE_TRIM * EXPOSURE_TARGET / max(hi, 1e-12)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "collision"))
    ap.add_argument("--out", default=str(ROOT / "media"))
    ap.add_argument("--frames", type=int, default=540)
    ap.add_argument("--still-only", action="store_true")
    ap.add_argument("--video-only", action="store_true")
    ap.add_argument("--exposure", type=float, default=None)
    ap.add_argument("--hero-time", type=float, default=HERO_TIME)
    ap.add_argument("--still-name", default="hero.png")
    ap.add_argument("--kernel", type=float, default=None)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    snaps = Snapshots(args.data)
    print(f"{len(snaps)} snapshots, t = 0 .. {snaps.t_gyr[-1]:.3f} Gyr, "
          f"{snaps.star_idx.size} star particles", flush=True)

    rng = np.random.default_rng(4242)
    colours, lum, _ = star_colours(snaps, rng)
    print("solving camera framing ...", flush=True)
    track = framing_track(snaps, lum)

    fr = FrameRenderer(width=WIDTH, height=HEIGHT, halo_gain=HALO_GAIN,
                       dof_strength=DOF_STRENGTH, bloom_gains=BLOOM_GAINS,
                       bloom_sigmas=BLOOM_SIGMAS, saturation=SATURATION,
                       sharp_frac=SHARP_FRAC)
    if args.kernel is not None:
        globals()["KERNEL_KPC"] = args.kernel
    if args.exposure is not None:
        exposure = args.exposure
    else:
        exposure = calibrate_exposure(snaps, fr, colours, lum,
                                      min(EXPOSURE_CAL_TIME, snaps.t_gyr[-1]),
                                      track=track)
    print(f"exposure = {exposure:.4g}", flush=True)

    if not args.video_only:
        import imageio.v3 as iio
        t_hero = min(args.hero_time, snaps.t_gyr[-1])
        img = render_frame(snaps, fr, colours, lum, t_hero,
                           dt_blur=0.0014, subframes=8, exposure=exposure,
                           track=track)
        iio.imwrite(out / args.still_name, img)
        small = img[::(HEIGHT // 225), ::(WIDTH // 400)][:225, :400]
        iio.imwrite(out / ("phone_" + args.still_name), small)
        print(f"wrote {out / args.still_name}  ({img.shape[1]}x{img.shape[0]}) "
              f"at t = {t_hero:.3f} Gyr", flush=True)

    if args.still_only:
        return 0

    n = args.frames
    writer = open_writer(out / "hero.mp4", size=(WIDTH, HEIGHT), fps=FPS)
    try:
        from tqdm import tqdm
        loop = tqdm(range(n), desc="render", unit="frame", mininterval=5.0)
    except ImportError:
        loop = range(n)
    for i in loop:
        u = i / max(n - 1, 1)
        t = video_time(u)
        t_next = video_time(min(1.0, (i + 1) / max(n - 1, 1)))
        img = render_frame(snaps, fr, colours, lum, t,
                           dt_blur=min(abs(t_next - t) * SHUTTER, MAX_BLUR_GYR),
                           subframes=SUBFRAMES, exposure=exposure,
                           track=track)
        writer.send(np.ascontiguousarray(img))
    writer.close()
    print(f"wrote {out / 'hero.mp4'}  ({n} frames, {n / FPS:.1f} s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
