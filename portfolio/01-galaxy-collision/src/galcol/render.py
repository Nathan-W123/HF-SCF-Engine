"""Cinematic renderer for the N-body output.

Nothing here touches the physics: it consumes the compressed snapshots
written by :mod:`galcol.simulate`.  The pipeline is

    perspective projection  ->  additive splatting into a float HDR buffer
    ->  depth-of-field size classes  ->  multi-scale bloom  ->  filmic
        tone map  ->  8-bit RGB

which is what makes ~10^5 point masses read as coherent luminous structure
instead of a scatter plot.

Rendering-only techniques (they change no simulated quantity, and are all
listed in the README):

  * per-particle luminosity scatter: an N-body "star" particle stands for
    ~2x10^5 Msun of stars, so its light is really an ensemble; a log-normal
    scatter is applied so the field looks like stars rather than a smooth
    density map;
  * depth of field: particles far from the focus plane are splatted with a
    wider kernel;
  * temporal motion blur: positions are linearly interpolated between
    snapshots and several sub-frames are accumulated into one output frame;
  * the dark-matter halo is drawn only as a heavily blurred, very low gain
    haze.

Colour is *not* an arbitrary palette: it is keyed to the particle's simulated
population and its initial radius in its host disk (a stand-in for the stellar
population gradient of a real spiral), so the tidal tails -- which are made of
outer-disk material -- come out blue and the bulges come out amber.
"""

from __future__ import annotations

import numpy as np
from numba import njit
from scipy.ndimage import gaussian_filter, uniform_filter

# --------------------------------------------------------------------------
# Camera
# --------------------------------------------------------------------------
class Camera:
    """Simple perspective look-at camera."""

    def __init__(self, eye, target=(0.0, 0.0, 0.0), up=(0.0, 0.0, 1.0),
                 fov_deg=30.0, width=1920, height=1080, roll_deg=0.0):
        self.eye = np.asarray(eye, dtype=np.float64)
        self.target = np.asarray(target, dtype=np.float64)
        self.width = int(width)
        self.height = int(height)
        self.fov = np.radians(fov_deg)
        fwd = self.target - self.eye
        fwd /= np.linalg.norm(fwd)
        up = np.asarray(up, dtype=np.float64)
        right = np.cross(fwd, up)
        nr = np.linalg.norm(right)
        if nr < 1e-8:                       # degenerate: pick any perpendicular
            right = np.cross(fwd, np.array([1.0, 0.0, 0.0]))
            nr = np.linalg.norm(right)
        right /= nr
        true_up = np.cross(right, fwd)
        if roll_deg:
            c, s = np.cos(np.radians(roll_deg)), np.sin(np.radians(roll_deg))
            right, true_up = c * right + s * true_up, -s * right + c * true_up
        self.basis = np.stack([right, true_up, fwd])    # rows
        self.focal = 0.5 * self.height / np.tan(0.5 * self.fov)

    def project(self, pos):
        """World (N,3) -> screen x, y (pixels) and depth along the view axis."""
        rel = pos - self.eye
        cam = rel @ self.basis.T
        depth = cam[:, 2]
        d = np.maximum(depth, 1e-6)
        sx = 0.5 * self.width + self.focal * cam[:, 0] / d
        sy = 0.5 * self.height - self.focal * cam[:, 1] / d
        return sx, sy, depth


def orbit_eye(target, distance, azimuth_deg, elevation_deg):
    """Camera position on a sphere around ``target``."""
    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)
    d = np.asarray(target, dtype=float) + distance * np.array(
        [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    return d


# --------------------------------------------------------------------------
# Splatting
# --------------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def _splat_kernel(buf, x, y, rgb, weight):
    H = buf.shape[0]
    W = buf.shape[1]
    for i in range(x.shape[0]):
        w = weight[i]
        if w <= 0.0:
            continue
        xf = x[i]
        yf = y[i]
        if not (xf >= 0.0 and xf < W - 1 and yf >= 0.0 and yf < H - 1):
            continue
        x0 = int(xf)
        y0 = int(yf)
        fx = xf - x0
        fy = yf - y0
        w00 = (1.0 - fx) * (1.0 - fy) * w
        w10 = fx * (1.0 - fy) * w
        w01 = (1.0 - fx) * fy * w
        w11 = fx * fy * w
        r = rgb[i, 0]; g = rgb[i, 1]; b = rgb[i, 2]
        buf[y0, x0, 0] += w00 * r; buf[y0, x0, 1] += w00 * g; buf[y0, x0, 2] += w00 * b
        buf[y0, x0 + 1, 0] += w10 * r; buf[y0, x0 + 1, 1] += w10 * g; buf[y0, x0 + 1, 2] += w10 * b
        buf[y0 + 1, x0, 0] += w01 * r; buf[y0 + 1, x0, 1] += w01 * g; buf[y0 + 1, x0, 2] += w01 * b
        buf[y0 + 1, x0 + 1, 0] += w11 * r; buf[y0 + 1, x0 + 1, 1] += w11 * g; buf[y0 + 1, x0 + 1, 2] += w11 * b


def splat(buf, x, y, rgb, weight):
    """Additively accumulate weighted RGB energy with bilinear deposition.

    ``buf`` is (H, W, 3) float32.  The shared playbook recommends
    ``np.bincount`` over ``np.add.at``; that is right for pure NumPy, but with
    a 1920x1080 target ``minlength = 2.07e6`` makes every bincount allocate and
    zero a 16 MB accumulator, which dominates when only ~10^4-10^5 particles
    are being deposited.  A one-line numba scatter avoids the allocation
    entirely and is ~50x faster here.
    """
    _splat_kernel(buf, np.ascontiguousarray(x, np.float32),
                  np.ascontiguousarray(y, np.float32),
                  np.ascontiguousarray(rgb, np.float32),
                  np.ascontiguousarray(weight, np.float32))
    return buf


def _upsample(a, factor, smooth=False):
    """Nearest-neighbour upsample, ~40x faster than ``scipy.ndimage.zoom`` at
    1080p.  Nearest alone leaves visible `factor`-pixel blocks once the tone
    curve stretches the dark end, so either a later full-resolution blur must
    cover them (the star buffers) or ``smooth=True`` runs a box filter of the
    same width, which turns nearest into proper bilinear (the bloom)."""
    if factor == 1:
        return a
    out = np.repeat(np.repeat(a, factor, axis=0), factor, axis=1)
    if smooth:
        out = uniform_filter(out, size=(factor, factor, 1))
    return out


# --------------------------------------------------------------------------
# Tone mapping
# --------------------------------------------------------------------------
_LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)

_DITHER_CACHE = {}


def _dither(shape):
    """Fixed-pattern sub-LSB dither, cached per frame size.

    The dark-matter haze is an extremely smooth, extremely dim gradient.  After
    the tone curve and the 1/2.2 gamma, neighbouring 8-bit levels are far apart
    in scene luminance, so the haze quantises into visible concentric contour
    rings around each galaxy -- the image looks like a topographic map.  Adding
    a uniform [0,1) offset before truncation turns that structured error into
    unstructured noise below the quantisation step, which is invisible at 1080p.
    The pattern is fixed (not re-drawn per frame) so it does not crawl in video.
    """
    key = tuple(shape)
    if key not in _DITHER_CACHE:
        _DITHER_CACHE[key] = np.random.default_rng(90210).random(
            shape).astype(np.float32)
    return _DITHER_CACHE[key]


def _filmic(x):
    """ACES-like filmic curve; keeps highlights from clipping flat."""
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)


def tonemap(buf, exposure=1.0, bloom_sigmas=(2.0, 8.0, 28.0),
            bloom_gains=(0.55, 0.30, 0.18), gamma=2.2, saturation=1.12,
            bloom_scale=4, hue_preserve=0.45, dither=True):
    """Multi-scale bloom + filmic tone map -> uint8 RGB.

    ``bloom_scale`` runs the bloom convolutions on a downsampled copy, which
    is what makes a 1080p frame affordable.

    ``hue_preserve`` blends two ways of applying the filmic curve.  Applied
    per channel it desaturates highlights toward white, which erases exactly
    the thing this project wants visible: the amber of the old bulge
    population against the blue of the outer disk.  Applied to *luminance*,
    with the chroma carried through unchanged, hue survives into the
    highlights.  The blend keeps some of the per-channel roll-off (which
    looks natural on the very brightest cores) while retaining most of the
    population colour.
    """
    hdr = buf * np.float32(exposure)
    if bloom_scale > 1:
        H, W = hdr.shape[0], hdr.shape[1]
        small = np.ascontiguousarray(hdr[::bloom_scale, ::bloom_scale, :])
        glow_s = np.zeros_like(small)
        for sg, g in zip(bloom_sigmas, bloom_gains):
            glow_s += np.float32(g) * gaussian_filter(
                small, sigma=(sg / bloom_scale, sg / bloom_scale, 0))
        glow = _upsample(glow_s, bloom_scale, smooth=True)[:H, :W, :]
        if glow.shape[:2] != (H, W):
            glow = np.pad(glow, [(0, H - glow.shape[0]),
                                 (0, W - glow.shape[1]), (0, 0)], mode="edge")
    else:
        glow = np.zeros_like(hdr)
        for sg, g in zip(bloom_sigmas, bloom_gains):
            glow += np.float32(g) * gaussian_filter(hdr, sigma=(sg, sg, 0))
    hdr = hdr + glow
    per_channel = _filmic(hdr)
    if hue_preserve > 0.0:
        L = np.maximum(hdr @ _LUMA, 1e-8)
        scaled = hdr * (_filmic(L) / L)[..., None]
        ldr = (1.0 - hue_preserve) * per_channel + hue_preserve * np.clip(scaled, 0, 1)
    else:
        ldr = per_channel
    lum = ldr @ _LUMA
    ldr = np.clip(lum[..., None] + (ldr - lum[..., None]) * saturation, 0, 1)
    val = np.power(ldr, 1.0 / gamma) * 255.0
    if dither:
        val = val + _dither(val.shape)
    else:
        val = val + 0.5
    return np.clip(val, 0.0, 255.0).astype(np.uint8)


# --------------------------------------------------------------------------
# Population colours
# --------------------------------------------------------------------------
BULGE_RGB = np.array([1.00, 0.55, 0.20], np.float32)     # old, warm amber
DISK_IN_RGB = np.array([1.00, 0.80, 0.48], np.float32)   # inner disk, yellow-white
DISK_MID_RGB = np.array([0.98, 0.88, 0.72], np.float32)  # mid disk, cream
DISK_OUT_RGB = np.array([0.40, 0.62, 1.00], np.float32)  # outer disk, blue-white
HALO_RGB = np.array([0.26, 0.30, 0.62], np.float32)      # faint cold haze


def population_colours(ptype, r_init, r_scale=3.0, r_blue=12.0):
    """RGB per star particle from population identity + initial disk radius.

    The initial radius stands in for the stellar-population gradient of a real
    spiral: inner disk older/redder, outer disk younger/bluer.  Because tidal
    tails are drawn from the outer disk, they inherit the blue end of the ramp
    automatically.
    """
    ptype = np.asarray(ptype)
    r_init = np.asarray(r_init, dtype=np.float32)
    frac = np.clip((r_init - 0.35 * r_scale) / (r_blue - 0.35 * r_scale), 0.0, 1.0)
    frac = frac ** 0.70
    # three-stop ramp: amber-white core -> white mid-disk -> blue outer disk,
    # so the tails (drawn from the outer disk) separate in hue from the arms
    f = np.clip(frac * 2.0, 0.0, 1.0)[:, None]
    g = np.clip(frac * 2.0 - 1.0, 0.0, 1.0)[:, None]
    col = (DISK_IN_RGB[None, :] * (1 - f) + DISK_MID_RGB[None, :] * f) * (1 - g) \
        + DISK_OUT_RGB[None, :] * g
    col[ptype == 1] = BULGE_RGB
    return np.ascontiguousarray(col, dtype=np.float32)


def luminosity_scatter(n, rng, sigma_dex=0.32):
    """Log-normal per-particle luminosity scatter (rendering only)."""
    lw = rng.normal(0.0, sigma_dex * np.log(10.0), n)
    lw = np.exp(lw - 0.5 * (sigma_dex * np.log(10.0)) ** 2)   # unit mean
    return lw.astype(np.float32)


# --------------------------------------------------------------------------
# Frame assembly
# --------------------------------------------------------------------------
class FrameRenderer:
    """Renders one frame from star positions plus a halo subsample.

    Kernel size is physical, not arbitrary.  An N-body particle is not a
    point: it is a Plummer sphere of softening length ``eps``, which is the
    smallest structure the simulation resolves.  The base splat kernel is
    therefore the *projected softening*,

        sigma_px = focal_length_px * KERNEL_KPC / camera_distance_kpc,

    so the image softens exactly as much as the simulation is blurred, and it
    softens automatically as the camera pushes in.  Depth of field then
    widens that kernel for particles away from the focus plane.

    Three depth-of-field classes are splatted into buffers whose *resolution
    matches their kernel* -- sharp at 1080p, medium at half, soft at quarter.
    They are summed and blurred ONCE at full resolution with the in-focus
    sigma; that single pass both gives the in-focus class its kernel and
    erases the block structure left by the cheap nearest-neighbour upsample.

    On top of that the renderer is *dual scale*.  A fraction ``sharp_frac`` of
    each in-focus particle's light is deposited into a separate buffer that is
    blurred with a near-pixel kernel, and composited over the diffuse result
    before bloom.  Without it every particle is a soft blob of the same size
    and the tails read as cotton wool; with it each particle is a crisp point
    sitting inside a smooth envelope, which is what gives a real astronomical
    image its bite.  Out-of-focus particles get no sharp component at all, so
    depth of field still reads.
    """

    DOF_FACTORS = (1, 2, 4)
    DOF_SIGMA_MULT = (1.0, 1.65, 2.8)

    def __init__(self, width=1920, height=1080,
                 dof_split=(0.34, 0.95), dof_strength=1.0,
                 halo_gain=0.030, halo_sigma=13.0, halo_scale=4,
                 exposure=1.0, star_gain=1.0, base_sigma=2.6,
                 sharp_frac=0.42, sharp_sigma=0.85,
                 bloom_sigmas=(3.0, 11.0, 34.0), bloom_gains=(0.55, 0.34, 0.24),
                 bloom_scale=4, saturation=1.20, gamma=2.2):
        self.W = int(width)
        self.H = int(height)
        self.dof_split = dof_split
        self.dof_strength = dof_strength
        self.halo_gain = halo_gain
        self.halo_sigma = halo_sigma
        self.halo_scale = halo_scale
        self.exposure = exposure
        self.star_gain = star_gain
        self.base_sigma = base_sigma
        self.sharp_frac = float(sharp_frac)
        self.sharp_sigma = float(sharp_sigma)
        self.bloom_sigmas = bloom_sigmas
        self.bloom_gains = bloom_gains
        self.bloom_scale = bloom_scale
        self.saturation = saturation
        self.gamma = gamma
        self.bufs = [np.zeros((self.H // f, self.W // f, 3), np.float32)
                     for f in self.DOF_FACTORS]
        self.sharp_buf = np.zeros((self.H, self.W, 3), np.float32)
        hh = max(1, self.H // halo_scale)
        hw = max(1, self.W // halo_scale)
        self.halo_buf = np.zeros((hh, hw, 3), np.float32)

    # -- kernel sizing ----------------------------------------------------
    def set_kernel_from_camera(self, cam, kernel_kpc, distance,
                               lo=1.3, hi=6.5):
        self.base_sigma = float(np.clip(cam.focal * kernel_kpc / max(distance, 1e-6),
                                        lo, hi))
        return self.base_sigma

    def clear(self):
        for b in self.bufs:
            b.fill(0.0)
        self.sharp_buf.fill(0.0)
        self.halo_buf.fill(0.0)

    # -- deposition -------------------------------------------------------
    def add_stars(self, cam: Camera, pos, colours, weights, sub_weight=1.0,
                  focus_depth=None, depth_scale=55.0, ref_depth=None):
        sx, sy, depth = cam.project(pos)
        vis = depth > 1e-3
        if not np.any(vis):
            return
        if ref_depth is None:
            ref_depth = float(np.median(depth[vis]))
        if focus_depth is None:
            focus_depth = ref_depth
        # Inverse-square flux fall-off with camera distance -- this is the
        # physical reason near-side structure reads brighter, and it is what
        # gives the tidal tails their sense of depth.
        w = weights * (ref_depth / np.maximum(depth, 1e-3)) ** 2
        w = np.where(vis, w, 0.0).astype(np.float32)
        w *= np.float32(sub_weight * self.star_gain)
        key = self.dof_strength * np.abs(depth - focus_depth) / depth_scale
        cls = np.digitize(key, self.dof_split)
        for k, fac in enumerate(self.DOF_FACTORS):
            m = cls == k
            if not np.any(m):
                continue
            inv = 1.0 / fac
            wk = w[m]
            if k == 0 and self.sharp_frac > 0.0:
                # in focus: split into a crisp point plus a diffuse halo
                splat(self.sharp_buf, sx[m], sy[m], colours[m],
                      wk * np.float32(self.sharp_frac))
                wk = wk * np.float32(1.0 - self.sharp_frac)
            splat(self.bufs[k], sx[m] * inv, sy[m] * inv,
                  colours[m], wk * np.float32(fac * fac))

    def add_halo(self, cam: Camera, pos, weight=1.0):
        if self.halo_gain <= 0 or pos.shape[0] == 0:
            return
        sx, sy, depth = cam.project(pos)
        s = self.halo_scale
        w = np.where(depth > 1e-3, weight, 0.0).astype(np.float32)
        col = np.repeat(HALO_RGB[None, :], pos.shape[0], axis=0)
        splat(self.halo_buf, sx / s, sy / s, col, w)

    # -- resolve ----------------------------------------------------------
    def resolve(self):
        s0 = self.base_sigma
        acc = self.bufs[0].copy()
        for buf, fac, mult in zip(self.bufs[1:], self.DOF_FACTORS[1:],
                                  self.DOF_SIGMA_MULT[1:]):
            target = s0 * mult
            # the final full-resolution pass contributes s0 in quadrature
            sig_low = np.sqrt(max(target ** 2 - s0 ** 2, 0.0)) / fac
            b = gaussian_filter(buf, sigma=(sig_low, sig_low, 0)) if sig_low > 0.05 else buf
            acc += _upsample(b, fac)[:self.H, :self.W, :] * np.float32(1.0 / (fac * fac))
        if self.halo_gain > 0:
            f = self.halo_scale
            sig_low = max(self.halo_sigma, 1.0)
            hb = gaussian_filter(self.halo_buf, sigma=(sig_low, sig_low, 0))
            hb = _upsample(hb, f)[:self.H, :self.W, :] * np.float32(1.0 / (f * f))
            acc += np.float32(self.halo_gain) * hb
        acc = gaussian_filter(acc, sigma=(s0, s0, 0), truncate=3.0)
        if self.sharp_frac > 0.0:
            acc += gaussian_filter(self.sharp_buf,
                                   sigma=(self.sharp_sigma, self.sharp_sigma, 0),
                                   truncate=3.0)
        return tonemap(acc, exposure=self.exposure,
                       bloom_sigmas=self.bloom_sigmas,
                       bloom_gains=self.bloom_gains,
                       bloom_scale=self.bloom_scale,
                       gamma=self.gamma, saturation=self.saturation)


# --------------------------------------------------------------------------
# Video writer (proven configuration from the shared rendering playbook)
# --------------------------------------------------------------------------
def open_writer(path, size=(1920, 1080), fps=30, crf=16, preset="slow"):
    import imageio_ffmpeg
    w = imageio_ffmpeg.write_frames(
        str(path), size, fps=fps, codec="libx264",
        quality=None, bitrate=None, macro_block_size=1,
        ffmpeg_log_level="error",
        output_params=["-crf", str(crf), "-pix_fmt", "yuv420p",
                       "-preset", preset, "-movflags", "+faststart"],
    )
    w.send(None)          # priming send is REQUIRED
    return w
