"""Additive splatting, multi-scale bloom and filmic tone mapping.

Follows the shared portfolio rendering playbook: emitters are accumulated into a
float HDR buffer with bilinear deposition (``np.bincount``, never ``np.add.at``),
then bloomed at three scales and tone-mapped with an ACES-like filmic curve.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

__all__ = ["splat", "bloom", "tonemap", "open_writer", "upsample_bilinear"]


def splat(buf, x, y, rgb, weight):
    """Additively accumulate weighted RGB energy at float pixel coords.

    ``buf`` is ``(H, W, 3)`` float32; ``x``/``y`` are float pixel coordinates;
    ``rgb`` is ``(N, 3)``; ``weight`` is ``(N,)``.
    """
    h, w, _ = buf.shape
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    m &= (x >= 0) & (x < w - 1) & (y >= 0) & (y < h - 1) & (weight > 0)
    if not np.any(m):
        return buf
    x, y, rgb, weight = x[m], y[m], np.asarray(rgb)[m], np.asarray(weight)[m]
    x0 = x.astype(np.int64)
    y0 = y.astype(np.int64)
    fx = x - x0
    fy = y - y0
    base = y0 * w + x0
    # All four bilinear corners are concatenated so that only three bincounts
    # (one per colour channel) are issued instead of twelve: the dominant cost
    # is the W*H-long accumulator each call allocates, not the deposits.
    idx = np.concatenate([base, base + 1, base + w, base + w + 1])
    ww = np.concatenate([
        (1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy
    ]) * np.tile(weight, 4)
    rgb4 = np.tile(rgb, (4, 1))
    flat = buf.reshape(-1, 3)
    n = w * h
    for c in range(3):
        flat[:, c] += np.bincount(
            idx, weights=ww * rgb4[:, c], minlength=n
        ).astype(np.float32)
    return buf


def upsample_bilinear(small, height, width):
    """Separable bilinear upsample of an ``(h, w, 3)`` float array.

    ``scipy.ndimage.zoom`` costs ~0.3 s at 1080p and was by far the single
    largest per-frame cost; doing the two 1-D interpolations explicitly is an
    order of magnitude faster and visually identical.
    """
    h, w = small.shape[:2]
    # Clamp the sample coordinates *before* taking the fractional part so the
    # border rows/columns are edge-extended rather than linearly extrapolated
    # (extrapolation would push values outside the source range).
    yi = np.clip((np.arange(height, dtype=np.float32) + 0.5) * (h / height) - 0.5,
                 0.0, h - 1.0)
    xi = np.clip((np.arange(width, dtype=np.float32) + 0.5) * (w / width) - 0.5,
                 0.0, w - 1.0)
    y0 = np.floor(yi).astype(np.int32)
    x0 = np.floor(xi).astype(np.int32)
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    fy = (yi - y0).astype(np.float32)[:, None, None]
    fx = (xi - x0).astype(np.float32)[None, :, None]
    rows = small[y0] * (1.0 - fy) + small[y1] * fy          # (height, w, 3)
    return rows[:, x0] * (1.0 - fx) + rows[:, x1] * fx      # (height, width, 3)


def _blur_at(hdr, sigma, step, truncate=3.0):
    """Gaussian blur evaluated on a ``1/step`` copy and resampled back up."""
    if step == 1:
        return gaussian_filter(hdr, sigma=(sigma, sigma, 0), truncate=truncate)
    small = np.ascontiguousarray(hdr[::step, ::step])
    blurred = gaussian_filter(small, sigma=(sigma / step, sigma / step, 0),
                              truncate=truncate)
    return upsample_bilinear(blurred, hdr.shape[0], hdr.shape[1])


def bloom(hdr, sigmas=(2.0, 8.0, 28.0), gains=(0.55, 0.30, 0.18)):
    """Three-scale additive bloom.

    Wide kernels are evaluated on downsampled copies -- a Gaussian of sigma 28
    carries no detail a 1/8 copy cannot represent, and it is ~50x cheaper.
    """
    glow = None
    for sg, g in zip(sigmas, gains):
        step = 1 if sg < 4.0 else (2 if sg < 12.0 else 8)
        b = _blur_at(hdr, sg, step) * np.float32(g)
        glow = b if glow is None else glow + b
    return glow


def tonemap(buf, exposure=1.0, bloom_sigmas=(2.0, 8.0, 28.0),
            bloom_gains=(0.55, 0.30, 0.18), gamma=2.2, saturation=1.12):
    """HDR float buffer -> 8-bit sRGB frame (ACES-like filmic curve)."""
    hdr = np.asarray(buf, dtype=np.float32) * np.float32(exposure)
    hdr = hdr + bloom(hdr, bloom_sigmas, bloom_gains)
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    ldr = np.clip((hdr * (a * hdr + b)) / (hdr * (c * hdr + d) + e), 0.0, 1.0)
    lum = ldr @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    ldr = np.clip(lum[..., None] + (ldr - lum[..., None]) * saturation, 0.0, 1.0)
    return (np.power(ldr, 1.0 / gamma) * 255.0 + 0.5).astype(np.uint8)


def open_writer(path, size=(1920, 1080), fps=30, crf=16, preset="slow"):
    """Proven 1080p H.264 writer from the portfolio rendering playbook."""
    import imageio_ffmpeg

    w = imageio_ffmpeg.write_frames(
        str(path), size, fps=fps, codec="libx264",
        quality=None, bitrate=None, macro_block_size=1,
        ffmpeg_log_level="error",
        output_params=["-crf", str(crf), "-pix_fmt", "yuv420p",
                       "-preset", preset, "-movflags", "+faststart"],
    )
    w.send(None)  # priming send is REQUIRED
    return w
