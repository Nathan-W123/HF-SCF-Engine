# Cinematic rendering playbook (shared reference)

All three portfolio projects must emit `media/hero.png` (1920x1080) and
`media/hero.mp4` (1920x1080, 30 fps) generated **entirely from real simulation
output**. This file records the environment facts and the proven code paths so
each project does not have to rediscover them.

## Environment facts

- Python 3.11.15, system interpreter (`/usr/local/bin/python3`), no venv.
- numpy 2.4.6, scipy 1.17.1, matplotlib 3.11.2, numba 0.67.0, pytest 9.1.1,
  imageio 2.37.4, imageio-ffmpeg 0.6.0, Pillow 12.3.0, tqdm 4.70.1.
- 4 CPU cores, ~15 GB RAM, no GPU, **no display**.
- Always `import matplotlib; matplotlib.use("Agg")` before pyplot.
- `ffmpeg` is **not on PATH**. Use the static binary shipped with
  imageio-ffmpeg: `imageio_ffmpeg.get_ffmpeg_exe()` (ffmpeg 7.0.2, libx264).

## Proven 1080p H.264 writer

```python
import numpy as np, imageio_ffmpeg

def open_writer(path, size=(1920, 1080), fps=30, crf=16, preset="slow"):
    w = imageio_ffmpeg.write_frames(
        path, size, fps=fps, codec="libx264",
        quality=None, bitrate=None, macro_block_size=1,
        ffmpeg_log_level="error",
        output_params=["-crf", str(crf), "-pix_fmt", "yuv420p",
                       "-preset", preset, "-movflags", "+faststart"],
    )
    w.send(None)          # priming send is REQUIRED
    return w

w = open_writer("media/hero.mp4")
for frame in frames:      # frame: uint8 (H, W, 3) RGB
    w.send(np.ascontiguousarray(frame))
w.close()
```

`macro_block_size=1` avoids silent resizing. `quality=None, bitrate=None` is
required or they fight with `-crf`.

## Why not `plt.scatter`

Marker scatter plots read as "paper figure". For anything made of many
emitters (stars, exhaust, glowing trails) render by **additive splatting into a
float RGB buffer**, then tone-map. This is both faster and far better looking.

```python
import numpy as np
from scipy.ndimage import gaussian_filter

def splat(buf, x, y, rgb, weight, W, H):
    """Additively accumulate weighted RGB energy at float pixel coords
    using bilinear deposition. buf: (H, W, 3) float32."""
    m = (x >= 0) & (x < W - 1) & (y >= 0) & (y < H - 1)
    x, y, rgb, weight = x[m], y[m], rgb[m], weight[m]
    x0 = x.astype(np.int32); y0 = y.astype(np.int32)
    fx = x - x0; fy = y - y0
    for dx, dy, wgt in ((0, 0, (1-fx)*(1-fy)), (1, 0, fx*(1-fy)),
                        (0, 1, (1-fx)*fy),     (1, 1, fx*fy)):
        idx = (y0 + dy) * W + (x0 + dx)
        ww = (wgt * weight).astype(np.float32)
        for c in range(3):
            np.add.at  # too slow -- use bincount instead:
            buf.reshape(-1, 3)[:, c] += np.bincount(
                idx, weights=ww * rgb[:, c], minlength=W * H).astype(np.float32)
    return buf
```

`np.bincount` is the fast accumulation primitive; never use `np.add.at` in a
per-frame loop. If a project needs per-particle finite size, splat into a
half- or quarter-resolution buffer and upsample, or convolve the accumulated
buffer with a small Gaussian.

## Bloom + filmic tone map (turns raw energy into a cinematic frame)

```python
def tonemap(buf, exposure=1.0, bloom_sigmas=(2.0, 8.0, 28.0),
            bloom_gains=(0.55, 0.30, 0.18), gamma=2.2, saturation=1.12):
    hdr = buf * exposure
    glow = np.zeros_like(hdr)
    for s, g in zip(bloom_sigmas, bloom_gains):
        glow += g * gaussian_filter(hdr, sigma=(s, s, 0))
    hdr = hdr + glow
    # ACES-ish filmic curve: keeps highlights from clipping to flat white
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    ldr = np.clip((hdr * (a * hdr + b)) / (hdr * (c * hdr + d) + e), 0.0, 1.0)
    lum = ldr @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    ldr = np.clip(lum[..., None] + (ldr - lum[..., None]) * saturation, 0, 1)
    return (np.power(ldr, 1.0 / gamma) * 255.0 + 0.5).astype(np.uint8)
```

Multi-scale bloom (three sigmas) is what separates "glowing" from "blurry".
Run the Gaussian on a downsampled copy for the large sigma if it is slow.

## Colour

Derive colour from a **simulated physical quantity**, never from an arbitrary
palette index. Map the quantity to a physically-motivated ramp (blackbody-like
for temperature/heat flux, population identity for stellar components, speed or
stress for vehicles). Keep the ramp perceptually smooth; avoid rainbow.

## Hero acceptance bar

Before declaring done, open the PNG and look at it. Reject it if it
reads as a technical figure: visible axes/ticks/legend/colorbar, grid of
panels, watermark text, dashboard overlays, flat grey background, or particles
that look like uniform random dots rather than coherent structure. The hero
must have one dominant subject, high dynamic range, and read clearly at phone
size (test by downsampling to ~400 px wide and re-checking).
