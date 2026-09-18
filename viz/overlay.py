"""
Typographic and chart overlays drawn on top of rendered frames.

Everything is PIL drawing on the final 8-bit image, at output resolution, so
text stays crisp and is not resampled with the 3-D buffer.
"""
from __future__ import annotations

import glob
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── Fonts ─────────────────────────────────────────────────────────────────────

_FONT_DIRS = ["/usr/share/fonts/truetype/liberation", "/usr/share/fonts/truetype/dejavu",
              "/usr/local/lib/python3.11/dist-packages/matplotlib/mpl-data/fonts/ttf"]
_FACES = {
    "bold":    ["LiberationSans-Bold.ttf", "DejaVuSans-Bold.ttf"],
    "regular": ["LiberationSans-Regular.ttf", "DejaVuSans.ttf"],
    "mono":    ["LiberationMono-Regular.ttf", "DejaVuSansMono.ttf"],
    "monob":   ["LiberationMono-Bold.ttf", "DejaVuSansMono-Bold.ttf"],
}
_cache: dict = {}
_cmaps: dict = {}


def _covers(path: str, s: str) -> bool:
    """Does the face at `path` have a glyph for every character of `s`?"""
    if path not in _cmaps:
        try:
            from fontTools.ttLib import TTFont
            tt = TTFont(path, fontNumber=0, lazy=True)
            cm: set = set()
            for t in tt["cmap"].tables:
                cm |= set(t.cmap.keys())
            _cmaps[path] = cm
        except Exception:
            _cmaps[path] = None
    cm = _cmaps[path]
    return True if cm is None else all(ord(c) in cm for c in s)


def _paths(face: str):
    for name in _FACES[face]:
        for d in _FONT_DIRS:
            path = os.path.join(d, name)
            if os.path.exists(path):
                yield path


def font(face: str, size: int, s: str | None = None) -> ImageFont.FreeTypeFont:
    """
    Load a face, falling back to the next one in the list when the preferred
    face has no glyph for something in `s`.  Liberation Sans sets the tone but
    is missing a few subscripts and superscripts that DejaVu carries, and a
    missing glyph shows up as a tofu box in the middle of a caption.
    """
    chosen = None
    for path in _paths(face):
        if chosen is None:
            chosen = path
        if s is None or _covers(path, s):
            chosen = path
            break
    if chosen is None:
        hits = glob.glob("/usr/share/fonts/**/*.ttf", recursive=True)
        if not hits:
            raise RuntimeError("no usable TrueType font found")
        chosen = hits[0]
    key = (chosen, size)
    if key not in _cache:
        _cache[key] = ImageFont.truetype(chosen, size)
    return _cache[key]


# ── Colours ───────────────────────────────────────────────────────────────────

# Neutral greys: the backdrop is neutral, and a blue-tinted caption over it
# reads as a tint rather than as type.
INK        = (240, 241, 243)
INK_DIM    = (163, 166, 171)
INK_FAINT  = (120, 123, 128)
ACCENT     = (56, 189, 248)
POS_CHIP   = (56, 132, 250)
NEG_CHIP   = (250, 62, 78)
TEAL_CHIP  = (22, 205, 180)
PANEL_BG   = (12, 18, 34)


MINUS = "\u2212"          # typographic minus; the ASCII hyphen reads as a dash

_SUPS = str.maketrans("0123456789-", "\u2070\u00b9\u00b2\u00b3\u2074"
                                     "\u2075\u2076\u2077\u2078\u2079\u207b")


def _sup(n: int) -> str:
    """Superscript digits (the font loader falls back for any it lacks)."""
    return str(int(n)).translate(_SUPS)


def fix_minus(s: str) -> str:
    """Render leading/embedded ASCII minus signs as U+2212."""
    return s.replace("-", MINUS)


def _rgba(color, alpha: float):
    return tuple(int(c) for c in color) + (int(round(255 * max(0.0, min(1.0, alpha)))),)


def scrim(size, *, top=0.0, bottom=0.30, strength=0.58, color=(5, 9, 18)):
    """
    Vertical darkening at the top and bottom edges.

    The caption sits over whatever the render happens to put behind it, and a
    pale isosurface can swallow small type entirely.  A gradient along the
    bottom edge costs almost nothing visually on a dark scene and makes the
    text unconditionally legible.
    """
    w, h = size
    y = np.arange(h)
    a = np.zeros(h)
    if bottom:
        a = np.maximum(a, np.clip((y - (1.0 - bottom) * h) / (bottom * h), 0, 1) ** 1.7
                       * strength)
    if top:
        a = np.maximum(a, np.clip(((top * h) - y) / (top * h), 0, 1) ** 1.7
                       * strength * 0.85)
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[..., 0], arr[..., 1], arr[..., 2] = color
    arr[..., 3] = np.broadcast_to((a * 255).astype(np.uint8)[:, None], (h, w))
    return Image.fromarray(arr, "RGBA")


def draw_on(img: Image.Image, *, shade_edges=True):
    """Return (overlay, draw) — an RGBA scratch layer the size of `img`."""
    layer = (scrim(img.size) if shade_edges
             else Image.new("RGBA", img.size, (0, 0, 0, 0)))
    return layer, ImageDraw.Draw(layer)


def flatten(img: Image.Image, layer: Image.Image) -> Image.Image:
    return Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")


# ── Building blocks ───────────────────────────────────────────────────────────

def text(draw, xy, s, *, face="regular", size=28, color=INK, alpha=1.0,
         anchor="la", spacing=4, tracking=0.0):
    f = font(face, size, s)
    if tracking:
        x, y = xy
        for ch in s:
            draw.text((x, y), ch, font=font(face, size, ch),
                      fill=_rgba(color, alpha), anchor=anchor)
            x += draw.textlength(ch, font=f) + tracking
        return
    draw.text(xy, s, font=f, fill=_rgba(color, alpha), anchor=anchor, spacing=spacing)


def text_width(s, face="regular", size=28) -> float:
    return ImageDraw.Draw(Image.new("RGB", (1, 1))).textlength(s, font=font(face, size, s))


def panel(draw, box, *, alpha=0.55, radius=18, border=None, border_alpha=0.5):
    draw.rounded_rectangle(box, radius=radius, fill=_rgba(PANEL_BG, alpha),
                           outline=None if border is None else _rgba(border, border_alpha),
                           width=2)


def ease(t: float) -> float:
    """Smootherstep on [0, 1]."""
    t = min(1.0, max(0.0, t))
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def fade_window(i, n, hold_in, hold_out) -> float:
    """1 inside the clip, ramping in over `hold_in` frames and out over `hold_out`."""
    a = ease(i / hold_in) if hold_in else 1.0
    b = ease((n - 1 - i) / hold_out) if hold_out else 1.0
    return min(a, b)


# ── Caption ──────────────────────────────────────────────────────────────────
#
# One block, bottom left: a headline and at most one line under it.  Everything
# else a frame could say — the point group, the basis-function count, the phase
# legend, a tool credit — competes with the thing the clip is actually about,
# and none of it is read in eight seconds.  Anything that belongs with the work
# rather than in it goes in the post copy instead.

def caption(draw, W, H, title, subtitle=None, *, alpha=1.0):
    if alpha <= 0.01:
        return
    x = int(0.058 * W)
    y = int(0.845 * H)
    text(draw, (x, y), title, face="bold", size=int(0.062 * W),
         color=INK, alpha=alpha, anchor="ls")
    if subtitle:
        text(draw, (x, y + int(0.040 * H)), subtitle, size=int(0.0215 * W),
             color=INK_DIM, alpha=alpha, anchor="ls")


def caption_switch(draw, W, H, before, after, t, *, alpha=1.0):
    """
    One caption slot handing over from `before` to `after` across t in [0, 1].

    The outgoing caption is gone before the incoming one appears; drawing both
    at partial alpha in the same place just piles them on top of each other.
    """
    spec, a = (before, 1.0 - 2.0 * t) if t < 0.5 else (after, 2.0 * t - 1.0)
    if spec is None or a <= 0.01:
        return
    caption(draw, W, H, alpha=ease(a) * alpha, **spec)


# ── Convergence chart ─────────────────────────────────────────────────────────

def convergence_panel(draw, box, xs, ys, *, upto, alpha=1.0, log=True,
                      ylabel="", xlabel="cycle", value_text=None, color=ACCENT):
    """
    Small line chart inside `box` = (x0, y0, x1, y1).

    `xs`/`ys` are the full series; only the first `upto` points are drawn, so
    passing an increasing `upto` animates the trace.  All internal metrics are
    proportional to the box, so the panel holds together at any output size.
    """
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    panel(draw, box, alpha=0.66 * alpha, radius=int(0.07 * bh),
          border=(44, 62, 96), border_alpha=0.6 * alpha)

    f_tick = max(9, int(0.085 * bh))
    f_lab = max(10, int(0.105 * bh))
    ml, mr = 0.235 * bw, 0.055 * bw
    mt, mb = 0.33 * bh, 0.20 * bh
    px0, px1 = x0 + ml, x1 - mr
    py0, py1 = y0 + mt, y1 - mb

    v = np.asarray(ys, dtype=float)
    plot = np.log10(np.maximum(v, 1e-14)) if log else v
    lo, hi = float(plot.min()), float(plot.max())
    if hi - lo < 1e-9:
        lo, hi = lo - 0.5, hi + 0.5
    pad = 0.09 * (hi - lo)
    lo, hi = lo - pad, hi + pad

    xa = np.asarray(xs, dtype=float)
    xlo, xhi = float(xa.min()), float(xa.max())
    if xhi - xlo < 1e-9:
        xhi = xlo + 1.0

    def X(t):
        return px0 + (t - xlo) / (xhi - xlo) * (px1 - px0)

    def Y(t):
        return py1 - (t - lo) / (hi - lo) * (py1 - py0)

    # Gridlines on decades, thinned so labels never collide.
    decades = [g for g in range(int(np.floor(lo)), int(np.ceil(hi)) + 1) if lo <= g <= hi]
    step = max(1, int(np.ceil(len(decades) / max(1, (py1 - py0) / (2.2 * f_tick)))))
    for g in decades:
        gy = Y(g)
        draw.line([px0, gy, px1, gy], fill=_rgba((46, 60, 92), 0.55 * alpha), width=1)
        if (g - decades[0]) % step == 0:
            text(draw, (px0 - 0.035 * bw, gy), f"10{_sup(g)}", face="mono",
                 size=f_tick, color=INK_FAINT, alpha=alpha, anchor="rm")

    draw.line([px0, py0, px0, py1], fill=_rgba((70, 88, 126), 0.85 * alpha), width=2)
    draw.line([px0, py1, px1, py1], fill=_rgba((70, 88, 126), 0.85 * alpha), width=2)

    k = max(1, min(int(upto), len(v)))
    pts = [(X(xa[i]), Y(plot[i])) for i in range(k)]
    lw = max(2, int(0.022 * bh))
    if len(pts) > 1:
        draw.line([c for p in pts for c in p], fill=_rgba(color, alpha),
                  width=lw, joint="curve")
    r = max(2, int(0.02 * bh))
    for p in pts[:-1]:
        draw.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r],
                     fill=_rgba(color, 0.85 * alpha))
    if pts:
        hx, hy = pts[-1]
        rh = max(4, int(0.05 * bh))
        draw.ellipse([hx - rh, hy - rh, hx + rh, hy + rh], fill=_rgba(color, 0.22 * alpha))
        rh = max(3, int(0.028 * bh))
        draw.ellipse([hx - rh, hy - rh, hx + rh, hy + rh],
                     fill=_rgba((255, 255, 255), alpha))

    if ylabel:
        text(draw, (x0 + 0.055 * bw, y0 + 0.085 * bh), ylabel, face="bold",
             size=f_lab, color=INK_DIM, alpha=alpha)
    text(draw, ((px0 + px1) / 2, y1 - 0.045 * bh), xlabel, size=f_tick,
         color=INK_FAINT, alpha=alpha, anchor="ms")
    if value_text:
        text(draw, (x1 - 0.055 * bw, y0 + 0.075 * bh), value_text, face="monob",
             size=max(11, int(0.135 * bh)), color=INK, alpha=alpha, anchor="ra")


def colorbar(draw, box, ramp, *, labels=("", ""), title="", alpha=1.0, ticks=None):
    """
    Horizontal colour bar.  `ramp` is a callable t in [0,1] -> (r, g, b) floats.
    """
    x0, y0, x1, y1 = box
    n = max(2, int(x1 - x0))
    for i in range(n):
        t = i / (n - 1)
        c = tuple(int(round(255 * v)) for v in ramp(t))
        draw.rectangle([x0 + i, y0, x0 + i + 1, y1], fill=_rgba(c, alpha))
    draw.rounded_rectangle([x0, y0, x1, y1], radius=4,
                           outline=_rgba((90, 108, 148), 0.7 * alpha), width=2)
    if title:
        text(draw, ((x0 + x1) / 2, y0 - 12), title, face="bold", size=19,
             color=INK_DIM, alpha=alpha, anchor="ms")
    text(draw, (x0, y1 + 9), labels[0], face="mono", size=18, color=INK_DIM, alpha=alpha)
    text(draw, (x1, y1 + 9), labels[1], face="mono", size=18, color=INK_DIM,
         alpha=alpha, anchor="ra")
    for t, lab in (ticks or []):
        tx = x0 + t * (x1 - x0)
        draw.line([tx, y0, tx, y1], fill=_rgba((255, 255, 255), 0.5 * alpha), width=1)
        text(draw, (tx, y1 + 9), lab, face="mono", size=18, color=INK_DIM,
             alpha=alpha, anchor="ma")
