"""Custom perspective renderer for the hero visuals.

Nothing here uses matplotlib.  The pipeline per frame is

 1. a real pinhole perspective camera in numpy projects world points to pixels;
 2. flight trails are additively splatted into a float32 HDR buffer with
    bilinear deposition (``np.bincount`` accumulation, per the playbook), with
    energy falling off with trail age and with atmospheric extinction;
 3. no-fly cylinders are drawn as splatted glowing rims;
 4. each aircraft is drawn as an oriented delta-wing glyph -- a real 3-D mesh
    rotated by the vehicle's actual (psi, gamma, phi) -- rasterised with PIL's
    polygon fill, per-facet shaded, and composited into the HDR buffer;
 5. multi-scale bloom + an ACES-like filmic tone map turn the accumulated
    energy into an 8-bit frame.

Colour is derived from a simulated quantity: the vehicle's bank angle magnitude
blended with its collision-avoidance deflection, i.e. how hard it is manoeuvring.
Cruising vehicles are cool cyan-white; hard-manoeuvring ones glow amber.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter, zoom


# --------------------------------------------------------------------------
# Camera
# --------------------------------------------------------------------------
class Camera:
    """Right-handed pinhole camera with a look-at basis."""

    def __init__(self, eye, target, up=(0.0, 0.0, 1.0), fov_deg=38.0,
                 width=1920, height=1080, roll_deg=0.0):
        self.eye = np.asarray(eye, float)
        self.target = np.asarray(target, float)
        up = np.asarray(up, float)
        f = self.target - self.eye
        f = f / np.linalg.norm(f)
        r = np.cross(f, up)
        nr = np.linalg.norm(r)
        if nr < 1e-9:
            r = np.array([1.0, 0.0, 0.0])
            nr = 1.0
        r = r / nr
        u = np.cross(r, f)
        if roll_deg:
            # Roll about the optical axis (a canted / Dutch-angle camera).
            c, sn = np.cos(np.deg2rad(roll_deg)), np.sin(np.deg2rad(roll_deg))
            r, u = c * r + sn * u, -sn * r + c * u
        self.roll_deg = float(roll_deg)
        self.R = np.stack([r, u, f])          # world -> camera rows
        self.W, self.H = int(width), int(height)
        self.fov = np.deg2rad(fov_deg)
        self.focal = 0.5 * self.H / np.tan(0.5 * self.fov)

    def to_camera(self, pts):
        p = np.atleast_2d(np.asarray(pts, float)) - self.eye[None, :]
        return p @ self.R.T                    # (N, 3): x right, y up, z forward

    def project(self, pts):
        """World points -> (pixel_xy (N,2), depth (N,), visible mask (N,))."""
        c = self.to_camera(pts)
        z = c[:, 2]
        vis = z > 1.0
        zz = np.where(vis, z, 1.0)
        x = self.W * 0.5 + self.focal * c[:, 0] / zz
        y = self.H * 0.5 - self.focal * c[:, 1] / zz
        return np.stack([x, y], axis=1), z, vis


# --------------------------------------------------------------------------
# Additive splatting (playbook)
# --------------------------------------------------------------------------
def splat(buf, x, y, rgb, weight):
    """Bilinearly deposit weighted RGB energy into ``buf`` (H, W, 3) float32."""
    H, W = buf.shape[0], buf.shape[1]
    m = (x >= 0) & (x < W - 1) & (y >= 0) & (y < H - 1) & np.isfinite(x) & np.isfinite(y)
    if not m.any():
        return buf
    x, y, rgb, weight = x[m], y[m], rgb[m], weight[m]
    x0 = x.astype(np.int32)
    y0 = y.astype(np.int32)
    fx = x - x0
    fy = y - y0
    flat = buf.reshape(-1, 3)
    for dx, dy, wgt in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                        (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        idx = (y0 + dy) * W + (x0 + dx)
        ww = (wgt * weight).astype(np.float32)
        for c in range(3):
            flat[:, c] += np.bincount(idx, weights=ww * rgb[:, c],
                                      minlength=W * H).astype(np.float32)
    return buf


# --------------------------------------------------------------------------
# Tone mapping (playbook)
# --------------------------------------------------------------------------
def tonemap(buf, exposure=1.0, bloom_sigmas=(2.0, 8.0, 28.0),
            bloom_gains=(0.55, 0.30, 0.18), gamma=2.2, saturation=1.12,
            fast_large=True):
    hdr = buf * exposure
    glow = np.zeros_like(hdr)
    for s, g in zip(bloom_sigmas, bloom_gains):
        if fast_large and s > 10.0:
            small = hdr[::4, ::4]
            gl = gaussian_filter(small, sigma=(s / 4.0, s / 4.0, 0))
            gl = zoom(gl, (hdr.shape[0] / gl.shape[0], hdr.shape[1] / gl.shape[1], 1),
                      order=1)
            gl = gl[:hdr.shape[0], :hdr.shape[1]]
            if gl.shape[:2] != hdr.shape[:2]:
                pad = np.zeros_like(hdr)
                pad[:gl.shape[0], :gl.shape[1]] = gl
                gl = pad
            glow += g * gl
        else:
            glow += g * gaussian_filter(hdr, sigma=(s, s, 0))
    hdr = hdr + glow
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    ldr = np.clip((hdr * (a * hdr + b)) / (hdr * (c * hdr + d) + e), 0.0, 1.0)
    lum = ldr @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    ldr = np.clip(lum[..., None] + (ldr - lum[..., None]) * saturation, 0, 1)
    return (np.power(ldr, 1.0 / gamma) * 255.0 + 0.5).astype(np.uint8)


# --------------------------------------------------------------------------
# Colour ramp driven by a simulated quantity
# --------------------------------------------------------------------------
# Cool ramp: cruise altitude, low (deep indigo) to high (pale cyan-white).
_COOL = np.array([
    [0.09, 0.15, 0.68],
    [0.10, 0.34, 0.90],
    [0.18, 0.60, 0.98],
    [0.36, 0.80, 0.98],
    [0.58, 0.94, 0.94],
], dtype=np.float32)
_COOL_X = np.array([0.0, 0.28, 0.55, 0.80, 1.0], dtype=np.float32)

# Hot ramp: manoeuvre activity, from warm white through amber to red.
_HOT = np.array([
    [1.00, 0.92, 0.72],
    [1.00, 0.74, 0.32],
    [1.00, 0.48, 0.15],
    [1.00, 0.26, 0.16],
], dtype=np.float32)
_HOT_X = np.array([0.0, 0.40, 0.72, 1.0], dtype=np.float32)


def _ramp(q, xs, cs):
    q = np.clip(np.asarray(q, np.float32), 0.0, 1.0)
    out = np.empty(q.shape + (3,), np.float32)
    for c in range(3):
        out[..., c] = np.interp(q, xs, cs[:, c])
    return out


def altitude_colour(q):
    """Cool ramp driven by normalised cruise altitude."""
    return _ramp(q, _COOL_X, _COOL)


def activity_colour(q, altitude=None):
    """Colour from two simulated quantities.

    ``altitude`` (normalised into the swarm's operating band) sets the base hue
    on a cool indigo-to-pale-cyan ramp, so the vertical layering of the traffic
    is legible in a 2-D image.  ``q`` is normalised manoeuvre activity -- bank
    angle plus collision-avoidance deflection -- and blends that base towards a
    warm amber/red, so the aircraft that are actually working for their
    separation are the ones that glow hot.
    """
    q = np.clip(np.asarray(q, np.float32), 0.0, 1.0)
    base = (altitude_colour(altitude) if altitude is not None
            else _ramp(np.full_like(q, 0.6), _COOL_X, _COOL))
    hot = _ramp(q, _HOT_X, _HOT)
    w = (q ** 0.75)[..., None]
    return np.clip(base * (1.0 - w) + hot * w, 0.0, 1.0)


# --------------------------------------------------------------------------
# Aircraft glyph
# --------------------------------------------------------------------------
def _glyph_mesh(scale=1.0):
    """A swept-wing aircraft silhouette in body axes.

    Body axes are right-handed with x forward, y to the **left** and z up
    (forward x left = up), which is the handedness that pairs with the ENU world
    frame without an extra reflection.

    The mesh is deliberately made of facets whose normals point in genuinely
    different directions -- near-horizontal wings and tailplane, a vertical fin,
    angled fuselage flanks -- because flat-shading those against a fixed key
    light is what makes the vehicle read as a solid object, and what makes bank
    angle legible: as the aircraft rolls, the wing facets swing through the
    light while the fin does the opposite.

    Returns (vertices (V,3), facets [(indices, albedo, kind)]) where ``kind`` is
    0 for a lit surface and 1 for an emissive one.
    """
    L = scale
    v = np.array([
        [1.15 * L, 0.00, 0.000],      # 0  nose
        [0.10 * L, 0.00, 0.115],      # 1  fuselage spine (top)
        [0.10 * L, 0.00, -0.085],     # 2  fuselage keel (bottom)
        [-0.92 * L, 0.00, 0.020],     # 3  tail cone
        [0.20 * L, 0.00, 0.000],      # 4  wing root, forward
        [-0.34 * L, 0.00, 0.000],     # 5  wing root, aft
        [-0.20 * L, 0.95, 0.055],     # 6  left wingtip (swept, dihedral)
        [-0.20 * L, -0.95, 0.055],    # 7  right wingtip
        [-0.50 * L, 0.58, 0.030],     # 8  left trailing edge
        [-0.50 * L, -0.58, 0.030],    # 9  right trailing edge
        [-0.74 * L, 0.36, 0.045],     # 10 left tailplane tip
        [-0.74 * L, -0.36, 0.045],    # 11 right tailplane tip
        [-0.95 * L, 0.00, 0.045],     # 12 tailplane apex
        [-0.86 * L, 0.00, 0.400],     # 13 fin top
        [-0.48 * L, 0.00, 0.040],     # 14 fin base
        [0.48 * L, 0.00, 0.085],      # 15 canopy forward
        [0.02 * L, 0.00, 0.105],      # 16 canopy aft
        [-0.88 * L, 0.00, 0.020],     # 17 exhaust point
    ], dtype=np.float32)
    facets = [
        ((0, 6, 5), 1.00, 0),      # left wing, forward panel
        ((6, 8, 5), 0.88, 0),      # left wing, aft panel
        ((0, 5, 7), 0.94, 0),      # right wing, forward panel
        ((7, 5, 9), 0.82, 0),      # right wing, aft panel
        ((0, 1, 3), 0.72, 0),      # fuselage upper spine
        ((0, 2, 3), 0.46, 0),      # fuselage keel
        ((3, 10, 12), 0.66, 0),    # left tailplane
        ((3, 12, 11), 0.60, 0),    # right tailplane
        ((14, 13, 3), 0.78, 0),    # fin
        ((15, 16, 1), 1.00, 1),    # canopy (emissive)
    ]
    return v, facets


EXHAUST_INDEX = 17      # vertex carrying the hot engine point


_MESH_V, _MESH_F = _glyph_mesh()


def attitude_matrix(psi, gamma, phi):
    """Body -> world rotation from heading, flight-path angle and bank.

    ENU world; body axes right-handed with x forward, y left, z up.  The
    composition is the usual intrinsic yaw-pitch-roll ``Rz(psi) Ry' Rx(phi)``
    with ``Ry'`` pitching the nose *up* for positive ``gamma``, so that the body
    x-axis reproduces :func:`swarmsim.dynamics.air_velocity` exactly, and
    positive ``phi`` (a right bank) lifts the left wing.
    """
    cp, sp = np.cos(psi), np.sin(psi)
    cg, sg = np.cos(gamma), np.sin(gamma)
    cr, sr = np.cos(phi), np.sin(phi)
    n = np.size(psi)
    Rz = np.zeros((n, 3, 3))
    Rz[:, 0, 0] = cp; Rz[:, 0, 1] = -sp; Rz[:, 1, 0] = sp; Rz[:, 1, 1] = cp
    Rz[:, 2, 2] = 1.0
    Ry = np.zeros((n, 3, 3))
    Ry[:, 0, 0] = cg; Ry[:, 0, 2] = -sg; Ry[:, 1, 1] = 1.0
    Ry[:, 2, 0] = sg; Ry[:, 2, 2] = cg
    Rx = np.zeros((n, 3, 3))
    Rx[:, 0, 0] = 1.0
    Rx[:, 1, 1] = cr; Rx[:, 1, 2] = -sr
    Rx[:, 2, 1] = sr; Rx[:, 2, 2] = cr
    return Rz @ Ry @ Rx


def circle_points(cx, cy, z, r, n=180):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([cx + r * np.cos(a), cy + r * np.sin(a), np.full(n, z)], axis=1)
