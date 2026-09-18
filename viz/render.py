"""
A small software renderer for molecules and orbital isosurfaces.

Two kinds of geometry, composited by depth so they interleave correctly:

  * the molecule — spheres and cylinders, ray-traced analytically per pixel,
    which keeps them perfectly smooth at any zoom;
  * isosurfaces — extracted once with marching cubes, then drawn each frame as
    a depth-buffered point splat (points are sampled per triangle in proportion
    to its area, so coverage is uniform).

Rendering is orthographic.  World coordinates are Ångström; camera space is
x right, y up, z increasing away from the viewer.
"""
from __future__ import annotations

from dataclasses import dataclass, field as _dc_field

import numpy as np

# ── Palette ───────────────────────────────────────────────────────────────────

BG_INNER = np.array([0.038, 0.052, 0.092])
BG_OUTER = np.array([0.012, 0.018, 0.038])

ATOM_COLORS = {
    "H":  (0.92, 0.93, 0.96), "C":  (0.42, 0.46, 0.53), "N":  (0.33, 0.45, 0.95),
    "O":  (0.95, 0.31, 0.31), "F":  (0.56, 0.88, 0.31), "S":  (0.95, 0.85, 0.25),
    "Cl": (0.35, 0.90, 0.35), "P":  (0.98, 0.55, 0.15), "B":  (0.95, 0.66, 0.66),
    "Br": (0.72, 0.36, 0.28), "I":  (0.62, 0.32, 0.68), "Li": (0.70, 0.45, 0.95),
    "Na": (0.60, 0.40, 0.92), "Mg": (0.45, 0.85, 0.25), "Si": (0.75, 0.68, 0.55),
}
ATOM_COLOR_DEFAULT = (0.80, 0.60, 0.60)

ATOM_RADII = {"H": 0.26, "C": 0.40, "N": 0.38, "O": 0.37, "F": 0.35, "S": 0.50,
              "Cl": 0.50, "P": 0.52, "B": 0.44, "Br": 0.55, "I": 0.62}
ATOM_RADIUS_DEFAULT = 0.45
BOND_RADIUS = 0.105
BOND_COLOR = (0.46, 0.51, 0.60)

COVALENT = {"H": 0.31, "He": 0.28, "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.76,
            "N": 0.71, "O": 0.66, "F": 0.57, "Ne": 0.58, "Na": 1.66, "Mg": 1.41,
            "Al": 1.21, "Si": 1.11, "P": 1.07, "S": 1.05, "Cl": 1.02, "Ar": 1.06,
            "Br": 1.20, "I": 1.39}

PHASE_POS = (0.11, 0.42, 0.98)      # + lobe, blue
PHASE_NEG = (0.98, 0.16, 0.24)      # - lobe, red
DENSITY_COLOR = (0.06, 0.78, 0.68)  # teal

LIGHT_KEY  = np.array([-0.45,  0.62, -1.00])
LIGHT_FILL = np.array([ 0.70,  0.25, -0.65])


# ── Camera ────────────────────────────────────────────────────────────────────

@dataclass
class Camera:
    center: np.ndarray        # world point held at screen centre (Å)
    radius: float             # half-height of the view (Å)
    width: int
    height: int
    R: np.ndarray = _dc_field(default_factory=lambda: np.eye(3))

    @property
    def scale(self) -> float:
        """Pixels per Ångström."""
        return 0.5 * self.height / self.radius

    def to_camera(self, pts: np.ndarray) -> np.ndarray:
        return (pts - self.center) @ self.R.T

    def project(self, pts_cam: np.ndarray):
        """Camera-space points -> (px, py, depth)."""
        s = self.scale
        return (0.5 * self.width + pts_cam[..., 0] * s,
                0.5 * self.height - pts_cam[..., 1] * s,
                pts_cam[..., 2])

    def pixel_rays(self):
        """Camera-space (X, Y) of every pixel centre; rays run along +z."""
        s = self.scale
        xs = (np.arange(self.width) + 0.5 - 0.5 * self.width) / s
        ys = (0.5 * self.height - np.arange(self.height) - 0.5) / s
        return xs[None, :], ys[:, None]


def rotation(yaw: float, pitch: float = 0.0, roll: float = 0.0) -> np.ndarray:
    """World -> camera rotation from yaw (about world y), then pitch, then roll."""
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    Ry = np.array([[cy, 0, -sy], [0, 1, 0], [sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return Rz @ Rx @ Ry


# ── Layers ────────────────────────────────────────────────────────────────────

class Layer:
    """An RGB + depth + alpha buffer."""

    def __init__(self, h, w):
        self.rgb = np.zeros((h, w, 3), dtype=np.float32)
        self.depth = np.full((h, w), np.inf, dtype=np.float32)
        self.alpha = np.zeros((h, w), dtype=np.float32)

    @property
    def empty(self) -> bool:
        return not np.any(self.alpha > 0)


def shade(normals, base_rgb, *, specular=0.45, shininess=36.0,
          ambient=0.16, rim=0.0, rim_rgb=(1.0, 1.0, 1.0)):
    """Blinn-Phong with two lights, plus an optional rim term."""
    n = normals
    view = np.array([0.0, 0.0, -1.0], dtype=np.float32)

    n = np.asarray(n, dtype=np.float32)
    base = np.asarray(base_rgb, dtype=np.float32)
    out = np.broadcast_to(base * np.float32(ambient), n.shape).copy()

    for light, weight in ((LIGHT_KEY, 0.74), (LIGHT_FILL, 0.20)):
        L = (light / np.linalg.norm(light)).astype(np.float32)
        ndl = np.clip(n @ L, 0.0, None)
        out += base * (np.float32(weight) * ndl)[..., None]
        H = L + view
        H /= np.linalg.norm(H)
        ndh = np.clip(n @ H, 0.0, None)
        out += (specular * weight * ndh ** shininess)[..., None]

    if rim:
        fres = np.clip(1.0 - np.abs(n @ view), 0.0, 1.0) ** 2.5
        out += np.asarray(rim_rgb, dtype=np.float32) * (rim * fres)[..., None]

    return out


# ── Molecule: analytic spheres and cylinders ──────────────────────────────────

def _bonds(atoms) -> list[tuple[int, int]]:
    pos = np.array([[a["x"], a["y"], a["z"]] for a in atoms])
    out = []
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            ri = COVALENT.get(atoms[i]["symbol"], 0.8)
            rj = COVALENT.get(atoms[j]["symbol"], 0.8)
            if np.linalg.norm(pos[i] - pos[j]) < 1.25 * (ri + rj):
                out.append((i, j))
    return out


def _write(layer, sl, hit, depth, rgb, alpha):
    """Depth-test and write into a sub-rectangle of a layer."""
    sub_d = layer.depth[sl]
    better = hit & (depth < sub_d)
    if not np.any(better):
        return
    sub_d[better] = depth[better]
    layer.rgb[sl][better] = rgb[better]
    layer.alpha[sl][better] = alpha


def render_molecule(cam: Camera, atoms, *, alpha=1.0, atom_scale=1.0,
                    bond_radius=BOND_RADIUS, dim=1.0) -> Layer:
    """Ray-trace ball-and-stick geometry into a layer."""
    h, w = cam.height, cam.width
    layer = Layer(h, w)
    s = cam.scale

    pos = np.array([[a["x"], a["y"], a["z"]] for a in atoms])
    cpos = cam.to_camera(pos)
    Xs, Ys = cam.pixel_rays()

    def window(cx, cy, pad):
        """Pixel slice covering a camera-space disc, or None if off-screen."""
        px = 0.5 * w + cx * s
        py = 0.5 * h - cy * s
        rp = pad * s + 2
        x0, x1 = int(np.floor(px - rp)), int(np.ceil(px + rp)) + 1
        y0, y1 = int(np.floor(py - rp)), int(np.ceil(py + rp)) + 1
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, w), min(y1, h)
        if x1 <= x0 or y1 <= y0:
            return None
        return (slice(y0, y1), slice(x0, x1))

    # Spheres
    for i, a in enumerate(atoms):
        r = ATOM_RADII.get(a["symbol"], ATOM_RADIUS_DEFAULT) * atom_scale
        sl = window(cpos[i, 0], cpos[i, 1], r)
        if sl is None:
            continue
        X = Xs[:, sl[1]]
        Y = Ys[sl[0], :]
        dx = X - cpos[i, 0]
        dy = Y - cpos[i, 1]
        disc = r * r - dx * dx - dy * dy
        hit = disc > 0.0
        if not np.any(hit):
            continue
        sq = np.sqrt(np.where(hit, disc, 0.0))
        depth = cpos[i, 2] - sq
        nx = np.broadcast_to(dx, hit.shape)
        ny = np.broadcast_to(dy, hit.shape)
        normals = np.stack([nx, ny, -sq], axis=-1) / r
        base = np.asarray(ATOM_COLORS.get(a["symbol"], ATOM_COLOR_DEFAULT)) * dim
        rgb = shade(normals, base, specular=0.55, shininess=42.0, ambient=0.22)
        _write(layer, sl, hit, depth, rgb, alpha)

    # Cylinders
    for i, j in _bonds(atoms):
        A, B = cpos[i], cpos[j]
        L = float(np.linalg.norm(B - A))
        if L < 1e-6:
            continue
        ax = (B - A) / L
        cx, cy = 0.5 * (A[0] + B[0]), 0.5 * (A[1] + B[1])
        pad = 0.5 * L + bond_radius
        sl = window(cx, cy, pad)
        if sl is None:
            continue
        X = Xs[:, sl[1]]
        Y = Ys[sl[0], :]
        ux = np.broadcast_to(X - A[0], (sl[0].stop - sl[0].start, sl[1].stop - sl[1].start))
        uy = np.broadcast_to(Y - A[1], ux.shape)
        uz = np.full(ux.shape, -A[2])
        uda = ux * ax[0] + uy * ax[1] + uz * ax[2]

        p0 = np.stack([ux - uda * ax[0], uy - uda * ax[1], uz - uda * ax[2]], axis=-1)
        p1 = np.array([-ax[2] * ax[0], -ax[2] * ax[1], 1.0 - ax[2] * ax[2]])
        qa = float(p1 @ p1)
        if qa < 1e-9:                      # bond points straight at the camera
            continue
        qb = 2.0 * (p0 @ p1)
        qc = np.einsum("...k,...k->...", p0, p0) - bond_radius ** 2
        disc = qb * qb - 4.0 * qa * qc
        hit = disc > 0.0
        if not np.any(hit):
            continue
        t = (-qb - np.sqrt(np.where(hit, disc, 0.0))) / (2.0 * qa)
        along = uda + t * ax[2]
        hit &= (along >= 0.0) & (along <= L)
        if not np.any(hit):
            continue
        normals = p0 + t[..., None] * p1
        normals /= np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-12
        rgb = shade(normals, np.asarray(BOND_COLOR) * dim,
                    specular=0.40, shininess=30.0, ambient=0.20)
        _write(layer, sl, hit, t, rgb, alpha)

    return layer


# ── Isosurfaces: marching cubes -> area-weighted point cloud ──────────────────

class SurfacePoints:
    """Points sampled over an isosurface, with outward normals."""

    __slots__ = ("pos", "nrm", "n")

    def __init__(self, pos, nrm):
        self.pos = pos.astype(np.float32)
        self.nrm = nrm.astype(np.float32)
        self.n = len(pos)

    @staticmethod
    def concat(parts):
        """Merge point sets, returning (points, per-point colour array)."""
        live = [(p, c) for p, c in parts if p is not None and p.n]
        if not live:
            return None, None
        pos = np.concatenate([p.pos for p, _ in live])
        nrm = np.concatenate([p.nrm for p, _ in live])
        col = np.concatenate([np.broadcast_to(np.asarray(c, dtype=np.float32), (p.n, 3))
                              for p, c in live])
        return SurfacePoints(pos, nrm), col

    @classmethod
    def from_field(cls, field, grid, level, *, points_per_A2=2200.0,
                   max_points=4_000_000, seed=0):
        """
        Extract the `level` isosurface of `field` and scatter points over it.

        `points_per_A2` sets the sampling density; it should be a few times the
        square of the render scale (pixels per Å) so that splatting leaves no
        holes.  Returns None when the level is not crossed.
        """
        from skimage.measure import marching_cubes

        f = np.asarray(field, dtype=np.float32)
        if not (f.min() < level < f.max()):
            return None

        spacing = tuple(grid.spacing_ang)
        verts, faces, _, _ = marching_cubes(f, float(level), spacing=spacing,
                                            allow_degenerate=False)
        if len(faces) == 0:
            return None
        verts = verts + grid.origin_ang

        # Vertex normals from the field gradient: outward points away from the
        # region enclosed by the level, i.e. down-gradient for a positive level.
        gy = np.gradient(f, *spacing)
        idx = ((verts - grid.origin_ang) / grid.spacing_ang).T
        from scipy.ndimage import map_coordinates
        gv = np.stack([map_coordinates(g, idx, order=1, mode="nearest") for g in gy], axis=-1)
        gv *= -np.sign(level)
        gv /= np.linalg.norm(gv, axis=-1, keepdims=True) + 1e-12

        tri = verts[faces]
        e1 = tri[:, 1] - tri[:, 0]
        e2 = tri[:, 2] - tri[:, 0]
        area = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=-1)

        counts = np.maximum(1, np.ceil(area * points_per_A2).astype(np.int64))
        total = int(counts.sum())
        if total > max_points:                       # thin out uniformly
            counts = np.maximum(1, (counts * (max_points / total)).astype(np.int64))

        rng = np.random.default_rng(seed)
        fi = np.repeat(np.arange(len(faces)), counts)
        r1 = rng.random(len(fi), dtype=np.float32)
        r2 = rng.random(len(fi), dtype=np.float32)
        flip = (r1 + r2) > 1.0
        r1[flip] = 1.0 - r1[flip]
        r2[flip] = 1.0 - r2[flip]

        pos = tri[fi, 0] + r1[:, None] * e1[fi] + r2[:, None] * e2[fi]
        nv = faces[fi]
        nrm = ((1.0 - r1 - r2)[:, None] * gv[nv[:, 0]]
               + r1[:, None] * gv[nv[:, 1]]
               + r2[:, None] * gv[nv[:, 2]])
        nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True) + 1e-12
        return cls(pos, nrm)


def splat(cam: Camera, pts: SurfacePoints, base_rgb, *, alpha=0.86,
          rim=0.30, rim_rgb=None, vertex_rgb=None, splat_size=2,
          slant_bias=0.75, specular=0.34, shininess=28.0, ambient=0.17) -> Layer:
    """
    Depth-buffered point splat of a surface into its own layer.

    The depth test runs first and shading second, so lighting is evaluated once
    per visible pixel rather than once per point — the point cloud is several
    times larger than the frame, so this dominates the per-frame cost.
    """
    h, w = cam.height, cam.width
    layer = Layer(h, w)
    if pts is None or pts.n == 0:
        return layer

    P = cam.to_camera(pts.pos)
    N = pts.nrm @ cam.R.T

    keep = N[:, 2] < 0.0                        # front faces only
    if not np.any(keep):
        return layer
    P, N = P[keep], N[keep]
    cols = None if vertex_rgb is None else np.asarray(vertex_rgb, dtype=np.float32)[keep]

    px, py, z = cam.project(P)
    k = max(1, int(splat_size))
    ix = np.floor(px - 0.5 * (k - 1)).astype(np.int32)
    iy = np.floor(py - 0.5 * (k - 1)).astype(np.int32)
    inside = (ix >= -k) & (ix < w) & (iy >= -k) & (iy < h)
    if not np.any(inside):
        return layer
    ix, iy, z = ix[inside], iy[inside], z[inside]
    N = N[inside]
    if cols is not None:
        cols = cols[inside]

    if slant_bias:
        nz = np.abs(N[:, 2])
        slant = np.sqrt(np.clip(1.0 - nz * nz, 0.0, 1.0)) / np.maximum(nz, 0.10)
        z = z - np.float32(slant_bias / cam.scale) * np.minimum(slant, 10.0)

    # Depth test: walk far -> near so the nearest point wins each pixel.
    order = np.argsort(-z)
    ox, oy = ix[order], iy[order]
    won = np.full((h, w), -1, dtype=np.int64)
    for dy in range(k):
        for dx in range(k):
            gx, gy = ox + dx, oy + dy
            ok = (gx >= 0) & (gx < w) & (gy >= 0) & (gy < h)
            won[gy[ok], gx[ok]] = order[ok]
    mask = won >= 0
    if not np.any(mask):
        return layer
    sel = won[mask]

    base = base_rgb if cols is None else cols[sel]
    if rim_rgb is None:                          # tint of the surface's own colour
        rim_rgb = np.clip(0.42 + 0.58 * np.asarray(base, dtype=np.float32), 0.0, 1.0)
    rgb = shade(N[sel], base, specular=specular, shininess=shininess,
                ambient=ambient, rim=rim, rim_rgb=rim_rgb)

    layer.rgb[mask] = rgb
    layer.depth[mask] = z[sel]
    layer.alpha[mask] = alpha
    return layer


def fill_holes(layer: Layer, passes: int = 1):
    """Close isolated 1-pixel gaps left by splatting, using 4-neighbours."""
    for _ in range(passes):
        a = layer.alpha
        hole = a <= 0.0
        if not np.any(hole):
            return
        acc_rgb = np.zeros_like(layer.rgb)
        acc_d = np.zeros_like(layer.depth)
        cnt = np.zeros_like(layer.depth)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            r = np.roll(np.roll(layer.rgb, dy, 0), dx, 1)
            d = np.roll(np.roll(layer.depth, dy, 0), dx, 1)
            m = np.roll(np.roll(a > 0, dy, 0), dx, 1)
            acc_rgb += r * m[..., None]
            acc_d += np.where(m, d, 0.0)
            cnt += m
        fill = hole & (cnt >= 3)
        if not np.any(fill):
            return
        layer.rgb[fill] = acc_rgb[fill] / cnt[fill][:, None]
        layer.depth[fill] = acc_d[fill] / cnt[fill]
        layer.alpha[fill] = a.max()


def despeckle(rgb: np.ndarray, *, min_disagree=7, threshold=0.10):
    """
    Remove isolated pixels that disagree with their whole neighbourhood.

    A splatted point reports the depth of one spot on the surface, not of its
    whole pixel, so wherever two things pass close together in depth — the two
    phases of an orbital either side of a node, or a lobe grazing the atom it
    encloses — scattered pixels are won by the wrong one.  Single stray pixels
    of the opposite phase, or of the grey ball-and-stick behind, shimmer badly
    in motion.

    A pixel is replaced by the mean of its neighbours only when at least
    `min_disagree` of the eight differ from it by more than `threshold` in some
    channel.  At 7 of 8 even a one-pixel-wide line survives — its two
    neighbours along the line agree with it — while an isolated speck, which
    has nothing agreeing with it anywhere, does not.  Edges keep a whole side
    of agreeing neighbours and are never touched.
    """
    out = rgb
    h, w = out.shape[:2]
    pad = np.pad(out, ((1, 1), (1, 1), (0, 0)), mode="edge")
    offsets = [(dy, dx) for dy in (0, 1, 2) for dx in (0, 1, 2) if (dy, dx) != (1, 1)]

    # Pass 1: how many neighbours differ, per pixel.  Channel-at-a-time keeps
    # this to 2-D temporaries over a full-resolution frame.
    count = np.zeros((h, w), dtype=np.uint8)
    far = np.empty((h, w), dtype=bool)
    tmp = np.empty((h, w), dtype=bool)
    for dy, dx in offsets:
        n = pad[dy:dy + h, dx:dx + w]
        np.greater(np.abs(n[..., 0] - out[..., 0]), threshold, out=far)
        for c in (1, 2):
            np.greater(np.abs(n[..., c] - out[..., c]), threshold, out=tmp)
            np.logical_or(far, tmp, out=far)
        count += far

    ys, xs = np.nonzero(count >= min_disagree)
    if len(ys) == 0:
        return out

    # Pass 2: average the disagreeing neighbours, only where it matters.
    here = out[ys, xs]
    acc = np.zeros_like(here)
    hits = np.zeros(len(ys), dtype=np.float32)
    for dy, dx in offsets:
        n = pad[ys + dy, xs + dx]
        f = np.max(np.abs(n - here), axis=-1) > threshold
        acc += n * f[:, None]
        hits += f
    out[ys, xs] = acc / hits[:, None]
    return out


# ── Compositing ───────────────────────────────────────────────────────────────

def background(cam: Camera) -> np.ndarray:
    """Radial gradient with a light vignette."""
    h, w = cam.height, cam.width
    yy = (np.arange(h) - 0.5 * h) / (0.5 * h)
    xx = (np.arange(w) - 0.5 * w) / (0.5 * w)
    r = np.sqrt(xx[None, :] ** 2 + yy[:, None] ** 2) / np.sqrt(2.0)
    t = np.clip(r, 0.0, 1.0)[..., None] ** 1.25
    return (BG_INNER * (1.0 - t) + BG_OUTER * t).astype(np.float32)


def compose(cam: Camera, layers, bg=None, fog=0.30) -> np.ndarray:
    """Alpha-blend layers back to front, per pixel, by depth."""
    out = background(cam) if bg is None else bg.copy()
    layers = [l for l in layers if l is not None and not l.empty]
    if not layers:
        return out

    D = np.stack([l.depth for l in layers])
    A = np.stack([l.alpha for l in layers])
    C = np.stack([l.rgb for l in layers])

    if fog:
        # Darken with distance so overlapping lobes read as front and back.
        finite = np.isfinite(D)
        if np.any(finite):
            lo, hi = np.percentile(D[finite], [2, 98])
            if hi > lo:
                t = np.clip((D - lo) / (hi - lo), 0.0, 1.0)
                C = C * (1.0 - fog * np.where(finite, t, 0.0))[..., None]

    order = np.argsort(-D, axis=0)
    A = np.take_along_axis(A, order, axis=0)
    C = np.take_along_axis(C, order[..., None], axis=0)

    for k in range(len(layers)):
        a = A[k][..., None]
        out = out * (1.0 - a) + C[k] * a
    return out


def to_image(rgb: np.ndarray, out_size: int | tuple | None = None):
    """Tone-map to 8-bit and downsample the supersampled buffer."""
    from PIL import Image

    x = np.clip(rgb, 0.0, 1.0)
    x = np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)
    img = Image.fromarray((x * 255.0 + 0.5).astype(np.uint8), "RGB")
    if out_size is not None:
        size = (out_size, out_size) if isinstance(out_size, int) else out_size
        img = img.resize(size, Image.LANCZOS)
    return img
