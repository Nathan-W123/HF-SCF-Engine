"""Rendering pipeline: it must be energy-preserving, black where empty, and
produce exactly 1920x1080 uint8 frames."""
import numpy as np
import pytest

from galcol.render import (Camera, FrameRenderer, luminosity_scatter,
                           orbit_eye, population_colours, splat, tonemap,
                           _upsample)
from galcol.units import PTYPE_BULGE, PTYPE_DISK


def test_splat_conserves_total_deposited_energy():
    buf = np.zeros((64, 64, 3), np.float32)
    rng = np.random.default_rng(0)
    n = 500
    x = rng.uniform(2, 60, n).astype(np.float32)
    y = rng.uniform(2, 60, n).astype(np.float32)
    rgb = np.ones((n, 3), np.float32)
    w = rng.uniform(0.5, 2.0, n).astype(np.float32)
    splat(buf, x, y, rgb, w)
    assert buf.sum() == pytest.approx(3.0 * w.sum(), rel=1e-4)


def test_splat_is_bilinear_and_positional():
    buf = np.zeros((8, 8, 3), np.float32)
    splat(buf, np.array([3.25]), np.array([2.75]),
          np.ones((1, 3), np.float32), np.array([1.0]))
    assert buf[2, 3, 0] == pytest.approx(0.75 * 0.25, rel=1e-5)
    assert buf[3, 4, 0] == pytest.approx(0.25 * 0.75, rel=1e-5)
    assert buf.sum() == pytest.approx(3.0, rel=1e-5)


def test_splat_ignores_offscreen_particles():
    buf = np.zeros((16, 16, 3), np.float32)
    splat(buf, np.array([-5.0, 100.0, 8.0]), np.array([8.0, 8.0, -1.0]),
          np.ones((3, 3), np.float32), np.ones(3))
    assert buf.sum() == 0.0


def test_tonemap_output_contract():
    buf = np.zeros((90, 160, 3), np.float32)
    buf[45, 80] = 40.0
    img = tonemap(buf, exposure=1.0)
    assert img.dtype == np.uint8 and img.shape == (90, 160, 3)
    assert img.max() > 200                        # the bright core survives
    assert img[0, 0].max() < 8                    # empty space stays near-black


def test_tonemap_is_monotonic_in_exposure():
    rng = np.random.default_rng(1)
    buf = rng.random((60, 80, 3)).astype(np.float32) * 0.3
    lo = tonemap(buf, exposure=0.5).mean()
    hi = tonemap(buf, exposure=2.0).mean()
    assert hi > lo


def test_camera_projection_geometry():
    cam = Camera(orbit_eye((0, 0, 0), 100.0, 0.0, 0.0), fov_deg=30.0,
                 width=1920, height=1080)
    sx, sy, depth = cam.project(np.array([[0.0, 0.0, 0.0]]))
    assert sx[0] == pytest.approx(960.0, abs=1e-6)
    assert sy[0] == pytest.approx(540.0, abs=1e-6)
    assert depth[0] == pytest.approx(100.0, rel=1e-9)
    # a point offset along +z appears above the centre
    _, sy2, _ = cam.project(np.array([[0.0, 0.0, 10.0]]))
    assert sy2[0] < 540.0


def test_camera_perspective_shrinks_distant_objects():
    cam = Camera(orbit_eye((0, 0, 0), 200.0, 20.0, 15.0))
    p = np.array([[0.0, 0.0, 10.0], [0.0, 0.0, -10.0]])
    sx, sy, d = cam.project(p)
    near = np.argmin(d)
    off = np.abs(sy - 540.0)
    assert off[near] > off[1 - near]           # nearer point projects further out


def test_population_colours_are_physical():
    ptype = np.array([PTYPE_DISK, PTYPE_DISK, PTYPE_BULGE])
    col = population_colours(ptype, np.array([0.5, 14.0, 1.0]))
    assert col.shape == (3, 3) and col.dtype == np.float32
    # inner disk is warmer (R > B), outer disk is cooler (B > R)
    assert col[0, 0] > col[0, 2]
    assert col[1, 2] > col[1, 0]
    assert col[2, 0] > col[2, 2]               # bulge is amber
    assert np.all((col >= 0) & (col <= 1))


def test_luminosity_scatter_has_unit_mean_and_spread():
    lw = luminosity_scatter(200000, np.random.default_rng(3))
    assert lw.mean() == pytest.approx(1.0, rel=0.03)
    assert lw.std() > 0.3
    assert np.all(lw > 0)


def test_upsample_preserves_shape_and_values():
    a = np.arange(2 * 3 * 3, dtype=np.float32).reshape(2, 3, 3)
    b = _upsample(a, 2)
    assert b.shape == (4, 6, 3)
    assert np.allclose(b[0, 0], a[0, 0])


def test_frame_renderer_produces_a_1080p_frame():
    rng = np.random.default_rng(4)
    n = 4000
    pos = rng.normal(0, 12, (n, 3))
    col = np.ascontiguousarray(rng.random((n, 3)).astype(np.float32))
    w = np.ones(n, np.float32)
    cam = Camera(orbit_eye((0, 0, 0), 180.0, 30.0, 20.0), fov_deg=32.0)
    fr = FrameRenderer(width=1920, height=1080)
    fr.clear()
    fr.add_stars(cam, pos, col, w)
    fr.add_halo(cam, rng.normal(0, 40, (500, 3)), 1.0)
    img = fr.resolve()
    assert img.shape == (1080, 1920, 3) and img.dtype == np.uint8
    assert img.sum() > 0
    assert img[0, 0].max() < 20                 # corners stay dark


def test_empty_frame_is_black():
    fr = FrameRenderer(width=320, height=180)
    fr.clear()
    img = fr.resolve()
    assert img.max() == 0
