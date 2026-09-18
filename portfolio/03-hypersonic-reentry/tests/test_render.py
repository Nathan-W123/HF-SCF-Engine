"""Tests for the rendering pipeline (splatting, tone mapping, camera, scene)."""

import numpy as np
import pytest

from reentry import Vehicle, simulate
from reentry.atmosphere import USSA76
from reentry.constants import R_EARTH, SIGMA_SB
from reentry.dynamics import state_to_cartesian
from reentry.render.compose import bloom, splat, tonemap, upsample_bilinear
from reentry.render.hero import CameraPath, HeroScene
from reentry.render.raymarch import RAYLEIGH_BETA, build_noise_volumes


@pytest.fixture(scope="module")
def scene():
    res = simulate(vehicle=Vehicle(), atmosphere=USSA76(), gamma0_deg=-5.5,
                   velocity0=7800.0, rtol=1e-8, atol=1e-8, n_output=801)
    pos, _ = state_to_cartesian(res.radius, res.lon, res.lat, res.velocity,
                                res.gamma, res.psi)
    return HeroScene(positions=pos, times=res.t, q_dot=res.q_dot,
                     altitude=res.altitude, trail_samples=4000, n_stars=400)


def test_splat_conserves_energy():
    buf = np.zeros((40, 60, 3), np.float32)
    rng = np.random.default_rng(0)
    n = 500
    x = rng.uniform(2, 55, n)
    y = rng.uniform(2, 35, n)
    rgb = np.ones((n, 3))
    w = rng.uniform(0.1, 2.0, n)
    splat(buf, x, y, rgb, w)
    assert buf.sum() == pytest.approx(3.0 * w.sum(), rel=1e-5)


def test_splat_ignores_out_of_bounds_and_nan():
    buf = np.zeros((10, 10, 3), np.float32)
    x = np.array([-5.0, 20.0, np.nan, 5.0])
    y = np.array([5.0, 5.0, 5.0, 5.0])
    splat(buf, x, y, np.ones((4, 3)), np.ones(4))
    assert buf.sum() == pytest.approx(3.0, rel=1e-6)


def test_tonemap_output_range_and_dtype():
    buf = np.abs(np.random.default_rng(1).standard_normal((32, 48, 3))).astype(np.float32)
    img = tonemap(buf * 5.0)
    assert img.dtype == np.uint8 and img.shape == (32, 48, 3)
    assert img.min() >= 0 and img.max() <= 255


def test_tonemap_is_monotone_in_exposure():
    buf = np.full((8, 8, 3), 0.2, np.float32)
    a = tonemap(buf, exposure=0.5).mean()
    b = tonemap(buf, exposure=2.0).mean()
    assert b > a


def test_bloom_is_non_negative_and_energy_spreading():
    buf = np.zeros((64, 64, 3), np.float32)
    buf[32, 32] = 10.0
    g = bloom(buf)
    assert np.all(g >= 0.0)
    assert np.count_nonzero(g) > 100


def test_upsample_bilinear_shape_and_bounds():
    small = np.random.default_rng(2).random((17, 23, 3)).astype(np.float32)
    up = upsample_bilinear(small, 80, 100)
    assert up.shape == (80, 100, 3)
    assert up.min() >= small.min() - 1e-6 and up.max() <= small.max() + 1e-6


def test_noise_volumes_are_normalised_and_periodic_friendly():
    a, b = build_noise_volumes(seed=3, n_a=16, n_b=16, sigma_a=2.0, sigma_b=1.0)
    for v in (a, b):
        assert v.shape == (16, 16, 16)
        assert v.min() == pytest.approx(0.0, abs=1e-12)
        assert v.max() == pytest.approx(1.0, abs=1e-12)


def test_rayleigh_coefficients_are_blue_biased():
    assert RAYLEIGH_BETA[2] > RAYLEIGH_BETA[1] > RAYLEIGH_BETA[0]


def test_camera_basis_is_orthonormal(scene):
    for s in (0.0, 0.5, 1.0):
        t = scene.times[0] + s * (scene.times[-1] - scene.times[0])
        eye, fwd, right, up, tan_half = scene.camera_at(t, s)
        for v in (fwd, right, up):
            assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-9)
        assert abs(np.dot(fwd, right)) < 1e-9
        assert abs(np.dot(fwd, up)) < 1e-9
        assert abs(np.dot(right, up)) < 1e-9
        assert np.linalg.norm(eye) > R_EARTH
        assert tan_half > 0.0


def test_trail_colour_follows_the_planck_locus_of_the_wall_temperature(scene):
    """The trail colour must be the blackbody colour of T_w = (q/(eps sigma))^1/4."""
    tw = (np.maximum(scene.trail_q, 1.0) / (scene.emissivity * SIGMA_SB)) ** 0.25
    assert np.allclose(tw, scene.trail_tw, rtol=1e-12)
    hottest = int(np.argmax(scene.trail_q))
    coolest = int(np.argmin(scene.trail_q))
    # hotter wall => relatively more blue/green than the coolest sample
    assert scene.trail_rgb[hottest][2] > scene.trail_rgb[coolest][2]
    assert scene.trail_tw[hottest] > scene.trail_tw[coolest]


def test_trail_is_resampled_uniformly_in_arc_length(scene):
    d = np.linalg.norm(np.diff(scene.trail_pos, axis=0), axis=1)
    assert np.std(d) / np.mean(d) < 1e-4


def test_occlusion_hides_points_behind_the_earth(scene):
    eye = np.array([0.0, 0.0, 3.0 * R_EARTH])
    behind = np.array([[0.0, 0.0, -2.0 * R_EARTH]])
    front = np.array([[0.0, 0.0, 1.5 * R_EARTH]])
    assert scene._occluded(behind, eye, R_EARTH)[0]
    assert not scene._occluded(front, eye, R_EARTH)[0]


def test_scene_renders_a_finite_frame(scene):
    img = scene.render_image(90.0, 0.4, 160, 90, n_steps=8)
    assert img.shape == (90, 160, 3) and img.dtype == np.uint8
    assert np.isfinite(img).all()
    assert img.max() > 40          # something is actually lit
    assert img.mean() < 200        # and it is not a white-out


def test_frames_track_the_heating_history(scene):
    """The rendered trail must brighten into peak heating and fade afterwards."""
    i_peak = int(np.argmax(scene.q_dot))
    t_peak = float(scene.times[i_peak])
    lum = []
    for t in (t_peak - 45.0, t_peak, t_peak + 55.0):
        buf, _ = scene.render_frame(t, 0.4, 192, 108, n_steps=6)
        lum.append(float(buf.max()))
    assert lum[1] > lum[0] and lum[1] > lum[2]


def test_camera_path_interpolation_endpoints():
    cp = CameraPath()
    off0, fov0, roll0, blend0, yaw0 = cp.at(0.0)
    off1, fov1, roll1, blend1, yaw1 = cp.at(1.0)
    assert off0 == pytest.approx(cp.start_offset)
    assert off1 == pytest.approx(cp.end_offset)
    assert fov0 == pytest.approx(cp.start_fov_deg)
    assert fov1 == pytest.approx(cp.end_fov_deg)
