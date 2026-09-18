import numpy as np
import pytest

from swarmsim.render import (Camera, activity_colour, attitude_matrix,
                             circle_points, splat, tonemap)


def test_camera_projects_target_to_image_centre():
    cam = Camera(eye=[0, -1000, 300], target=[0, 0, 300], width=1920, height=1080)
    xy, z, vis = cam.project(np.array([[0.0, 0.0, 300.0]]))
    assert vis[0]
    assert xy[0, 0] == pytest.approx(960.0, abs=1e-6)
    assert xy[0, 1] == pytest.approx(540.0, abs=1e-6)
    assert z[0] == pytest.approx(1000.0)


def test_camera_up_is_up_and_right_is_right():
    cam = Camera(eye=[0, -1000, 0], target=[0, 0, 0], width=1920, height=1080)
    xy, _, _ = cam.project(np.array([[0.0, 0.0, 100.0], [100.0, 0.0, 0.0]]))
    assert xy[0, 1] < 540.0        # higher in world -> smaller pixel row
    assert xy[1, 0] > 960.0        # +east is to the right for this pose


def test_camera_rejects_points_behind():
    cam = Camera(eye=[0, -1000, 0], target=[0, 0, 0])
    _, _, vis = cam.project(np.array([[0.0, -2000.0, 0.0]]))
    assert not vis[0]


def test_perspective_size_falls_off_with_distance():
    cam = Camera(eye=[0, -1000, 0], target=[0, 0, 0])
    near, _, _ = cam.project(np.array([[0.0, 0.0, 0.0], [50.0, 0.0, 0.0]]))
    far, _, _ = cam.project(np.array([[0.0, 3000.0, 0.0], [50.0, 3000.0, 0.0]]))
    assert abs(near[1, 0] - near[0, 0]) > 3 * abs(far[1, 0] - far[0, 0])


def test_splat_conserves_energy():
    buf = np.zeros((64, 64, 3), np.float32)
    x = np.array([10.3, 30.7, 50.1])
    y = np.array([20.6, 40.2, 12.9])
    rgb = np.ones((3, 3), np.float32)
    w = np.array([1.0, 2.0, 3.0], np.float32)
    splat(buf, x, y, rgb, w)
    assert buf.sum() == pytest.approx(3 * w.sum(), rel=1e-5)


def test_splat_ignores_out_of_bounds():
    buf = np.zeros((16, 16, 3), np.float32)
    splat(buf, np.array([-5.0, 100.0, np.nan]), np.array([5.0, 5.0, 5.0]),
          np.ones((3, 3), np.float32), np.ones(3, np.float32))
    assert buf.sum() == 0.0


def test_tonemap_output_shape_and_range():
    rng = np.random.default_rng(0)
    buf = rng.random((90, 160, 3)).astype(np.float32) * 4.0
    out = tonemap(buf)
    assert out.shape == (90, 160, 3) and out.dtype == np.uint8


def test_tonemap_is_monotone_in_exposure():
    buf = np.full((32, 32, 3), 0.25, np.float32)
    lo = tonemap(buf, exposure=0.5).mean()
    hi = tonemap(buf, exposure=2.0).mean()
    assert hi > lo


def test_activity_colour_is_continuous_and_bounded():
    q = np.linspace(0, 1, 501)
    c = activity_colour(q)
    assert c.shape == (501, 3)
    assert c.min() >= 0.0 and c.max() <= 1.0
    assert np.abs(np.diff(c, axis=0)).max() < 0.05      # no palette jumps


def test_activity_colour_warms_with_activity():
    cold = activity_colour(np.array([0.0]))[0]
    hot = activity_colour(np.array([1.0]))[0]
    assert hot[0] > cold[0] and hot[2] < cold[2]


def test_attitude_matrix_is_orthonormal():
    rng = np.random.default_rng(2)
    n = 40
    R = attitude_matrix(rng.uniform(-np.pi, np.pi, n),
                        rng.uniform(-0.2, 0.2, n), rng.uniform(-0.8, 0.8, n))
    eye = np.einsum("nij,nkj->nik", R, R)
    assert np.allclose(eye, np.eye(3)[None, :, :], atol=1e-10)
    assert np.allclose(np.linalg.det(R), 1.0)


def test_attitude_matrix_body_x_matches_velocity_direction():
    from swarmsim.dynamics import air_velocity
    psi = np.array([0.7]); gam = np.array([0.1]); phi = np.array([0.4])
    R = attitude_matrix(psi, gam, phi)
    fwd = R[0] @ np.array([1.0, 0.0, 0.0])
    s = np.array([[0, 0, 0, 25.0, psi[0], gam[0], phi[0]]])
    v = air_velocity(s)[0]
    assert np.allclose(fwd, v / np.linalg.norm(v), atol=1e-12)


def test_circle_points_lie_on_the_circle():
    p = circle_points(10.0, -5.0, 300.0, 250.0, 64)
    assert p.shape == (64, 3)
    assert np.allclose(np.hypot(p[:, 0] - 10.0, p[:, 1] + 5.0), 250.0)
    assert np.allclose(p[:, 2], 300.0)
