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


def test_camera_roll_rotates_the_image_plane():
    import numpy as np
    from swarmsim.render import Camera
    pts = np.array([[0.0, 0.0, 400.0]])
    a = Camera(eye=[0, -2000, 0], target=[0, 0, 0], roll_deg=0.0)
    b = Camera(eye=[0, -2000, 0], target=[0, 0, 0], roll_deg=90.0)
    xa, _, _ = a.project(pts)
    xb, _, _ = b.project(pts)
    # a point directly above the target moves to the side under a 90 deg roll
    assert abs(xa[0, 0] - 960.0) < 1e-6 and xa[0, 1] < 540.0
    assert abs(xb[0, 1] - 540.0) < 1e-6 and abs(xb[0, 0] - 960.0) > 50.0
    # roll must not change the depth of any point
    _, za, _ = a.project(pts)
    _, zb, _ = b.project(pts)
    assert za[0] == pytest.approx(zb[0])


def test_camera_basis_stays_orthonormal_under_roll():
    import numpy as np
    from swarmsim.render import Camera
    for roll in (0.0, 17.0, 50.0, -33.0):
        c = Camera(eye=[1200, -900, 700], target=[0, 0, 500], roll_deg=roll)
        assert np.allclose(c.R @ c.R.T, np.eye(3), atol=1e-12)
        # rows are (right, up, forward); that triad is left-handed by
        # construction, which is the usual screen-space convention and is what
        # the y-flip in Camera.project pairs with.
        assert np.linalg.det(c.R) == pytest.approx(-1.0, abs=1e-12)


def test_altitude_colour_is_monotone_and_cool():
    import numpy as np
    from swarmsim.render import altitude_colour
    lo = altitude_colour(np.array([0.0]))[0]
    hi = altitude_colour(np.array([1.0]))[0]
    assert lo[2] > lo[0] and hi[1] > hi[0]      # both cool (blue/cyan dominant)
    assert hi.sum() > lo.sum()                   # higher altitude is brighter


def test_activity_colour_uses_altitude_for_the_base_hue():
    import numpy as np
    from swarmsim.render import activity_colour
    q = np.zeros(2)
    c = activity_colour(q, altitude=np.array([0.0, 1.0]))
    assert not np.allclose(c[0], c[1])


def test_glyph_mesh_facet_normals_are_varied():
    import numpy as np
    from swarmsim.render import _MESH_F, _MESH_V
    ns = []
    for idx, _, _ in _MESH_F:
        p = _MESH_V[list(idx)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        ns.append(n / np.linalg.norm(n))
    ns = np.array(ns)
    # at least one near-horizontal (wing) and one near-vertical (fin) facet
    assert np.abs(ns[:, 2]).max() > 0.9
    assert np.abs(ns[:, 2]).min() < 0.1
