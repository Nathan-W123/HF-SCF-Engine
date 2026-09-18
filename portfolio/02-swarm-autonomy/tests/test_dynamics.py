import numpy as np
import pytest

from swarmsim import dynamics as dy
from swarmsim.config import G0, VehicleParams


def _fit_circle(pts):
    x, y = pts[:, 0], pts[:, 1]
    A = np.stack([2 * x, 2 * y, np.ones_like(x)], axis=1)
    sol, *_ = np.linalg.lstsq(A, x ** 2 + y ** 2, rcond=None)
    cx, cy = sol[0], sol[1]
    return np.sqrt(sol[2] + cx ** 2 + cy ** 2)


@pytest.mark.parametrize("V,phi_deg", [(20.0, 15.0), (25.0, 30.0), (32.0, 40.0)])
def test_steady_turn_radius_matches_analytic(V, phi_deg):
    vp = VehicleParams()
    phi = np.deg2rad(phi_deg)
    R = dy.turn_radius(V, phi)
    dt = 0.01
    n = int(round(2.0 * 2 * np.pi * R / V / dt))
    s = np.array([[0.0, 0.0, 800.0, V, 0.0, 0.0, phi]])
    cmd = np.array([[V, phi, 0.0]])
    pts = np.empty((n, 2))
    for k in range(n):
        s = dy.rk4_step(s, cmd, dt, vp, wind=np.zeros((1, 3)))
        pts[k] = s[0, :2]
    assert abs(_fit_circle(pts) - R) / R < 1e-7


def test_turn_direction_right_bank_turns_clockwise():
    vp = VehicleParams()
    s = np.array([[0.0, 0.0, 800.0, 25.0, 0.0, 0.0, np.deg2rad(20.0)]])
    cmd = np.array([[25.0, np.deg2rad(20.0), 0.0]])
    for _ in range(50):
        s = dy.rk4_step(s, cmd, 0.02, vp, wind=np.zeros((1, 3)))
    # A right (positive) bank must decrease the East-to-North azimuth.
    assert s[0, 4] < 0.0


def test_envelope_limits_enforced_under_illegal_commands():
    vp = VehicleParams()
    N = 6
    s = np.zeros((N, 7))
    s[:, 2] = 900.0
    s[:, 3] = vp.V_nom
    s[:, 4] = np.linspace(-3, 3, N)
    cmd = np.tile([[80.0, np.deg2rad(89.0), np.deg2rad(60.0)]], (N, 1))
    for _ in range(600):
        s = dy.rk4_step(s, cmd, 0.02, vp, wind=np.zeros((N, 3)))
    assert s[:, 3].max() <= vp.V_max + 1e-9
    assert s[:, 3].min() >= vp.V_min - 1e-9
    assert np.abs(s[:, 5]).max() <= vp.gamma_max + 1e-9
    assert np.abs(s[:, 6]).max() <= vp.phi_limit + 1e-9
    assert dy.load_factor(s[:, 6]).max() <= vp.load_factor_max + 1e-6


def test_rate_limits_respected():
    vp = VehicleParams()
    dt = 0.01
    s = np.array([[0.0, 0.0, 800.0, vp.V_nom, 0.0, 0.0, 0.0]])
    cmd = np.array([[vp.V_max, vp.phi_limit, vp.gamma_max]])
    prev = s.copy()
    for _ in range(400):
        s = dy.rk4_step(s, cmd, dt, vp, wind=np.zeros((1, 3)))
        assert abs(s[0, 6] - prev[0, 6]) / dt <= vp.p_max * 1.02
        assert abs(s[0, 5] - prev[0, 5]) / dt <= vp.gamma_dot_max * 1.02
        assert abs(s[0, 3] - prev[0, 3]) / dt <= vp.accel_max * 1.02
        prev = s.copy()


def test_rk4_is_fourth_order():
    vp = VehicleParams()
    s0 = np.array([[0.0, 0.0, 800.0, 25.0, 0.4, np.deg2rad(2.0), np.deg2rad(12.0)]])
    cmd = np.array([[26.5, np.deg2rad(22.0), np.deg2rad(5.0)]])
    T = 10.0
    wind = np.array([[2.0, -1.0, 0.3]])

    def integrate(dt):
        s = s0.copy()
        for _ in range(int(round(T / dt))):
            s = dy.rk4_step(s, cmd, dt, vp, wind=wind, enforce_limits=False)
        return s[0, :3]

    ref = integrate(T / 12288)
    dts, errs = [], []
    for m in (256, 512, 1024):
        dt = T / m
        dts.append(dt)
        errs.append(np.linalg.norm(integrate(dt) - ref))
    slope = np.polyfit(np.log(dts), np.log(errs), 1)[0]
    assert 3.7 < slope < 4.3, f"measured order {slope}"


def test_ground_velocity_is_air_plus_wind():
    s = np.array([[0.0, 0.0, 500.0, 25.0, 0.3, 0.05, 0.1]])
    w = np.array([[4.0, -3.0, 0.5]])
    assert np.allclose(dy.ground_velocity(s, w), dy.air_velocity(s) + w)


def test_air_velocity_magnitude_equals_airspeed():
    rng = np.random.default_rng(0)
    s = np.zeros((20, 7))
    s[:, 3] = rng.uniform(18, 34, 20)
    s[:, 4] = rng.uniform(-np.pi, np.pi, 20)
    s[:, 5] = rng.uniform(-0.2, 0.2, 20)
    assert np.allclose(np.linalg.norm(dy.air_velocity(s), axis=1), s[:, 3])


def test_wrap_pi():
    a = np.array([0.0, np.pi, -np.pi, 3 * np.pi, -3.5 * np.pi])
    w = dy.wrap_pi(a)
    assert np.all(w > -np.pi - 1e-12) and np.all(w <= np.pi + 1e-12)
    assert np.allclose(np.cos(a), np.cos(w))
    assert np.allclose(np.sin(a), np.sin(w))


def test_commands_from_velocity_respects_bank_limit():
    vp = VehicleParams()
    rng = np.random.default_rng(3)
    N = 50
    s = np.zeros((N, 7))
    s[:, 3] = vp.V_nom
    s[:, 4] = rng.uniform(-np.pi, np.pi, N)
    v = rng.normal(size=(N, 3)) * 20.0
    cmd = dy.commands_from_velocity(s, v, vp, np.full(N, 200.0), 0.1)
    assert np.all(np.abs(cmd[:, 1]) <= vp.phi_limit + 1e-12)
    assert np.all(np.abs(cmd[:, 2]) <= vp.gamma_max + 1e-12)
    assert np.all(cmd[:, 0] >= vp.V_min - 1e-12)
    assert np.all(cmd[:, 0] <= vp.V_max + 1e-12)
