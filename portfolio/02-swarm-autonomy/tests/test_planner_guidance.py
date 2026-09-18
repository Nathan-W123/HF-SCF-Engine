import numpy as np
import pytest

from swarmsim.config import GuidanceParams, NoFlyZone, VehicleParams
from swarmsim.guidance import (CirclePath, PolylinePath, l1_distance,
                               lateral_acceleration, preferred_velocity)
from swarmsim.planner import (VisibilityGraphPlanner, path_length,
                              segment_circle_blocked)

ZONES = [NoFlyZone(0, 0, 300), NoFlyZone(700, 400, 250), NoFlyZone(-600, 500, 400),
         NoFlyZone(300, -700, 350), NoFlyZone(-300, -1100, 280)]


def _penetration(wp, centres, radii):
    worst = -np.inf
    for i in range(len(wp) - 1):
        p0, p1 = wp[i], wp[i + 1]
        d = p1 - p0
        dd = max(float(d @ d), 1e-12)
        t = np.clip(((centres - p0) @ d) / dd, 0, 1)
        proj = p0[None] + t[:, None] * d[None]
        worst = max(worst, float((radii - np.linalg.norm(centres - proj, axis=1)).max()))
    return worst


@pytest.mark.parametrize("k", range(12))
def test_planned_paths_clear_inflated_zones(k):
    p = VisibilityGraphPlanner(ZONES, clearance=90.0)
    th = 2 * np.pi * k / 12
    a = np.array([2400 * np.cos(th), 2400 * np.sin(th)])
    res = p.plan(a, -a)
    assert res.ok
    # never penetrate the inflated disc by more than a centimetre
    assert _penetration(res.waypoints, p.centres, p.radii) < 1e-2
    # and therefore always keep essentially the full clearance to the real zone
    assert _penetration(res.waypoints, p.centres,
                        p.radii - p.clearance) < -p.clearance + 1e-2


def test_planner_returns_direct_line_when_unobstructed():
    p = VisibilityGraphPlanner(ZONES, clearance=90.0)
    a = np.array([-3000.0, 2600.0])
    b = np.array([3000.0, 2600.0])
    res = p.plan(a, b)
    assert res.ok and len(res.waypoints) == 2
    assert res.length == pytest.approx(np.linalg.norm(b - a))


def test_planner_detour_is_modest():
    p = VisibilityGraphPlanner(ZONES, clearance=90.0)
    ratios = []
    for k in range(16):
        th = 2 * np.pi * k / 16
        a = np.array([2600 * np.cos(th), 2600 * np.sin(th)])
        res = p.plan(a, -a)
        ratios.append(res.length / np.linalg.norm(2 * a))
    assert max(ratios) < 1.35 and min(ratios) >= 1.0 - 1e-9


def test_planner_with_no_zones_is_a_straight_line():
    p = VisibilityGraphPlanner([], clearance=50.0)
    res = p.plan([0, 0], [1000, 500])
    assert res.ok and len(res.waypoints) == 2


def test_segment_circle_blocked_basic():
    c = np.array([[0.0, 0.0]])
    r = np.array([100.0])
    assert segment_circle_blocked([-500, 0], [500, 0], c, r)[0]
    assert not segment_circle_blocked([-500, 200], [500, 200], c, r)[0]
    # a tangent segment is not blocked
    assert not segment_circle_blocked([-500, 100], [500, 100], c, r)[0]


def test_polyline_projection_and_cross_track_sign():
    path = PolylinePath(np.array([[0.0, 0.0], [1000.0, 0.0]]))
    s, q, xte = path.project(np.array([300.0, 40.0]))
    assert s == pytest.approx(300.0)
    assert q == pytest.approx(np.array([300.0, 0.0]))
    assert xte == pytest.approx(40.0)          # left of +x direction is +y
    _, _, xte2 = path.project(np.array([300.0, -40.0]))
    assert xte2 == pytest.approx(-40.0)


def test_l1_reference_is_at_L1_distance_on_straight_path():
    path = PolylinePath(np.array([[0.0, 0.0], [5000.0, 0.0]]))
    p = np.array([100.0, 60.0])
    L1 = 300.0
    ref = path.l1_reference(p, L1)
    assert np.linalg.norm(ref - p) == pytest.approx(L1, abs=1e-6)


def test_l1_reference_on_circle_is_at_L1_distance():
    path = CirclePath((0.0, 0.0), 600.0, +1)
    for off in (-150.0, -20.0, 30.0, 180.0):
        p = np.array([600.0 + off, 0.0])
        ref = path.l1_reference(p, 250.0)
        assert np.linalg.norm(ref - p) == pytest.approx(250.0, abs=1e-6)


def test_l1_reference_degenerates_gracefully_when_offset_exceeds_L1():
    path = PolylinePath(np.array([[0.0, 0.0], [5000.0, 0.0]]))
    ref = path.l1_reference(np.array([100.0, 900.0]), 300.0)
    assert ref == pytest.approx(np.array([100.0, 0.0]))


def test_lateral_acceleration_sign_and_magnitude():
    vg = np.array([25.0, 0.0])
    # reference point 90 degrees to the left -> positive (left) acceleration
    a = lateral_acceleration(vg, np.array([0.0, 200.0]), 200.0)
    assert a == pytest.approx(2 * 25.0 ** 2 / 200.0)
    a2 = lateral_acceleration(vg, np.array([0.0, -200.0]), 200.0)
    assert a2 == pytest.approx(-2 * 25.0 ** 2 / 200.0)
    assert lateral_acceleration(vg, np.array([200.0, 0.0]), 200.0) == pytest.approx(0.0)


def test_l1_distance_is_clipped():
    gp = GuidanceParams()
    assert l1_distance(1.0, gp) == gp.L1_min
    assert l1_distance(1000.0, gp) == gp.L1_max


def test_preferred_velocity_climbs_towards_reference_altitude():
    gp = GuidanceParams()
    path = PolylinePath(np.array([[0.0, 0.0], [5000.0, 0.0]]))
    v, L1 = preferred_velocity(np.array([0.0, 0.0, 500.0]),
                               np.array([25.0, 0.0, 0.0]), path, 900.0, 25.0, gp)
    assert v[2] > 0 and v[2] <= gp.climb_rate_max + 1e-9
    v2, _ = preferred_velocity(np.array([0.0, 0.0, 1300.0]),
                               np.array([25.0, 0.0, 0.0]), path, 900.0, 25.0, gp)
    assert v2[2] < 0


def test_path_length_helper():
    wp = np.array([[0.0, 0.0], [3.0, 4.0], [3.0, 14.0]])
    assert path_length(wp) == pytest.approx(15.0)
