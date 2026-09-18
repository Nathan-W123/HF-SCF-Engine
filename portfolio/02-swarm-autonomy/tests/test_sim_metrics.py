import numpy as np
import pytest

from swarmsim import metrics, scenarios, sim
from swarmsim.config import NoFlyZone, SimParams, WindParams
from swarmsim.sim import SwarmSpec, build_paths


def _tiny(n=6, t_max=110.0, zones=None, alt_band=(500.0, 1100.0), **kw):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    R = 900.0
    alt = np.linspace(alt_band[0], alt_band[1], n)
    starts = np.stack([R * np.cos(th), R * np.sin(th), alt], axis=1)
    goals = np.stack([-R * np.cos(th), -R * np.sin(th), alt], axis=1)
    return SwarmSpec(name="tiny", starts=starts, goals=goals,
                     zones=zones or [], sim=SimParams(t_max=t_max, seed=1), **kw)


def test_tiny_scenario_runs_and_is_shaped_correctly():
    r = sim.run(_tiny())
    T = len(r.t)
    assert r.states.shape == (T, 6, 7)
    assert r.wind.shape == (T, 6, 3)
    assert r.v_ground.shape == (T, 6, 3)
    assert np.isfinite(r.states).all()


def test_simulation_is_deterministic_for_a_fixed_seed():
    a = sim.run(_tiny())
    b = sim.run(_tiny())
    assert np.array_equal(a.states, b.states)
    assert np.array_equal(a.deflection, b.deflection)


def test_avoidance_improves_separation():
    # A coplanar antipodal swap: without the avoidance layer these eight
    # vehicles fly straight through one another.
    kw = dict(n=8, t_max=110.0, alt_band=(800.0, 800.0))
    on = sim.run(_tiny(**kw))
    off = sim.run(_tiny(avoidance_on=False, **kw))
    m_on = metrics.scenario_metrics(on)
    m_off = metrics.scenario_metrics(off)
    assert m_off["min_separation_m"] < m_off["R_collision_m"]
    assert m_on["min_separation_m"] >= m_on["R_min_m"]
    assert m_on["min_separation_m"] > 8 * m_off["min_separation_m"]


def test_envelope_is_respected_in_a_full_run():
    r = sim.run(_tiny())
    env = metrics.envelope_check(r)
    assert env["V_within_limits"]
    assert env["phi_within_limits"]
    assert env["gamma_within_limits"]
    assert env["load_factor_within_limits"]


def test_ground_velocity_log_equals_air_plus_wind():
    from swarmsim.dynamics import air_velocity
    r = sim.run(_tiny(wind=WindParams(mean=(5.0, -3.0, 0.0))))
    k = len(r.t) // 2
    assert np.allclose(r.v_ground[k], air_velocity(r.states[k]) + r.wind[k])


def test_zone_avoidance_keeps_vehicles_out_of_no_fly_cylinders():
    zones = [NoFlyZone(0.0, 0.0, 260.0, 0.0, 2000.0)]
    r = sim.run(_tiny(n=8, t_max=120.0, zones=zones))
    z = metrics.zone_incursions(r)
    assert z["zone_incursion_samples"] == 0
    assert z["min_zone_clearance_m"] > 0.0


def test_failure_injection_makes_one_vehicle_non_cooperative():
    spec = _tiny(n=8, t_max=70.0)
    spec.fail_id = 3
    spec.fail_time = 20.0
    spec.fail_bank = np.deg2rad(15.0)
    r = sim.run(spec)
    assert r.cooperative[0, 3]
    assert not r.cooperative[-1, 3]
    assert r.cooperative[-1].sum() == 7


def test_build_paths_reports_no_failures_for_reachable_goals():
    zones = [NoFlyZone(0.0, 0.0, 300.0)]
    paths, lengths, fails = build_paths(_tiny(n=6, zones=zones))
    assert fails == 0
    assert len(paths) == 6 and np.all(lengths > 0)


def test_metrics_keys_and_consistency():
    r = sim.run(_tiny(t_max=140.0))
    m = metrics.scenario_metrics(r)
    for k in ("min_separation_m", "separation_violation_samples", "collisions",
              "path_efficiency_mean", "goal_completion", "control_effort_mean_rad_s",
              "sampling_gap_m"):
        assert k in m
    assert m["goal_completion"] > 0.0
    assert m["path_efficiency_mean"] >= 1.0 - 1e-6
    assert 0.0 <= m["goal_completion"] <= 1.0
    assert m["collisions"] <= m["separation_violation_samples"]


def test_pairwise_min_distance_matches_brute_force():
    r = sim.run(_tiny(n=5, t_max=25.0))
    d, iu, ju = metrics.pairwise_min_distance(r.states)
    k = 7
    pos = r.states[k, :, :3]
    brute = min(np.linalg.norm(pos[i] - pos[j])
                for i in range(5) for j in range(i + 1, 5))
    assert d[k].min() == pytest.approx(brute)


def test_scenario_factories_build_valid_specs():
    for name, fn in scenarios.ALL.items():
        spec = fn()
        assert spec.starts.shape == spec.goals.shape
        assert spec.starts.shape[1] == 3
        assert len(spec.starts) >= 10
        # goals must be far enough apart to be simultaneously occupiable
        g = spec.goals
        d = np.linalg.norm(g[:, None, :] - g[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        assert d.min() > spec.avoid.R_min, name
