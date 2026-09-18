"""The numba kernel must reproduce the readable reference implementation."""
import numpy as np
import pytest

from swarmsim.avoidance import CollisionAvoidance
from swarmsim.config import AvoidanceParams, NoFlyZone, VehicleParams

AP = AvoidanceParams()
VP = VehicleParams()
ZONES = [NoFlyZone(0.0, 0.0, 300.0, 0.0, 2000.0),
         NoFlyZone(800.0, -400.0, 260.0, 0.0, 2000.0)]


def _random_state(rng, n, spread=900.0):
    pos = rng.uniform(-spread, spread, (n, 3))
    pos[:, 2] = rng.uniform(400.0, 1400.0, n)
    dirs = rng.normal(size=(n, 3))
    dirs[:, 2] *= 0.15
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    vel = dirs * rng.uniform(VP.V_min, VP.V_max, (n, 1))
    pref = vel + rng.normal(size=(n, 3)) * 1.5
    return pos, vel, pref


@pytest.mark.parametrize("seed,n,zones", [(0, 12, []), (1, 20, ZONES),
                                          (2, 30, ZONES), (3, 8, [])])
def test_kernel_matches_reference(seed, n, zones):
    rng = np.random.default_rng(seed)
    ca = CollisionAvoidance(AP, VP, zones=zones)
    pos, vel, pref = _random_state(rng, n)
    coop = rng.random(n) > 0.12
    v_ref, d_ref = ca.filter(pos, vel, pref, 0.1, cooperative=coop)
    v_fast, d_fast = ca.filter_fast(pos, vel, pref, 0.1, cooperative=coop)
    assert np.abs(v_ref - v_fast).max() < 1e-9
    assert np.array_equal(d_ref.n_constraints, d_fast.n_constraints)
    assert np.array_equal(d_ref.fallback_used, d_fast.fallback_used)


def test_kernel_matches_reference_in_a_dense_cluster():
    """Dense case: exercises the infeasible / fallback branch."""
    rng = np.random.default_rng(7)
    ca = CollisionAvoidance(AP, VP, zones=ZONES)
    pos, vel, pref = _random_state(rng, 36, spread=180.0)
    v_ref, d_ref = ca.filter(pos, vel, pref, 0.1)
    v_fast, d_fast = ca.filter_fast(pos, vel, pref, 0.1)
    assert np.abs(v_ref - v_fast).max() < 1e-9
    assert np.array_equal(d_ref.fallback_used, d_fast.fallback_used)
    assert d_ref.fallback_used.sum() > 0      # the branch was actually taken


def test_kernel_is_used_by_default():
    from swarmsim._kernels import HAVE_NUMBA
    assert HAVE_NUMBA, "numba is pinned in requirements.txt and must be present"
