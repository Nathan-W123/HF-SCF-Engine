"""End-to-end smoke test of the production pipeline at tiny N."""
import json

import numpy as np
import pytest

from galcol.ics import Encounter, Resolution, make_collision
from galcol.production import ENCOUNTER, MODEL, QUICK_SIM, RESOLUTION
from galcol.profiles import default_galaxy
from galcol.simulate import (SimConfig, angmom_error, energy_error,
                             run_simulation)
from galcol.units import TIME_UNIT_GYR

TINY = Resolution(n_disk=600, n_bulge=200, n_halo=700,
                  eps_disk=0.5, eps_bulge=0.5, eps_halo=1.2)


def test_short_collision_runs_and_conserves(tmp_path):
    p = make_collision(default_galaxy(), TINY, Encounter())
    cfg = SimConfig(dt=0.003, t_end_gyr=0.12, theta=0.7, leaf_size=12,
                    snap_dt_gyr=0.03, diag_every=2,
                    out_dir=str(tmp_path / "run"), label="test")
    diag, meta = run_simulation(p, cfg, progress=False, store_snapshots=True)

    assert meta["n_particles"] == p.n
    assert meta["n_steps"] == cfg.n_steps
    assert energy_error(diag)["max_rel_error"] < 5e-3
    assert angmom_error(diag)["max_rel_error"] < 1e-2

    # The initial conditions carry exactly zero net momentum.  A Barnes-Hut
    # force is NOT exactly antisymmetric -- two particles in different cells
    # see each other through different multipole approximations -- so momentum
    # is conserved only to the tree's force accuracy, not to machine
    # precision.  What matters is that the resulting centre-of-mass drift is
    # negligible against the internal velocities (~220 km/s).
    P = np.stack([diag["px"], diag["py"], diag["pz"]], axis=1)
    com_speed = np.linalg.norm(P, axis=1).max() / p.mass.sum()
    assert com_speed < 1.0          # km/s

    snaps = sorted((tmp_path / "run").glob("snap_*.npz"))
    assert len(snaps) >= 3
    hdr = np.load(tmp_path / "run" / "header.npz", allow_pickle=True)
    cfg_back = json.loads(str(hdr["config"]))
    assert cfg_back["dt"] == cfg.dt
    d = np.load(snaps[-1])
    assert d["star_pos"].shape[1] == 3
    assert d["star_pos"].dtype == np.float32
    assert float(d["t_gyr"]) == pytest.approx(cfg.t_end * TIME_UNIT_GYR, rel=0.2)


def test_probe_hook_is_called(tmp_path):
    p = make_collision(default_galaxy(), TINY, Encounter())
    cfg = SimConfig(dt=0.003, t_end_gyr=0.06, theta=0.8,
                    out_dir=str(tmp_path / "r2"), diag_every=5)
    seen = []
    run_simulation(p, cfg, progress=False, store_snapshots=False,
                   probe=lambda s, t, pos, vel: seen.append((s, t)),
                   probe_every=4)
    assert len(seen) >= 3
    assert seen[0][0] == 0
    assert seen[-1][0] == cfg.n_steps


def test_production_config_is_self_consistent():
    assert RESOLUTION.n_disk > 0 and RESOLUTION.n_halo > 0
    assert 0.3 <= ENCOUNTER.ecc < 1.0
    assert ENCOUNTER.r_start > ENCOUNTER.r_peri
    assert QUICK_SIM.n_steps > 10
    assert MODEL.disk.r_trunc > MODEL.disk.r_scale
    # softening must be smaller than the disk scale height's neighbourhood
    assert RESOLUTION.eps_disk < MODEL.disk.z_scale * 1.5
    # the tidal encounter must be strong: pericentre within a few scale lengths
    assert ENCOUNTER.r_peri < 6.0 * MODEL.disk.r_scale


def test_live_halo_is_used_in_production():
    from galcol.production import build_initial_conditions
    p = build_initial_conditions(quick=True)
    assert (p.ptype == 2).sum() > 0
    assert p.mass[p.ptype == 2].sum() > p.mass[p.ptype != 2].sum()
