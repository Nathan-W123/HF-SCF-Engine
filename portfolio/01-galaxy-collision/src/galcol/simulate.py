"""Production simulation driver: runs the N-body integration, writes
compressed snapshots and records conserved-quantity diagnostics.

Snapshots are written as compressed .npz so that rendering is completely
decoupled from simulation -- a re-render never re-runs the physics.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np

from .ics import ParticleSet
from .integrator import (TreeGravity, angular_momentum, kinetic_energy,
                         linear_momentum, potential_energy,
                         timestep_criterion)
from .tree import build_tree
from .units import TIME_UNIT_GYR, PTYPE_HALO


@dataclass
class SimConfig:
    dt: float = 0.001                  # code time (= 0.978 Myr)
    t_end_gyr: float = 2.2
    theta: float = 0.8
    leaf_size: int = 12
    snap_dt_gyr: float = 0.008
    diag_every: int = 5
    out_dir: str = "data/collision"
    n_halo_render: int = 8000          # halo particles kept in each snapshot
    label: str = "collision"

    @property
    def t_end(self):
        return self.t_end_gyr / TIME_UNIT_GYR

    @property
    def n_steps(self):
        return int(round(self.t_end / self.dt))

    @property
    def snap_every(self):
        return max(1, int(round((self.snap_dt_gyr / TIME_UNIT_GYR) / self.dt)))


def _galaxy_centre(pos, mask):
    """Robust centre of a galaxy: median of its bulge (or disk) particles.

    The median is insensitive to the tidal debris that a mean would follow.
    """
    if mask.sum() == 0:
        return np.full(3, np.nan)
    return np.median(pos[mask], axis=0)


def run_simulation(p: ParticleSet, cfg: SimConfig, progress=True,
                   store_snapshots=True, probe=None, probe_every=None):
    """Run the integration.

    ``probe(step, t, pos, vel)`` -- if given -- is called every
    ``probe_every`` steps with the live (synchronised) state, which lets a
    validation script build a structural time series without a second pass
    over the data.
    """
    out = Path(cfg.out_dir)
    if store_snapshots:
        out.mkdir(parents=True, exist_ok=True)

    n = p.n
    pos, vel = p.pos, p.vel
    mass, eps2 = p.mass, p.eps2
    gravity = TreeGravity(theta=cfg.theta, leaf_size=cfg.leaf_size)

    star = p.ptype != PTYPE_HALO
    star_idx = np.flatnonzero(star)
    halo_idx = np.flatnonzero(~star)
    rng = np.random.default_rng(12345)
    if halo_idx.size > cfg.n_halo_render:
        halo_sub = np.sort(rng.choice(halo_idx, cfg.n_halo_render, replace=False))
    else:
        halo_sub = halo_idx

    bulge1 = (p.ptype == 1) & (p.gid == 0)
    bulge2 = (p.ptype == 1) & (p.gid == 1)
    if bulge1.sum() == 0:
        bulge1 = (p.ptype == 0) & (p.gid == 0)
        bulge2 = (p.ptype == 0) & (p.gid == 1)

    if store_snapshots:
        np.savez_compressed(
            out / "header.npz",
            mass=mass.astype(np.float32), eps=p.eps.astype(np.float32),
            ptype=p.ptype, gid=p.gid,
            star_idx=star_idx.astype(np.int32),
            halo_idx=halo_sub.astype(np.int32),
            config=json.dumps(asdict(cfg)),
        )

    diag = {k: [] for k in ("step", "t", "ekin", "epot", "etot", "lx", "ly",
                            "lz", "px", "py", "pz", "sep", "c1x", "c1y", "c1z",
                            "c2x", "c2y", "c2z")}
    snaps = []
    step_times = []

    n_steps = cfg.n_steps
    snap_every = cfg.snap_every
    half = 0.5 * cfg.dt

    acc, pot = gravity(pos, mass, eps2, want_potential=True)
    dt_req = timestep_criterion(acc, p.eps, eta=0.025)

    def record(step, t, want_pot_arr):
        ek = kinetic_energy(mass, vel)
        ep = potential_energy(mass, want_pot_arr)
        L = angular_momentum(mass, pos, vel)
        P = linear_momentum(mass, vel)
        c1 = _galaxy_centre(pos, bulge1)
        c2 = _galaxy_centre(pos, bulge2)
        diag["step"].append(step); diag["t"].append(t)
        diag["ekin"].append(ek); diag["epot"].append(ep); diag["etot"].append(ek + ep)
        diag["lx"].append(L[0]); diag["ly"].append(L[1]); diag["lz"].append(L[2])
        diag["px"].append(P[0]); diag["py"].append(P[1]); diag["pz"].append(P[2])
        diag["sep"].append(float(np.linalg.norm(c1 - c2)))
        for k, v in zip(("c1x", "c1y", "c1z"), c1):
            diag[k].append(float(v))
        for k, v in zip(("c2x", "c2y", "c2z"), c2):
            diag[k].append(float(v))

    def snapshot(idx, t):
        np.savez_compressed(
            out / f"snap_{idx:04d}.npz",
            t=np.float32(t), t_gyr=np.float32(t * TIME_UNIT_GYR),
            star_pos=pos[star_idx].astype(np.float32),
            star_vel=vel[star_idx].astype(np.float32),
            halo_pos=pos[halo_sub].astype(np.float32),
        )
        snaps.append(idx)

    record(0, 0.0, pot)
    if store_snapshots:
        snapshot(0, 0.0)
    isnap = 1
    if probe_every is None:
        probe_every = max(1, n_steps // 40)
    if probe is not None:
        probe(0, 0.0, pos, vel)

    iterator = range(1, n_steps + 1)
    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc=cfg.label, unit="step",
                            mininterval=5.0, smoothing=0.05)
        except ImportError:
            pass

    t0_wall = time.perf_counter()
    for step in iterator:
        ts = time.perf_counter()
        vel += half * acc
        pos += cfg.dt * vel
        want_pot = (step % cfg.diag_every == 0) or (step == n_steps)
        acc, pot = gravity(pos, mass, eps2, want_potential=want_pot)
        vel += half * acc
        step_times.append(time.perf_counter() - ts)
        t = step * cfg.dt
        if want_pot:
            record(step, t, pot)
        if store_snapshots and step % snap_every == 0:
            snapshot(isnap, t)
            isnap += 1
        if probe is not None and (step % probe_every == 0 or step == n_steps):
            probe(step, t, pos, vel)
    wall = time.perf_counter() - t0_wall

    diag = {k: np.asarray(v) for k, v in diag.items()}
    st = np.asarray(step_times)
    meta = {
        "n_particles": int(n),
        "n_steps": int(n_steps),
        "dt_code": float(cfg.dt),
        "dt_myr": float(cfg.dt * TIME_UNIT_GYR * 1000.0),
        "t_end_gyr": float(cfg.t_end_gyr),
        "theta": float(cfg.theta),
        "leaf_size": int(cfg.leaf_size),
        "wall_clock_s": float(wall),
        "wall_per_step_s": float(st.mean()),
        "wall_per_step_median_s": float(np.median(st)),
        "wall_per_step_per_particle_us": float(st.mean() / n * 1e6),
        "n_snapshots": len(snaps),
        "softening_kpc": {"disk": float(p.eps[p.ptype == 0][0]) if (p.ptype == 0).any() else None,
                          "bulge": float(p.eps[p.ptype == 1][0]) if (p.ptype == 1).any() else None,
                          "halo": float(p.eps[p.ptype == 2][0]) if (p.ptype == 2).any() else None},
        "dt_criterion_p01_code": float(np.percentile(dt_req, 1)),
        "dt_criterion_median_code": float(np.median(dt_req)),
        "dt_criterion_p01_myr": float(np.percentile(dt_req, 1) * TIME_UNIT_GYR * 1000.0),
        "dt_criterion_median_myr": float(np.median(dt_req) * TIME_UNIT_GYR * 1000.0),
    }
    if store_snapshots:
        np.savez_compressed(out / "diagnostics.npz", **diag,
                            step_times=st, meta=json.dumps(meta))
    return diag, meta


def energy_error(diag):
    """max |Delta E| / |E_0| and whether the drift is secular or bounded."""
    e = diag["etot"]
    e0 = e[0]
    rel = np.abs(e - e0) / abs(e0)
    # linear trend of the *relative* error over time -> secular drift rate
    t = diag["t"]
    if t.size > 2:
        slope = np.polyfit(t, (e - e0) / abs(e0), 1)[0]
    else:
        slope = 0.0
    return {
        "e0": float(e0),
        "max_rel_error": float(rel.max()),
        "final_rel_error": float(rel[-1]),
        "rms_rel_error": float(np.sqrt(np.mean(((e - e0) / abs(e0)) ** 2))),
        "secular_drift_per_gyr": float(slope / TIME_UNIT_GYR),
    }


def angmom_error(diag):
    L = np.stack([diag["lx"], diag["ly"], diag["lz"]], axis=1)
    L0 = L[0]
    n0 = np.linalg.norm(L0)
    dl = np.linalg.norm(L - L0, axis=1) / n0
    return {
        "l0": float(n0),
        "l0_vec": [float(x) for x in L0],
        "max_rel_error": float(dl.max()),
        "final_rel_error": float(dl[-1]),
    }
