"""The sampled galaxy must reproduce the analytic model it came from."""
import numpy as np
import pytest

from galcol.ics import (Encounter, Resolution, kepler_state, make_collision,
                        make_galaxy, make_isolated, orient, rotation_matrix,
                        sample_disk)
from galcol.profiles import default_galaxy
from galcol.units import G, PTYPE_BULGE, PTYPE_DISK, PTYPE_HALO

SMOKE = Resolution(n_disk=4000, n_bulge=1200, n_halo=4000)


def test_disk_radii_follow_the_exponential_profile():
    m = default_galaxy()
    d = sample_disk(m, 60000, np.random.default_rng(0))
    R = np.hypot(d.pos[:, 0], d.pos[:, 1])
    assert R.max() <= m.disk.r_trunc
    norm = m.disk.mass_enclosed_cyl(m.disk.r_trunc) / m.disk.mass
    for rq in (1.5, 3.0, 6.0, 10.0):
        want = m.disk.mass_enclosed_cyl(rq) / m.disk.mass / norm
        assert (R < rq).mean() == pytest.approx(want, abs=0.008)


def test_disk_vertical_profile_is_sech_squared():
    m = default_galaxy()
    d = sample_disk(m, 60000, np.random.default_rng(1))
    z = d.pos[:, 2]
    zd = m.disk.z_scale
    # CDF of sech^2(z/zd) is 0.5 (1 + tanh(z/zd))
    for zq in (0.15, 0.3, 0.6):
        assert (np.abs(z) < zq).mean() == pytest.approx(np.tanh(zq / zd),
                                                        abs=0.01)
    assert np.median(np.abs(z)) == pytest.approx(zd * np.arctanh(0.5), rel=0.05)


def test_component_masses_are_correct():
    m = default_galaxy()
    p = make_isolated(m, SMOKE)
    for pt, want in ((PTYPE_DISK, m.disk.mass),
                     (PTYPE_BULGE, m.bulge.mass_enclosed(m.bulge.r_trunc)),
                     (PTYPE_HALO, m.halo.mass_enclosed(m.halo.r_trunc))):
        assert p.mass[p.ptype == pt].sum() == pytest.approx(want, rel=1e-10)


def test_galaxy_has_zero_net_momentum_and_is_centred():
    m = default_galaxy()
    p = make_isolated(m, SMOKE)
    assert np.max(np.abs((p.mass[:, None] * p.vel).sum(0))) < 1e-9
    # positions are NOT re-centred on the noisy total COM, but the thin disk
    # must still sit on z = 0
    d = p.ptype == PTYPE_DISK
    assert abs(np.median(p.pos[d, 2])) < 0.03


def test_disk_rotates_near_the_circular_speed_with_asymmetric_drift():
    m = default_galaxy()
    p = make_isolated(m, Resolution(n_disk=20000, n_bulge=2000, n_halo=6000))
    d = p.ptype == PTYPE_DISK
    R = np.hypot(p.pos[d, 0], p.pos[d, 1])
    vphi = (p.pos[d, 0] * p.vel[d, 1] - p.pos[d, 1] * p.vel[d, 0]) / np.maximum(R, 1e-9)
    sel = (R > 6.0) & (R < 10.0)
    vc = float(m.v_circ(8.0)[0])
    mean_vphi = vphi[sel].mean()
    assert 0.85 * vc < mean_vphi < vc          # below v_c: asymmetric drift
    assert vphi[sel].std() < 0.25 * vc         # cold disk


def test_spheroid_dispersions_are_bounded_by_escape_speed():
    m = default_galaxy()
    p = make_isolated(m, SMOKE)
    for pt in (PTYPE_BULGE, PTYPE_HALO):
        s = p.ptype == pt
        assert np.all(np.isfinite(p.vel[s]))
        assert np.linalg.norm(p.vel[s], axis=1).max() < 900.0


def test_kepler_state_reproduces_its_orbital_elements():
    mu = G * 90.0
    r_peri, ecc, r0 = 12.0, 0.85, 90.0
    r, v = kepler_state(90.0, r_peri, ecc, r0)
    assert np.linalg.norm(r) == pytest.approx(r0, rel=1e-12)
    a = r_peri / (1 - ecc)
    energy = 0.5 * v @ v - mu / np.linalg.norm(r)
    assert energy == pytest.approx(-mu / (2 * a), rel=1e-10)
    h = np.cross(r, v)
    p = (h @ h) / mu
    assert p == pytest.approx(a * (1 - ecc**2), rel=1e-10)
    assert h[2] > 0                                   # prograde about +z
    assert r @ v < 0                                  # inbound


def test_rotation_matrix_is_orthonormal_and_tilts_the_spin():
    M = rotation_matrix(60.0, 30.0)
    assert np.allclose(M @ M.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(M) == pytest.approx(1.0, rel=1e-12)
    assert M[:, 2] @ np.array([0, 0, 1.0]) == pytest.approx(np.cos(np.radians(60.0)),
                                                           rel=1e-12)


def test_orient_changes_the_disk_angular_momentum_direction():
    m = default_galaxy()
    p = make_galaxy(m, SMOKE, np.random.default_rng(2))
    d = p.ptype == PTYPE_DISK
    L0 = np.cross(p.pos[d], p.vel[d]).sum(0)
    L0 /= np.linalg.norm(L0)
    orient(p, 60.0, 0.0)
    L1 = np.cross(p.pos[d], p.vel[d]).sum(0)
    L1 /= np.linalg.norm(L1)
    assert L0 @ L1 == pytest.approx(np.cos(np.radians(60.0)), abs=0.02)


def test_collision_setup_has_the_requested_separation():
    m = default_galaxy()
    enc = Encounter(r_peri=12.0, ecc=0.85, r_start=90.0)
    p = make_collision(m, SMOKE, enc)
    c = []
    for g in (0, 1):
        s = (p.gid == g) & (p.ptype == PTYPE_BULGE)
        c.append(np.median(p.pos[s], axis=0))
    assert np.linalg.norm(c[0] - c[1]) == pytest.approx(90.0, rel=0.02)
    assert np.max(np.abs((p.mass[:, None] * p.vel).sum(0))) < 1e-9
    assert p.n == 2 * (SMOKE.n_disk + SMOKE.n_bulge + SMOKE.n_halo)


def test_rigid_halo_option_omits_halo_particles():
    m = default_galaxy()
    p = make_isolated(m, SMOKE, live_halo=False)
    assert (p.ptype == PTYPE_HALO).sum() == 0
    assert p.n == SMOKE.n_disk + SMOKE.n_bulge
