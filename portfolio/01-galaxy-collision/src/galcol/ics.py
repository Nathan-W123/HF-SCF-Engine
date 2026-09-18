"""Initial conditions: equilibrium disk galaxies and the encounter orbit.

The recipe follows Hernquist (1993), "N-body realizations of compound
galaxies":

*Positions* are drawn from the analytic density profiles by exact inverse-CDF
sampling (no rejection, no tabulated inversion):

  - exponential disk: the radial PDF p(R) dR ~ R exp(-R/Rd) dR is a Gamma(2,Rd)
    distribution, so R = -Rd (ln u1 + ln u2);
  - sech^2 vertical profile: z = zd artanh(2u - 1);
  - Hernquist sphere: r = a sqrt(u) / (1 - sqrt(u)).

*Disk velocities* use the circular speed of the FULL composite potential
(Freeman's exact razor-thin exponential-disk term plus both Hernquist terms),
a radial dispersion set by a Toomre Q criterion, a vertical dispersion from
isothermal-sheet equilibrium, an azimuthal dispersion from the epicyclic
relation, and the asymmetric-drift correction

    <v_phi>^2 = v_c^2 + sigma_R^2 [1 - kappa^2/(4 Omega^2) - 2R/Rd]

where the last term is d ln(Sigma sigma_R^2)/d ln R for the adopted
exponential Sigma and sigma_R.

*Bulge and halo velocities* are isotropic Maxwellians whose dispersion comes
from integrating the spherical Jeans equation

    sigma_r^2(r) = (1/rho) Int_r^inf rho(r') G M_tot(r') / r'^2 dr'

in the spherically-averaged TOTAL potential (the disk enters through its
cylindrical enclosed mass, which is an approximation -- see README
limitations).  Equilibrium of the result is verified, not assumed, by the
isolated-galaxy validation run.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .profiles import GalaxyModel, default_galaxy
from .units import G, PTYPE_DISK, PTYPE_BULGE, PTYPE_HALO


# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Resolution:
    n_disk: int = 15000
    n_bulge: int = 4000
    n_halo: int = 17000
    eps_disk: float = 0.25    # kpc, Plummer softening
    eps_bulge: float = 0.30
    eps_halo: float = 0.70


PRODUCTION = Resolution()
SMOKE = Resolution(n_disk=1200, n_bulge=400, n_halo=1600)


class ParticleSet:
    """Positions, velocities, masses, softenings and type tags."""

    def __init__(self, pos, vel, mass, eps, ptype, gid):
        self.pos = np.ascontiguousarray(pos, dtype=np.float64)
        self.vel = np.ascontiguousarray(vel, dtype=np.float64)
        self.mass = np.ascontiguousarray(mass, dtype=np.float64)
        self.eps = np.ascontiguousarray(eps, dtype=np.float64)
        self.ptype = np.ascontiguousarray(ptype, dtype=np.int8)
        self.gid = np.ascontiguousarray(gid, dtype=np.int8)

    @property
    def n(self):
        return self.pos.shape[0]

    @property
    def eps2(self):
        return self.eps ** 2

    def copy(self):
        return ParticleSet(self.pos.copy(), self.vel.copy(), self.mass.copy(),
                           self.eps.copy(), self.ptype.copy(), self.gid.copy())


def concat(sets):
    return ParticleSet(
        np.vstack([s.pos for s in sets]),
        np.vstack([s.vel for s in sets]),
        np.concatenate([s.mass for s in sets]),
        np.concatenate([s.eps for s in sets]),
        np.concatenate([s.ptype for s in sets]),
        np.concatenate([s.gid for s in sets]),
    )


# --------------------------------------------------------------------------
# Spherical Jeans solve for the spheroidal components
# --------------------------------------------------------------------------
def _jeans_sigma(model: GalaxyModel, component, r_query):
    """Isotropic radial dispersion of ``component`` in the total potential."""
    r = np.logspace(-3.0, 4.0, 4000)
    rho = component.density(r)
    m_tot = model.mass_enclosed_sph(r)
    integrand = rho * G * m_tot / r**2
    # cumulative integral from r to infinity (trapezoid, reversed)
    dr = np.diff(r)
    seg = 0.5 * (integrand[1:] + integrand[:-1]) * dr
    tail = np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])
    sigma2 = tail / np.maximum(rho, 1e-300)
    sigma2 = np.maximum(sigma2, 0.0)
    return np.sqrt(np.interp(np.asarray(r_query), r, sigma2))


def _escape_speed(model: GalaxyModel, r):
    """Approximate escape speed: exact for the two Hernquist spheroids, with a
    Plummer-like stand-in for the disk potential."""
    phi = (model.bulge.potential(r) + model.halo.potential(r)
           - G * model.disk.mass / np.sqrt(np.asarray(r) ** 2 + model.disk.r_scale ** 2))
    return np.sqrt(2.0 * np.abs(phi))


def _cap_speed(vel, vmax):
    sp = np.linalg.norm(vel, axis=1)
    over = sp > vmax
    if np.any(over):
        vel[over] *= (vmax[over] / sp[over])[:, None]
    return vel


# --------------------------------------------------------------------------
# Component samplers
# --------------------------------------------------------------------------
def sample_disk(model: GalaxyModel, n, rng, toomre_q=1.5, eps=0.25, gid=0):
    d = model.disk
    # Gamma(2, Rd) radii, resampled until inside the truncation radius
    R = np.empty(0)
    while R.size < n:
        k = int((n - R.size) * 1.4) + 32
        cand = -d.r_scale * (np.log(rng.random(k)) + np.log(rng.random(k)))
        R = np.concatenate([R, cand[cand < d.r_trunc]])
    R = R[:n]
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    u = rng.uniform(-1.0 + 1e-9, 1.0 - 1e-9, n)
    z = d.z_scale * np.arctanh(u)
    z = np.clip(z, -6.0 * d.z_scale, 6.0 * d.z_scale)
    pos = np.stack([R * np.cos(phi), R * np.sin(phi), z], axis=1)

    # --- dispersions -----------------------------------------------------
    omega, kappa = model.omega_kappa(R)
    sigma = d.sigma(R)
    # sigma_R normalised so that Q(2.4 Rd) = toomre_q, with the standard
    # exp(-R / 2Rd) radial profile.
    r_ref = 2.4 * d.r_scale
    om_r, ka_r = model.omega_kappa(np.array([r_ref]))
    sig_ref = toomre_q * 3.36 * G * d.sigma(r_ref) / ka_r[0]
    sigma_R0 = sig_ref / np.exp(-r_ref / (2.0 * d.r_scale))
    sigma_R = np.maximum(sigma_R0 * np.exp(-R / (2.0 * d.r_scale)), 4.0)
    sigma_z = np.maximum(np.sqrt(np.pi * G * sigma * d.z_scale), 4.0)
    ratio = kappa**2 / (4.0 * omega**2)
    sigma_phi = np.maximum(sigma_R * np.sqrt(ratio), 3.0)

    vc2 = model.v_circ_sq(R)
    drift = sigma_R**2 * (1.0 - ratio - 2.0 * R / d.r_scale)
    vphi_mean = np.sqrt(np.maximum(vc2 + drift, 0.04 * vc2))

    vR = rng.normal(0.0, 1.0, n) * sigma_R
    vph = vphi_mean + rng.normal(0.0, 1.0, n) * sigma_phi
    vz = rng.normal(0.0, 1.0, n) * sigma_z
    c, s = np.cos(phi), np.sin(phi)
    vel = np.stack([vR * c - vph * s, vR * s + vph * c, vz], axis=1)
    vel = _cap_speed(vel, 0.95 * _escape_speed(model, np.sqrt(R**2 + z**2)))

    mass = np.full(n, d.mass / n)
    return ParticleSet(pos, vel, mass, np.full(n, eps),
                       np.full(n, PTYPE_DISK), np.full(n, gid))


def _sample_sphere(rng, n):
    ct = rng.uniform(-1.0, 1.0, n)
    st = np.sqrt(np.maximum(1.0 - ct**2, 0.0))
    ph = rng.uniform(0.0, 2.0 * np.pi, n)
    return np.stack([st * np.cos(ph), st * np.sin(ph), ct], axis=1)


def sample_spheroid(model: GalaxyModel, component, n, rng, ptype, eps, gid=0):
    r = component.sample_radius(rng, n)
    pos = _sample_sphere(rng, n) * r[:, None]
    sig = _jeans_sigma(model, component, r)
    vel = rng.normal(0.0, 1.0, (n, 3)) * sig[:, None]
    vel = _cap_speed(vel, 0.95 * _escape_speed(model, r))
    m_tot = component.mass_enclosed(component.r_trunc)
    mass = np.full(n, m_tot / n)
    return ParticleSet(pos, vel, mass, np.full(n, eps),
                       np.full(n, ptype), np.full(n, gid))


def make_galaxy(model: GalaxyModel, res: Resolution, rng, gid=0,
                live_halo=True, toomre_q=1.5):
    """Sample one equilibrium galaxy centred on the origin, at rest."""
    parts = [sample_disk(model, res.n_disk, rng, toomre_q=toomre_q,
                         eps=res.eps_disk, gid=gid)]
    if res.n_bulge > 0:
        parts.append(sample_spheroid(model, model.bulge, res.n_bulge, rng,
                                     PTYPE_BULGE, res.eps_bulge, gid=gid))
    if live_halo and res.n_halo > 0:
        parts.append(sample_spheroid(model, model.halo, res.n_halo, rng,
                                     PTYPE_HALO, res.eps_halo, gid=gid))
    p = concat(parts)
    # Zero the net momentum so the galaxy does not drift out of frame.
    #
    # Deliberately do NOT re-centre the positions on the sampled centre of
    # mass: the halo carries ~89% of the mass but is sampled with only
    # n_halo particles, so its Poisson centroid noise is ~sigma_r/sqrt(N),
    # several tenths of a kpc.  Subtracting it would displace the (thin) disk
    # by a sizeable fraction of its own scale height.  Every component is
    # sampled symmetrically about the origin, so the origin already IS the
    # dynamical centre.
    p.vel -= (p.mass[:, None] * p.vel).sum(0) / p.mass.sum()
    return p


# --------------------------------------------------------------------------
# Orientation and the encounter orbit
# --------------------------------------------------------------------------
def rotation_matrix(inclination_deg, node_deg=0.0):
    """Rotate a disk whose spin is +z by ``inclination`` about the x-axis,
    then by ``node`` about the z-axis."""
    i = np.radians(inclination_deg)
    w = np.radians(node_deg)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(i), -np.sin(i)],
                   [0, np.sin(i), np.cos(i)]])
    Rz = np.array([[np.cos(w), -np.sin(w), 0],
                   [np.sin(w), np.cos(w), 0],
                   [0, 0, 1]])
    return Rz @ Rx


def orient(p: ParticleSet, inclination_deg, node_deg=0.0):
    M = rotation_matrix(inclination_deg, node_deg)
    p.pos = np.ascontiguousarray(p.pos @ M.T)
    p.vel = np.ascontiguousarray(p.vel @ M.T)
    return p


def kepler_state(m_total, r_peri, ecc, r_start):
    """Relative position/velocity on a Kepler orbit of pericentre ``r_peri``
    and eccentricity ``ecc``, at separation ``r_start`` on the INBOUND branch.

    Orbit lies in the x-y plane with angular momentum along +z, so a disk with
    spin +z is prograde.  Returns (r_vec, v_vec).
    """
    mu = G * m_total
    if abs(ecc - 1.0) < 1e-9:
        p = 2.0 * r_peri
    else:
        a = r_peri / (1.0 - ecc)
        p = a * (1.0 - ecc**2)
    if r_start < r_peri:
        raise ValueError("r_start must exceed the pericentre distance")
    cos_nu = np.clip((p / r_start - 1.0) / max(ecc, 1e-12), -1.0, 1.0)
    nu = -np.arccos(cos_nu)                      # negative -> inbound
    r_vec = np.array([r_start * np.cos(nu), r_start * np.sin(nu), 0.0])
    k = np.sqrt(mu / p)
    v_vec = np.array([-k * np.sin(nu), k * (ecc + np.cos(nu)), 0.0])
    return r_vec, v_vec


@dataclass(frozen=True)
class Encounter:
    r_peri: float = 12.0         # kpc = 4 disk scale lengths
    ecc: float = 0.85
    r_start: float = 90.0        # kpc, inbound branch
    inc1: float = 10.0           # deg, near-coplanar prograde -> long tail
    node1: float = 0.0
    inc2: float = 60.0           # deg, inclined prograde
    node2: float = 30.0


def make_collision(model: GalaxyModel, res: Resolution, enc: Encounter = Encounter(),
                   seed=20260918, live_halo=True):
    """Two identical galaxies on the specified encounter orbit."""
    rng1 = np.random.default_rng(seed)
    rng2 = np.random.default_rng(seed + 777)
    g1 = make_galaxy(model, res, rng1, gid=0, live_halo=live_halo)
    g2 = make_galaxy(model, res, rng2, gid=1, live_halo=live_halo)
    orient(g1, enc.inc1, enc.node1)
    orient(g2, enc.inc2, enc.node2)

    m1 = g1.mass.sum()
    m2 = g2.mass.sum()
    m_tot = m1 + m2
    r_vec, v_vec = kepler_state(m_tot, enc.r_peri, enc.ecc, enc.r_start)

    g1.pos += (-m2 / m_tot) * r_vec
    g1.vel += (-m2 / m_tot) * v_vec
    g2.pos += (m1 / m_tot) * r_vec
    g2.vel += (m1 / m_tot) * v_vec

    p = concat([g1, g2])
    # A rigid translation/boost of the whole system distorts nothing, so the
    # global centre of mass and momentum are safe to zero here.
    p.vel -= (p.mass[:, None] * p.vel).sum(0) / p.mass.sum()
    p.pos -= (p.mass[:, None] * p.pos).sum(0) / p.mass.sum()
    return p


def make_isolated(model: GalaxyModel, res: Resolution, seed=20260918,
                  live_halo=True):
    return make_galaxy(model, res, np.random.default_rng(seed), gid=0,
                       live_halo=live_halo)
