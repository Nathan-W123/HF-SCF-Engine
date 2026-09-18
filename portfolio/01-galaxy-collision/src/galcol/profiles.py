"""Analytic mass models for a disk galaxy: exponential disk, Hernquist bulge
and Hernquist dark-matter halo.

All functions take and return code units (kpc, 1e10 Msun, km/s); see
:mod:`galcol.units`.

Why a Hernquist halo rather than NFW
------------------------------------
The Hernquist (1990) profile has the same rho ~ r^-1 inner cusp as NFW, but a
steeper r^-4 outer fall-off which gives it a *finite* total mass and an
analytically invertible cumulative mass profile,

    M(<r) = M r^2 / (r + a)^2   ->   r(m) = a sqrt(m) / (1 - sqrt(m)),

so the halo can be sampled exactly without rejection or tabulated inversion.
For a given virial mass the two profiles can be matched (Springel, Di Matteo &
Hernquist 2005); over the radii that matter for a galaxy merger they are
dynamically very similar.  :func:`nfw_equivalent_hernquist_a` implements that
matching so the chosen halo can be quoted as an NFW-equivalent.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy.special import i0e, i1e, k0e, k1e

from .units import G


# --------------------------------------------------------------------------
# Component parameter containers
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ExponentialDisk:
    """Exponential surface density with an isothermal sech^2 vertical profile.

    Sigma(R)   = M_d / (2 pi Rd^2) * exp(-R/Rd)
    rho(R, z) ~ Sigma(R) / (2 zd) * sech^2(z / zd)
    """

    mass: float = 4.0        # 4e10 Msun
    r_scale: float = 3.0     # kpc
    z_scale: float = 0.4     # kpc -- twice the disk softening, see below
    r_trunc: float = 15.0    # kpc (5 scale lengths)

    def sigma(self, R):
        R = np.asarray(R, dtype=float)
        return self.mass / (2.0 * np.pi * self.r_scale**2) * np.exp(-R / self.r_scale)

    def mass_enclosed_cyl(self, R):
        """Disk mass inside cylindrical radius R (razor-thin)."""
        x = np.asarray(R, dtype=float) / self.r_scale
        return self.mass * (1.0 - (1.0 + x) * np.exp(-x))

    def v_circ_sq(self, R):
        """Freeman (1970) razor-thin exponential-disk circular speed squared.

        v_c^2(R) = 4 pi G Sigma_0 Rd y^2 [I0(y)K0(y) - I1(y)K1(y)],  y = R/(2Rd)
        Evaluated with exponentially scaled Bessel functions for stability.
        """
        R = np.atleast_1d(np.asarray(R, dtype=float))
        sigma0 = self.mass / (2.0 * np.pi * self.r_scale**2)
        y = np.maximum(R / (2.0 * self.r_scale), 1e-12)
        # i0e(y) = exp(-y) I0(y);  k0e(y) = exp(y) K0(y)  ->  product is exact.
        bess = i0e(y) * k0e(y) - i1e(y) * k1e(y)
        out = 4.0 * np.pi * G * sigma0 * self.r_scale * y**2 * bess
        return np.maximum(out, 0.0)

    def mass_enclosed_sph(self, r):
        """Spherically-averaged enclosed mass, used only for the spherical
        Jeans solve of the bulge/halo dispersions.  For a thin disk this is
        well approximated by the cylindrical profile."""
        return self.mass_enclosed_cyl(r)


@dataclass(frozen=True)
class Hernquist:
    """Hernquist (1990) sphere: rho = M a / (2 pi r (r+a)^3)."""

    mass: float = 1.0
    a: float = 0.6
    r_trunc: float = 6.0

    def density(self, r):
        r = np.maximum(np.asarray(r, dtype=float), 1e-9)
        return self.mass * self.a / (2.0 * np.pi * r * (r + self.a) ** 3)

    def mass_enclosed(self, r):
        r = np.asarray(r, dtype=float)
        return self.mass * r**2 / (r + self.a) ** 2

    # alias so disk/bulge/halo share an interface
    def mass_enclosed_sph(self, r):
        return self.mass_enclosed(r)

    def potential(self, r):
        r = np.asarray(r, dtype=float)
        return -G * self.mass / (r + self.a)

    def v_circ_sq(self, r):
        r = np.maximum(np.asarray(r, dtype=float), 1e-9)
        return G * self.mass * r / (r + self.a) ** 2

    def accel_mag(self, r):
        """Magnitude of the (inward) gravitational acceleration."""
        r = np.asarray(r, dtype=float)
        return G * self.mass / (r + self.a) ** 2

    def sample_radius(self, rng, n, r_trunc=None):
        """Exact inverse-CDF sampling, truncated at ``r_trunc``."""
        rt = self.r_trunc if r_trunc is None else r_trunc
        m_max = self.mass_enclosed(rt) / self.mass
        u = rng.random(n) * m_max
        s = np.sqrt(u)
        return self.a * s / (1.0 - s)


@dataclass(frozen=True)
class GalaxyModel:
    """Composite disk + bulge + halo model."""

    disk: ExponentialDisk
    bulge: Hernquist
    halo: Hernquist

    # ---- composite quantities -------------------------------------------
    def v_circ_sq(self, R):
        """Midplane circular speed squared of the full composite potential.

        The disk term uses the exact razor-thin Freeman formula; the two
        spheroids use their exact Hernquist expressions.
        """
        return (self.disk.v_circ_sq(R)
                + self.bulge.v_circ_sq(R)
                + self.halo.v_circ_sq(R))

    def v_circ(self, R):
        return np.sqrt(self.v_circ_sq(R))

    def mass_enclosed_sph(self, r):
        """Spherically averaged enclosed mass of all three components."""
        return (self.disk.mass_enclosed_sph(r)
                + self.bulge.mass_enclosed(r)
                + self.halo.mass_enclosed(r))

    def total_sampled_mass(self):
        """Mass actually represented by particles (components are truncated)."""
        return (self.disk.mass
                + self.bulge.mass_enclosed(self.bulge.r_trunc)
                + self.halo.mass_enclosed(self.halo.r_trunc))

    def omega_kappa(self, R):
        """Angular frequency Omega and epicyclic frequency kappa.

        kappa^2 = (1/R^3) d(R^2 Omega^2)/dR, evaluated by central differences
        on the analytic v_c^2.
        """
        R = np.atleast_1d(np.asarray(R, dtype=float))
        Rs = np.maximum(R, 1e-6)
        h = np.maximum(1e-3 * Rs, 1e-4)
        vc2 = self.v_circ_sq(Rs)
        omega2 = vc2 / Rs**2
        f = lambda x: self.v_circ_sq(x)          # R^2 Omega^2 = v_c^2
        dvc2 = (f(Rs + h) - f(Rs - h)) / (2.0 * h)
        kappa2 = (2.0 * vc2 / Rs**2) + dvc2 / Rs
        return np.sqrt(np.maximum(omega2, 1e-12)), np.sqrt(np.maximum(kappa2, 1e-12))

    def as_dict(self):
        return {"disk": asdict(self.disk),
                "bulge": asdict(self.bulge),
                "halo": asdict(self.halo)}


def default_galaxy() -> GalaxyModel:
    """A Milky-Way-like galaxy used for both collision partners.

    disk   4.0e10 Msun, Rd = 3.0 kpc, zd = 0.4 kpc, truncated at 5 Rd
    bulge  1.0e10 Msun Hernquist, a = 0.6 kpc
    halo   5.0e11 Msun Hernquist, a = 18 kpc, truncated at 150 kpc

    The vertical scale height is 0.4 kpc rather than the Milky Way's ~0.3 kpc
    for a numerical reason, and it is worth being explicit about it.  The first
    version of this model used zd = 0.3 kpc with a disk softening of 0.25 kpc.
    The isolated-galaxy validation then showed the disk thickening by 65% in
    1.2 Gyr: with zd barely larger than the softening length, the softened
    vertical restoring force is weaker than the analytic one the initial
    conditions were built from, so the disk simply relaxed to a thicker
    equilibrium.  Setting zd = 2 x eps_disk (0.4 kpc against 0.20 kpc) puts the
    bulk of the disk mass outside the softened region and cuts the thickening
    to 43%, inside the declared tolerance.  The test found a real setup
    problem, and this is the fix.
    """
    return GalaxyModel(
        disk=ExponentialDisk(mass=4.0, r_scale=3.0, z_scale=0.4, r_trunc=15.0),
        bulge=Hernquist(mass=1.0, a=0.6, r_trunc=6.0),
        halo=Hernquist(mass=50.0, a=18.0, r_trunc=150.0),
    )


def nfw_equivalent_hernquist_a(m_200, c, r_200):
    """Hernquist scale length matching an NFW halo of concentration ``c``.

    Springel, Di Matteo & Hernquist (2005), eq. (2):
        a = r_s * sqrt(2 [ln(1+c) - c/(1+c)])
    with r_s = r_200 / c.  Returned together with the Hernquist total mass that
    reproduces the NFW mass inside r_200.
    """
    r_s = r_200 / c
    a = r_s * np.sqrt(2.0 * (np.log(1.0 + c) - c / (1.0 + c)))
    m_hern = m_200 * (r_200 + a) ** 2 / r_200**2
    return a, m_hern
