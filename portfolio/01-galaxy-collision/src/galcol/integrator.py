"""Kick-drift-kick leapfrog and the conserved-quantity diagnostics.

KDK leapfrog with a fixed timestep is symplectic and time-reversible, so the
energy error is *bounded and oscillatory* rather than secularly growing: the
integrator exactly conserves a "shadow" Hamiltonian that differs from the true
one by O(dt^2).  Demonstrating that bounded behaviour -- not merely a small
number -- is the point of the energy-conservation validation.

One full step from synchronised (x_n, v_n):

    v_{n+1/2} = v_n     + (dt/2) a(x_n)
    x_{n+1}   = x_n     + dt v_{n+1/2}
    v_{n+1}   = v_{n+1/2} + (dt/2) a(x_{n+1})

Consecutive steps share a force evaluation, so the cost is one force
evaluation per step.
"""

from __future__ import annotations

import numpy as np

from .tree import accel_tree, accel_direct, build_tree
from .units import G


# --------------------------------------------------------------------------
# Acceleration providers
# --------------------------------------------------------------------------
class TreeGravity:
    """Barnes-Hut self-gravity, optionally plus a rigid analytic halo."""

    def __init__(self, theta=0.7, leaf_size=12, rigid_halo=None,
                 rigid_centre=(0.0, 0.0, 0.0)):
        self.theta = theta
        self.leaf_size = leaf_size
        self.rigid_halo = rigid_halo
        self.rigid_centre = np.asarray(rigid_centre, dtype=float)
        self.n_evals = 0

    def __call__(self, pos, mass, eps2, want_potential=False):
        acc, pot = accel_tree(pos, mass, eps2, theta=self.theta,
                              leaf_size=self.leaf_size,
                              want_potential=want_potential)
        if self.rigid_halo is not None:
            acc, pot = _add_rigid(self.rigid_halo, self.rigid_centre,
                                  pos, acc, pot, want_potential)
        self.n_evals += 1
        return acc, pot


class DirectGravity:
    """Exact O(N^2) self-gravity; used by the tests and the Kepler check."""

    def __init__(self, rigid_halo=None, rigid_centre=(0.0, 0.0, 0.0)):
        self.rigid_halo = rigid_halo
        self.rigid_centre = np.asarray(rigid_centre, dtype=float)
        self.n_evals = 0

    def __call__(self, pos, mass, eps2, want_potential=False):
        acc, pot = accel_direct(pos, mass, eps2, want_potential=want_potential)
        if self.rigid_halo is not None:
            acc, pot = _add_rigid(self.rigid_halo, self.rigid_centre,
                                  pos, acc, pot, want_potential)
        self.n_evals += 1
        return acc, pot


def _add_rigid(halo, centre, pos, acc, pot, want_potential):
    """Acceleration and potential of a rigid (non-responding) Hernquist halo.

    A rigid halo is cheap and perfectly stable, which makes it useful for
    tests and for isolating disk physics -- but it cannot exert dynamical
    friction, so the PRODUCTION merger must use live halo particles.
    """
    d = pos - centre
    r = np.sqrt((d * d).sum(axis=1))
    rs = np.maximum(r, 1e-8)
    amag = G * halo.mass / (rs + halo.a) ** 2
    acc = acc - (amag / rs)[:, None] * d
    if want_potential and pot is not None:
        pot = pot - G * halo.mass / (rs + halo.a)
    return acc, pot


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------
def kinetic_energy(mass, vel):
    return 0.5 * float(np.sum(mass * (vel * vel).sum(axis=1)))


def potential_energy(mass, pot):
    """Total potential energy from the per-particle specific potential.

    ``pot`` already excludes the self-term, so each pair is counted twice and
    the factor 1/2 removes the double counting.
    """
    return 0.5 * float(np.sum(mass * pot))


def angular_momentum(mass, pos, vel):
    return (mass[:, None] * np.cross(pos, vel)).sum(axis=0)


def linear_momentum(mass, vel):
    return (mass[:, None] * vel).sum(axis=0)


# --------------------------------------------------------------------------
# Leapfrog
# --------------------------------------------------------------------------
def leapfrog_kdk(pos, vel, mass, eps2, gravity, dt, n_steps,
                 callback=None, acc=None):
    """Advance ``n_steps`` KDK leapfrog steps in place.

    ``callback(step, t, pos, vel, acc, pot)`` is invoked after each completed
    step with synchronised positions and velocities.  ``pot`` is the specific
    potential at the *new* positions when the callback asked for it via the
    ``wants_potential`` attribute, else ``None``.
    """
    want_pot = bool(getattr(callback, "wants_potential", False))
    if acc is None:
        acc, pot = gravity(pos, mass, eps2, want_potential=want_pot)
    else:
        pot = None
    t = 0.0
    if callback is not None:
        callback(0, t, pos, vel, acc, pot)
    half = 0.5 * dt
    for step in range(1, n_steps + 1):
        vel += half * acc
        pos += dt * vel
        acc, pot = gravity(pos, mass, eps2, want_potential=want_pot)
        vel += half * acc
        t = step * dt
        if callback is not None:
            callback(step, t, pos, vel, acc, pot)
    return pos, vel, acc


def timestep_criterion(acc, eps, eta=0.025):
    """Gadget-style acceleration timestep, dt_i = sqrt(2 eta eps_i / |a_i|).

    Returned as the array of per-particle limits; the production run uses a
    single global fixed dt and this function is used to *report* how that
    choice compares with the per-particle requirement.
    """
    amag = np.sqrt((acc * acc).sum(axis=1))
    return np.sqrt(2.0 * eta * eps / np.maximum(amag, 1e-30))
