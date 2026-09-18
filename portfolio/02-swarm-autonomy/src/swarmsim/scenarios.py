"""Scenario definitions.

Four scenarios plus a hero-rendering variant:

``transit``  Nominal mission: 32 vehicles cross a 6 km corridor containing a
             field of seven no-fly cylinders.  The goal line reverses the start
             ordering, so the routes fan across each other as well as around the
             obstacles.  Light steady crosswind, no turbulence.

``swap``     High-density stress test: 36 vehicles on a ring, each assigned the
             antipodal goal, so every trajectory passes through the same centre
             point.  No obstacles -- the conflict is the point.  This is where
             the collision-avoidance invariant is asserted.

``gust``     The transit scenario re-flown in a strong steady wind plus Dryden
             turbulence (sigma_w = 2.6 m/s, moderate-to-severe for this class of
             airframe), to test tracking and deconfliction under disturbance.

``failure``  The swap scenario with one vehicle losing control authority at
             t = 40 s: it locks a 15-degree bank and stops cooperating, so it
             spirals through the swarm while the others must deconflict around
             an agent that does not reciprocate.

``hero``     The ``swap`` stress test with an undecimated state log, used for
             the hero render.  The hero visual is therefore a picture of the
             same run the safety invariant is asserted on.
"""

from __future__ import annotations

from typing import List

import numpy as np

from .config import (AvoidanceParams, DrydenParams, GuidanceParams, NoFlyZone,
                     SimParams, VehicleParams, WindParams)
from .sim import SwarmSpec

CRUISE_BAND = (340.0, 1480.0)    # altitude band used by all scenarios [m]


def _zone_field() -> List[NoFlyZone]:
    """Seven no-fly cylinders spanning the whole operating altitude band."""
    spec = [(0.0, 120.0, 330.0), (900.0, 700.0, 260.0), (-850.0, 620.0, 300.0),
            (520.0, -780.0, 290.0), (-620.0, -820.0, 250.0),
            (1500.0, -250.0, 240.0), (-1550.0, -120.0, 270.0)]
    return [NoFlyZone(x, y, r, z_low=0.0, z_high=2000.0) for x, y, r in spec]


def _alt_ladder(n, seed=3):
    """Stagger cruise altitudes across the band so the swarm is not coplanar."""
    rng = np.random.default_rng(seed)
    lo, hi = CRUISE_BAND
    base = np.linspace(lo, hi, n)
    # A fixed random permutation of the altitude slots.  A structured
    # interleave would be tuning the scenario for or against the avoidance
    # layer; a seeded shuffle is neutral and still reproducible.
    order = rng.permutation(n)
    return np.clip(base[order] + rng.uniform(-12.0, 12.0, n), lo - 20, hi + 20)


# --------------------------------------------------------------------------
def transit(n=32, seed=0, t_max=390.0, wind=None, name="transit",
            zones=None, log_every=2) -> SwarmSpec:
    rng = np.random.default_rng(1000 + seed)
    # Lateral spacing must exceed the required separation, or the goal set
    # itself would be infeasible: 32 vehicles over 5.0 km is 161 m apart.
    y0 = np.linspace(-2500.0, 2500.0, n) + rng.uniform(-45, 45, n)
    y1 = np.linspace(-2500.0, 2500.0, n)[::-1] + rng.uniform(-45, 45, n)
    alt = _alt_ladder(n)
    starts = np.stack([np.full(n, -3000.0) + rng.uniform(-120, 120, n), y0, alt], axis=1)
    goals = np.stack([np.full(n, 3000.0), y1, alt[::-1].copy()], axis=1)
    return SwarmSpec(
        name=name,
        starts=starts,
        goals=goals,
        zones=_zone_field() if zones is None else zones,
        wind=wind if wind is not None else WindParams(mean=(0.0, 3.0, 0.0)),
        sim=SimParams(t_max=t_max, seed=seed, log_every=log_every),
        vehicle=VehicleParams(),
        guidance=GuidanceParams(),
        avoid=AvoidanceParams(),
        clearance=95.0,
    )


def swap(n=36, seed=0, t_max=290.0, wind=None, name="swap",
         fail_id=None, fail_time=1e9, fail_bank=0.0, log_every=2) -> SwarmSpec:
    """Antipodal exchange on a ring -- every path runs through the centre."""
    rng = np.random.default_rng(2000 + seed)
    R = 2100.0
    th = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    th = th + rng.uniform(-0.012, 0.012, n)
    alt = _alt_ladder(n, seed=7 + seed)
    starts = np.stack([R * np.cos(th), R * np.sin(th), alt], axis=1)
    goals = np.stack([-R * np.cos(th), -R * np.sin(th), alt], axis=1)
    return SwarmSpec(
        name=name,
        starts=starts,
        goals=goals,
        zones=[],
        wind=wind if wind is not None else WindParams(),
        sim=SimParams(t_max=t_max, seed=seed, log_every=log_every),
        vehicle=VehicleParams(),
        guidance=GuidanceParams(),
        avoid=AvoidanceParams(),
        clearance=90.0,
        fail_id=fail_id,
        fail_time=fail_time,
        fail_bank=fail_bank,
    )


def gust(n=32, seed=0, t_max=390.0, sigma_scale=2.6, name="gust") -> SwarmSpec:
    w = WindParams(mean=(-7.0, 4.5, 0.0),
                   dryden=DrydenParams().scaled(sigma_scale),
                   turbulence_on=True)
    s = transit(n=n, seed=seed, t_max=t_max, wind=w, name=name)
    return s


def failure(n=36, seed=0, t_max=290.0, name="failure") -> SwarmSpec:
    return swap(n=n, seed=seed, t_max=t_max, name=name,
                fail_id=n // 2, fail_time=40.0,
                fail_bank=np.deg2rad(15.0))


def hero(n=36, seed=0, t_max=290.0) -> SwarmSpec:
    """The hero render: the validated high-density stress test itself.

    This is exactly :func:`swap` -- the 36-vehicle antipodal exchange that
    validation 3 asserts the safety invariant on -- with the state log left
    undecimated so the rendered flight trails are smooth.  Nothing about the
    dynamics, the guidance, the avoidance or the scenario differs, so the hero
    image and video are pictures of the run whose numbers are in the README.

    The moment worth rendering is the conflict-resolution burst around
    t = 80 s: every vehicle's nominal route passes through one common point, so
    the whole swarm arrives there at once, banks hard to open separation, and
    then re-forms on the far side.  Mean bank angle peaks near 15 degrees with a
    95th percentile above 40, and the mean avoidance deflection peaks near
    18 m/s -- which is why the coordination reads in the picture.
    """
    spec = swap(n=n, seed=seed, t_max=t_max, name="hero", log_every=1)
    return spec


ALL = {
    "transit": transit,
    "swap": swap,
    "gust": gust,
    "failure": failure,
    "hero": hero,
}
