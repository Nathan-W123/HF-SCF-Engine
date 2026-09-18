"""The single source of truth for the production run's parameters.

Every entry point -- run_all.py, the validation scripts, the figure scripts
and the renderer -- imports the configuration from here, so there is exactly
one definition of "the production run".
"""

from __future__ import annotations

from pathlib import Path

from .ics import Encounter, Resolution
from .profiles import default_galaxy
from .simulate import SimConfig

ROOT = Path(__file__).resolve().parents[2]

SEED = 20260918

# ---- resolution ----------------------------------------------------------
# 36,000 particles per galaxy, 72,000 in total.  The split is deliberately
# star-heavy: the halo needs enough particles to produce converged dynamical
# friction, but the disk is what is measured and what is rendered.
#
# Softening is per-component.  The halo particles are ~9x more massive than
# the disk particles, so they get a ~3x larger softening to keep their
# two-body heating of the thin disk under control; the isolated-galaxy
# validation is what checks that this actually worked.
RESOLUTION = Resolution(
    n_disk=15000, n_bulge=4000, n_halo=17000,
    eps_disk=0.20, eps_bulge=0.30, eps_halo=0.70,
)

# ---- encounter geometry --------------------------------------------------
# Prograde encounter in the Toomre & Toomre (1972) style.  The pericentre of
# 12 kpc is 4 disk scale lengths, so the disks (truncated at 15 kpc) very
# nearly graze each other at closest approach.  Galaxy 1 is almost coplanar
# with the orbit and therefore develops the long classic tail; galaxy 2 is
# inclined by 60 degrees and develops a shorter, warped one.  The orbit is
# moderately eccentric rather than exactly parabolic: e = 0.85 is what makes
# the pair come back and merge inside the integrated time (see README).
ENCOUNTER = Encounter(
    r_peri=12.0, ecc=0.85, r_start=90.0,
    inc1=10.0, node1=0.0, inc2=60.0, node2=30.0,
)

# ---- integration ---------------------------------------------------------
SIM = SimConfig(
    dt=0.0015,              # 1.467 Myr
    t_end_gyr=2.2,
    theta=0.7,
    leaf_size=12,
    snap_dt_gyr=0.008,
    diag_every=2,
    out_dir=str(ROOT / "data" / "collision"),
    label="collision",
)

MODEL = default_galaxy()

# ---- reduced settings used by `make quick` and the test suite -------------
QUICK_RESOLUTION = Resolution(n_disk=2500, n_bulge=700, n_halo=2800,
                              eps_disk=0.4, eps_bulge=0.45, eps_halo=1.0)
QUICK_SIM = SimConfig(dt=0.003, t_end_gyr=1.4, theta=0.8, leaf_size=12,
                      snap_dt_gyr=0.02, diag_every=4,
                      out_dir=str(ROOT / "data" / "collision"),
                      label="collision-quick")


def build_initial_conditions(quick=False):
    from .ics import make_collision
    res = QUICK_RESOLUTION if quick else RESOLUTION
    return make_collision(MODEL, res, ENCOUNTER, seed=SEED, live_halo=True)
