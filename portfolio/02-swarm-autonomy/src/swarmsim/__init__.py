"""swarmsim -- decentralised fixed-wing swarm simulation.

Modules
-------
config      parameter dataclasses and the no-fly-zone primitive
dynamics    3-DOF coordinated-turn fixed-wing model + RK4 integrator
wind        steady wind and MIL-F-8785C Dryden turbulence
planner     tangent-visibility-graph + A* global path planning
guidance    L1 nonlinear guidance, altitude and airspeed loops
qp          small dense QP (Hildreth) used by the avoidance layer
avoidance   3-D ORCA + velocity control-barrier-function safety filter
sim         the simulation engine
scenarios   scenario definitions
metrics     scenario metrics
render      custom perspective renderer for the hero visuals
"""

__version__ = "1.0.0"

from . import (avoidance, config, dynamics, guidance, metrics, planner, qp,
               scenarios, sim, wind)

__all__ = ["avoidance", "config", "dynamics", "guidance", "metrics",
           "planner", "qp", "scenarios", "sim", "wind"]
