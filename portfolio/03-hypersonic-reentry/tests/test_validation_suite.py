"""Run the validation scripts themselves as part of the automated test suite.

These are the same four numbered validations that ``make all`` writes to
``validation/``; running them here means a regression in the physics fails
``pytest`` too, not only the full reproduction.
"""

import pytest

import val_01_orbit_energy
import val_02_allen_eggers
import val_03_atmosphere_table
import val_04_energy_convergence


@pytest.mark.parametrize(
    "module",
    [val_01_orbit_energy, val_02_allen_eggers, val_03_atmosphere_table,
     val_04_energy_convergence],
    ids=["orbit_energy", "allen_eggers", "atmosphere_table", "energy_convergence"],
)
def test_validation_module_passes(module):
    report = module.run()
    failed = [c["name"] for c in report["checks"] if not c["passed"]]
    assert not failed, f"{report['name']} failed checks: {failed}"
    assert report["passed"]
    assert len(report["checks"]) >= 3
