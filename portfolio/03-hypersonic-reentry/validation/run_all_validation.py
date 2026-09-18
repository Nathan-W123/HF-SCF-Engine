#!/usr/bin/env python3
"""Run every validation script, write the individual and combined reports.

Outputs
-------
``validation/val_0X_*.json`` / ``.txt``   per-test machine- and human-readable
``validation/validation_report.json``     all four tests combined
``validation/validation_report.txt``      printable summary table

Exit status is non-zero if any check fails, so ``make all`` stops on a
validation failure rather than quietly producing figures from a broken model.
"""

from __future__ import annotations

import json

from _bootstrap import VALIDATION_DIR, format_report, write_report  # noqa: E402

import val_01_orbit_energy  # noqa: E402
import val_02_allen_eggers  # noqa: E402
import val_03_atmosphere_table  # noqa: E402
import val_04_energy_convergence  # noqa: E402

MODULES = [
    ("val_01_orbit_energy", val_01_orbit_energy),
    ("val_02_allen_eggers", val_02_allen_eggers),
    ("val_03_atmosphere_table", val_03_atmosphere_table),
    ("val_04_energy_convergence", val_04_energy_convergence),
]


def run_all() -> dict:
    reports = []
    for stem, mod in MODULES:
        report = mod.run()
        write_report(report, stem)
        reports.append(report)
        print(format_report(report))
        print()
    n_checks = sum(len(r["checks"]) for r in reports)
    n_failed = sum(1 for r in reports for c in r["checks"] if not c["passed"])
    combined = {
        "all_passed": all(r["passed"] for r in reports),
        "n_reports": len(reports),
        "n_checks": n_checks,
        "n_failed": n_failed,
        "reports": reports,
    }
    (VALIDATION_DIR / "validation_report.json").write_text(json.dumps(combined, indent=2))
    lines = [format_report(r) for r in reports]
    lines.append(
        f"\nOVERALL: {n_checks - n_failed}/{n_checks} checks passed "
        f"({'ALL PASS' if combined['all_passed'] else 'FAILURES PRESENT'})\n"
    )
    (VALIDATION_DIR / "validation_report.txt").write_text("\n\n".join(lines))
    return combined


def main() -> int:
    combined = run_all()
    print(
        f"OVERALL: {combined['n_checks'] - combined['n_failed']}/"
        f"{combined['n_checks']} checks passed"
    )
    return 0 if combined["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
