"""Path bootstrap + shared reporting helpers for the validation scripts.

Every validation script is directly runnable (``python3 validation/val_xx.py``)
and also importable by ``validation/run_all_validation.py`` and by the test
suite.  Each exposes ``run() -> dict`` with the schema::

    {"name": str, "description": str, "passed": bool,
     "checks": [{"name", "value", "threshold", "comparison", "units", "passed"}],
     "data": {...}}
"""

from __future__ import annotations

import json
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

VALIDATION_DIR = PROJECT_ROOT / "validation"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
MEDIA_DIR = PROJECT_ROOT / "media"


def check(name, value, threshold, comparison="<", units="", note=""):
    """Build one pass/fail record."""
    v = float(value)
    t = float(threshold)
    passed = v < t if comparison == "<" else (v > t if comparison == ">" else v <= t)
    return {
        "name": name,
        "value": v,
        "threshold": t,
        "comparison": comparison,
        "units": units,
        "note": note,
        "passed": bool(passed),
    }


def finish(name, description, checks, data=None):
    return {
        "name": name,
        "description": description,
        "passed": bool(all(c["passed"] for c in checks)),
        "checks": checks,
        "data": data or {},
    }


def format_report(report) -> str:
    lines = []
    lines.append("=" * 88)
    lines.append(f"{report['name']}  --  {'PASS' if report['passed'] else 'FAIL'}")
    lines.append(report["description"].strip())
    lines.append("-" * 88)
    lines.append(f"{'check':<52}{'value':>14}{'':>3}{'threshold':>13}  ")
    for c in report["checks"]:
        flag = "ok " if c["passed"] else "FAIL"
        lines.append(
            f"{c['name']:<52}{c['value']:>14.6g}{'':>3}{c['comparison']:>2}"
            f"{c['threshold']:>11.4g}  {c['units']:<10} {flag}"
        )
    lines.append("=" * 88)
    return "\n".join(lines)


def write_report(report, stem: str) -> None:
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    (VALIDATION_DIR / f"{stem}.json").write_text(json.dumps(report, indent=2))
    (VALIDATION_DIR / f"{stem}.txt").write_text(format_report(report) + "\n")


def main(run_fn, stem: str) -> int:
    report = run_fn()
    write_report(report, stem)
    print(format_report(report))
    return 0 if report["passed"] else 1
