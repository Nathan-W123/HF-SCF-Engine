#!/usr/bin/env python3
"""Validation 4 -- deterministic regression.

A fixed-seed scenario must reproduce a stored trajectory signature.  Two
different things are checked, because they fail for different reasons:

1. **Run-to-run determinism inside one process.**  The same spec executed twice
   must give bit-identical states.  This catches accidental use of global RNG
   state, set/dict iteration order, or uninitialised memory.

2. **Regression against a stored reference.**  A decimated trajectory tensor
   (every 20th logged sample, all vehicles, all 7 states) and its SHA-256 are
   stored in ``validation/regression_reference.npz``.  A later run must match
   the stored array to within ``1e-9`` element-wise, and the hash is reported.
   The reference is created on the first run (and the script says so); after
   that it is a hard check.  Delete the file to re-baseline deliberately.

The scenario used is a deterministic 16-vehicle swap with turbulence enabled,
so the check also covers the seeded Dryden filters.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np

from common import HERE, report                              # noqa: E402
from swarmsim import scenarios, sim                           # noqa: E402
from swarmsim.config import DrydenParams, WindParams          # noqa: E402

REF = os.path.join(HERE, "regression_reference.npz")
DECIMATE = 20


def signature_spec():
    w = WindParams(mean=(-5.0, 2.0, 0.0), dryden=DrydenParams().scaled(1.8),
                   turbulence_on=True)
    return scenarios.swap(n=16, seed=20240607, t_max=120.0, wind=w,
                          name="regression")


def signature(result):
    sig = result.states[::DECIMATE].astype(np.float64)
    h = hashlib.sha256(np.ascontiguousarray(sig).tobytes()).hexdigest()
    return sig, h


def run():
    spec = signature_spec()
    r1 = sim.run(spec)
    r2 = sim.run(signature_spec())

    s1, h1 = signature(r1)
    s2, h2 = signature(r2)
    identical = bool(np.array_equal(s1, s2))

    created = False
    if not os.path.exists(REF):
        np.savez_compressed(REF, signature=s1, sha256=np.array(h1))
        created = True
        max_dev = 0.0
        ref_hash = h1
        hash_match = True
    else:
        z = np.load(REF, allow_pickle=False)
        ref = z["signature"]
        ref_hash = str(z["sha256"])
        if ref.shape != s1.shape:
            max_dev = np.inf
            hash_match = False
        else:
            max_dev = float(np.max(np.abs(ref - s1)))
            hash_match = bool(ref_hash == h1)

    checks = {
        "same_process_reruns_bit_identical": identical,
        "matches_stored_reference_within_1e-9": bool(max_dev <= 1e-9),
    }
    payload = {
        "name": "deterministic_regression",
        "description": "Fixed-seed trajectory signature reproducibility.",
        "scenario": "swap(n=16, seed=20240607, t_max=120 s, Dryden on)",
        "signature_shape": list(s1.shape),
        "signature_sha256": h1,
        "reference_sha256": ref_hash,
        "reference_hash_matches": hash_match,
        "reference_created_this_run": created,
        "max_abs_deviation_from_reference": max_dev,
        "decimation": DECIMATE,
        "checks": checks,
        "all_passed": all(checks.values()),
    }
    return payload


if __name__ == "__main__":
    report("validate_regression", run())
