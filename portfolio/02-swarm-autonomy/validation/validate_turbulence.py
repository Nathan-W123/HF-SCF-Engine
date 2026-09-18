#!/usr/bin/env python3
"""Validation 5 -- realised Dryden turbulence vs the analytic MIL-F-8785C PSD.

For each of the three gust components the script generates 48 independent
realisations from the shaping filters, estimates the one-sided power spectral
density with Welch's method, averages the periodograms across realisations, and
compares the result with the analytic one-sided Dryden spectrum

    G(omega) = Phi(omega / V) / V        [ (m/s)^2 / (rad/s) ]

over the band 0.03-30 rad/s.  ``scipy.signal.welch`` returns a density per Hz,
so the comparison divides by ``2 pi``.

Checks
------
* realised RMS within 3 % of the commanded ``sigma`` for each component;
* the band-median of ``G_realised / G_analytic`` within 8 % of 1;
* the RMS of ``log10(G_realised / G_analytic)``, after averaging into 24
  logarithmically spaced bins, below 0.05 decades (~12 %).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import welch

from common import report                                    # noqa: E402
from swarmsim.config import DrydenParams                      # noqa: E402
from swarmsim.wind import (dryden_psd_temporal,               # noqa: E402
                           dryden_series_batch)

V = 25.0
DT = 0.02
N_SAMPLES = 24000
N_REAL = 48
BAND = (0.03, 30.0)
N_BINS = 24


def one_component(comp, dp, rng):
    sigma = {"u": dp.sigma_u, "v": dp.sigma_v, "w": dp.sigma_w}[comp]
    L = {"u": dp.L_u, "v": dp.L_v, "w": dp.L_w}[comp]
    x = dryden_series_batch(dp, V, DT, N_SAMPLES, N_REAL, comp, rng)
    rms = float(x.std())

    f, P = welch(x, fs=1.0 / DT, nperseg=4096, axis=1)
    P = P.mean(axis=0)
    omega = 2.0 * np.pi * f
    G_real = P / (2.0 * np.pi)

    m = (omega >= BAND[0]) & (omega <= BAND[1])
    om = omega[m]
    gr = G_real[m]
    ga = dryden_psd_temporal(om, sigma, L, V, comp)
    ratio = gr / ga

    edges = np.logspace(np.log10(BAND[0]), np.log10(BAND[1]), N_BINS + 1)
    idx = np.clip(np.digitize(om, edges) - 1, 0, N_BINS - 1)
    binned = np.array([ratio[idx == b].mean() if np.any(idx == b) else np.nan
                       for b in range(N_BINS)])
    good = np.isfinite(binned)
    log_rms = float(np.sqrt(np.mean(np.log10(binned[good]) ** 2)))

    return {
        "component": comp,
        "sigma_target_mps": sigma,
        "sigma_realised_mps": rms,
        "sigma_rel_error": abs(rms - sigma) / sigma,
        "scale_length_m": L,
        "median_psd_ratio": float(np.median(ratio)),
        "log10_ratio_rms_decades": log_rms,
        "band_rad_s": list(BAND),
        "omega": om[::4].tolist(),
        "psd_realised": gr[::4].tolist(),
        "psd_analytic": ga[::4].tolist(),
    }


def run():
    dp = DrydenParams()
    rng = np.random.default_rng(4242)
    comps = [one_component(c, dp, rng) for c in ("u", "v", "w")]

    checks = {}
    for c in comps:
        k = c["component"]
        checks[f"{k}_rms_within_3pct"] = bool(c["sigma_rel_error"] < 0.03)
        checks[f"{k}_median_psd_ratio_within_8pct"] = bool(
            abs(c["median_psd_ratio"] - 1.0) < 0.08)
        checks[f"{k}_log10_psd_rms_lt_0.05_decades"] = bool(
            c["log10_ratio_rms_decades"] < 0.05)

    payload = {
        "name": "dryden_turbulence",
        "description": ("Realised Dryden PSD vs analytic MIL-F-8785C spectrum, "
                        f"{N_REAL} realisations x {N_SAMPLES} samples at "
                        f"dt = {DT} s, V = {V} m/s."),
        "airspeed_mps": V,
        "dt_s": DT,
        "n_realisations": N_REAL,
        "n_samples": N_SAMPLES,
        "components": comps,
        "checks": checks,
        "all_passed": all(checks.values()),
    }
    return payload


if __name__ == "__main__":
    report("validate_turbulence", run())
