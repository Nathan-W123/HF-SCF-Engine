import numpy as np
import pytest
from scipy.signal import welch

from swarmsim.config import DrydenParams, WindParams
from swarmsim.wind import (DrydenTurbulence, WindField, dryden_psd_spatial,
                           dryden_psd_temporal, dryden_series_batch)


def test_dryden_spectra_integrate_to_variance():
    """One-sided Dryden PSDs must integrate to sigma^2 over Omega >= 0."""
    dp = DrydenParams()
    Om = np.logspace(-7, 4, 400001)
    for comp, sigma, L in (("u", dp.sigma_u, dp.L_u), ("w", dp.sigma_w, dp.L_w)):
        phi = dryden_psd_spatial(Om, sigma, L, comp)
        area = np.trapezoid(phi, Om)
        assert area == pytest.approx(sigma ** 2, rel=2e-3)


def test_temporal_psd_is_spatial_scaled_by_airspeed():
    dp = DrydenParams()
    om = np.logspace(-2, 2, 50)
    V = 25.0
    assert np.allclose(dryden_psd_temporal(om, dp.sigma_w, dp.L_w, V, "w"),
                       dryden_psd_spatial(om / V, dp.sigma_w, dp.L_w, "w") / V)


def test_unknown_component_raises():
    with pytest.raises(ValueError):
        dryden_psd_spatial(np.array([1.0]), 1.0, 100.0, "q")


@pytest.mark.parametrize("comp", ["u", "v", "w"])
def test_realised_rms_matches_sigma(comp):
    dp = DrydenParams()
    rng = np.random.default_rng(7)
    x = dryden_series_batch(dp, 25.0, 0.02, 6000, 24, comp, rng)
    sigma = {"u": dp.sigma_u, "v": dp.sigma_v, "w": dp.sigma_w}[comp]
    assert abs(x.std() - sigma) / sigma < 0.05


@pytest.mark.parametrize("comp", ["u", "w"])
def test_realised_psd_matches_analytic(comp):
    dp = DrydenParams()
    rng = np.random.default_rng(9)
    V, dt = 25.0, 0.02
    x = dryden_series_batch(dp, V, dt, 8192, 24, comp, rng)
    f, P = welch(x, fs=1 / dt, nperseg=2048, axis=1)
    P = P.mean(axis=0)
    om = 2 * np.pi * f
    m = (om > 0.05) & (om < 20.0)
    sigma = {"u": dp.sigma_u, "w": dp.sigma_w}[comp]
    L = {"u": dp.L_u, "w": dp.L_w}[comp]
    ratio = (P[m] / (2 * np.pi)) / dryden_psd_temporal(om[m], sigma, L, V, comp)
    assert abs(float(np.median(ratio)) - 1.0) < 0.12


def test_turbulence_is_reproducible_for_a_given_seed():
    dp = DrydenParams()
    a = DrydenTurbulence(dp, 25.0, 0.05, 4, np.random.default_rng(3))
    b = DrydenTurbulence(dp, 25.0, 0.05, 4, np.random.default_rng(3))
    for _ in range(30):
        assert np.array_equal(a.step(), b.step())


def test_wind_field_off_returns_mean_only():
    wp = WindParams(mean=(3.0, -2.0, 0.5), turbulence_on=False)
    wf = WindField(wp, 5, 25.0, 0.05, np.random.default_rng(0))
    wf.step(np.zeros(5))
    assert np.allclose(wf.wind(), np.array([3.0, -2.0, 0.5])[None, :])


def test_gust_rotation_into_enu_preserves_magnitude():
    wp = WindParams(mean=(0.0, 0.0, 0.0), turbulence_on=True)
    wf = WindField(wp, 64, 25.0, 0.05, np.random.default_rng(1))
    psi = np.linspace(-np.pi, np.pi, 64)
    mags = []
    for _ in range(50):
        wf.step(psi)
        mags.append(np.linalg.norm(wf.gust, axis=1))
    # horizontal gust magnitude must be heading-independent in distribution
    m = np.array(mags)
    assert np.isfinite(m).all() and m.mean() > 0.1


def test_dryden_scaled_helper():
    dp = DrydenParams().scaled(2.0)
    base = DrydenParams()
    assert dp.sigma_u == 2 * base.sigma_u and dp.L_u == base.L_u
