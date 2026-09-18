"""Equations of motion, integrators and end-to-end trajectory behaviour."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from reentry import Vehicle, simulate
from reentry.atmosphere import ExponentialAtmosphere, USSA76, VacuumAtmosphere
from reentry.constants import MU_EARTH, OMEGA_EARTH, R_EARTH
from reentry.dynamics import EntryModel, specific_orbital_elements
from reentry.integrators import integrate_dop853, integrate_rk4

ATM = USSA76()
VEH = Vehicle()


@pytest.fixture(scope="module")
def ballistic():
    return simulate(vehicle=VEH, atmosphere=ATM, gamma0_deg=-5.5, velocity0=7800.0,
                    rtol=1e-10, atol=1e-10, n_output=2001)


def test_vacuum_circular_orbit_is_conserved():
    r0 = R_EARTH + 400e3
    v = float(np.sqrt(MU_EARTH / r0))
    period = 2 * np.pi * np.sqrt(r0**3 / MU_EARTH)
    res = simulate(vehicle=VEH, atmosphere=VacuumAtmosphere(), altitude0=400e3,
                   velocity0=v, gamma0_deg=0.0, t_max=2 * period, rtol=1e-12,
                   atol=1e-12, terminal_altitude=-R_EARTH, exit_altitude=1e10,
                   n_output=801)
    pos, vel = res.cartesian()
    e, h, a = specific_orbital_elements(pos, vel)
    assert np.max(np.abs(e - e[0]) / abs(e[0])) < 1e-11
    hm = np.linalg.norm(h, axis=-1)
    assert np.max(np.abs(hm - hm[0]) / hm[0]) < 1e-11
    assert np.max(np.abs(res.radius - r0)) < 1e-3
    assert np.max(np.abs(a - r0)) < 1e-3


def test_rotating_frame_matches_inertial_two_body():
    res = simulate(vehicle=VEH, atmosphere=VacuumAtmosphere(), altitude0=500e3,
                   velocity0=7600.0, gamma0_deg=6.0, psi0_deg=40.0, lat0_deg=20.0,
                   lon0_deg=-30.0, omega=OMEGA_EARTH, t_max=1500.0, rtol=1e-12,
                   atol=1e-12, terminal_altitude=-R_EARTH, exit_altitude=1e10,
                   n_output=301)
    pos, vel = res.cartesian(omega=OMEGA_EARTH)

    def f(t, y):
        r = y[:3]
        return np.concatenate([y[3:], -MU_EARTH * r / np.linalg.norm(r) ** 3])

    sol = solve_ivp(f, (0.0, res.t[-1]), np.concatenate([pos[0], vel[0]]),
                    method="DOP853", rtol=1e-13, atol=1e-9, t_eval=res.t)
    assert np.max(np.linalg.norm(sol.y[:3].T - pos, axis=1)) < 1e-2


def test_energy_budget_closes(ballistic):
    e = 0.5 * ballistic.velocity**2 - MU_EARTH / ballistic.radius
    closure = abs((e[-1] - e[0]) + ballistic.drag_work) / abs(e[-1] - e[0])
    assert float(closure[-1]) < 1e-10


def test_lift_does_no_work():
    lifting = simulate(vehicle=Vehicle(alpha_trim_rad=np.radians(-20.0)),
                       atmosphere=ATM, gamma0_deg=-5.5, velocity0=7800.0,
                       rtol=1e-10, atol=1e-10, n_output=1501)
    e = 0.5 * lifting.velocity**2 - MU_EARTH / lifting.radius
    closure = abs((e[-1] - e[0]) + lifting.drag_work[-1]) / abs(e[-1] - e[0])
    assert closure < 1e-10


def test_rk4_is_fourth_order_on_a_smooth_problem():
    model = EntryModel(vehicle=VEH, atmosphere=ExponentialAtmosphere())
    y0 = np.array([R_EARTH + 120e3, 0.0, 0.0, 7000.0, np.radians(-30.0),
                   np.radians(90.0), 0.0, 0.0, 0.0])
    ref = integrate_dop853(model.rhs, (0.0, 60.0), y0, rtol=1e-13, atol=1e-13)
    y_ref = ref.sol(60.0)
    scale = np.maximum(np.abs(y_ref), 1e-2)
    errs = []
    for n in (200, 400, 800):
        _, ys = integrate_rk4(model.rhs, (0.0, 60.0), y0, n)
        errs.append(float(np.max(np.abs((ys[-1] - y_ref) / scale))))
    orders = [np.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]
    assert all(3.7 < o < 4.4 for o in orders)


def test_ballistic_entry_is_physically_sane(ballistic):
    s = ballistic.summary()
    assert s["termination"] == "terminal_altitude"
    assert np.all(np.diff(ballistic.altitude) < 0)          # monotone descent
    assert np.all(np.diff(ballistic.heat_load) >= 0)        # heat load accumulates
    # Above the sensible atmosphere gravity briefly *accelerates* the vehicle
    # (drag is negligible at 120 km); the gain must be small, and once the drag
    # pulse starts the speed must fall monotonically.
    assert ballistic.velocity.max() / ballistic.velocity[0] < 1.01
    i0 = int(np.argmax(ballistic.velocity))
    assert np.all(np.diff(ballistic.velocity[i0:]) < 0)
    assert 30.0 < s["peak_heat_flux_altitude_km"] < 80.0
    assert 20.0 < s["peak_g_altitude_km"] < 70.0
    # peak heating always happens above (earlier than) peak deceleration
    assert s["peak_heat_flux_altitude_km"] > s["peak_g_altitude_km"]
    assert s["peak_heat_flux_time_s"] < s["peak_g_time_s"]
    assert 1.0 < s["peak_g_load"] < 60.0
    assert s["final_mach"] < 1.0


def test_steeper_entry_gives_higher_g_and_less_downrange():
    out = {}
    for g in (-3.0, -6.0, -9.0):
        r = simulate(vehicle=VEH, atmosphere=ATM, gamma0_deg=g, velocity0=7800.0,
                     rtol=1e-9, atol=1e-9, n_output=801)
        out[g] = r.summary()
    assert out[-3.0]["peak_g_load"] < out[-6.0]["peak_g_load"] < out[-9.0]["peak_g_load"]
    assert out[-3.0]["downrange_km"] > out[-6.0]["downrange_km"] > out[-9.0]["downrange_km"]
    # total heat load is larger for the long shallow entry
    assert out[-3.0]["total_heat_load_J_cm2"] > out[-9.0]["total_heat_load_J_cm2"]


def test_lift_up_reduces_peak_g_and_extends_downrange(ballistic):
    lifting = simulate(vehicle=Vehicle(alpha_trim_rad=np.radians(-20.0)),
                       atmosphere=ATM, gamma0_deg=-5.5, velocity0=7800.0,
                       rtol=1e-9, atol=1e-9, n_output=1501)
    a = ballistic.summary()
    b = lifting.summary()
    assert b["peak_g_load"] < a["peak_g_load"]
    assert b["downrange_km"] > a["downrange_km"]
    assert b["total_heat_load_J_cm2"] > a["total_heat_load_J_cm2"]


def test_super_circular_shallow_entry_skips_out():
    r = simulate(vehicle=VEH, atmosphere=ATM, velocity0=11000.0, gamma0_deg=-4.0,
                 exit_altitude=120e3, rtol=1e-9, atol=1e-9, n_output=801)
    assert r.termination == "skip_out"
    r2 = simulate(vehicle=VEH, atmosphere=ATM, velocity0=11000.0, gamma0_deg=-7.0,
                  exit_altitude=120e3, rtol=1e-9, atol=1e-9, n_output=801)
    assert r2.termination == "terminal_altitude"
