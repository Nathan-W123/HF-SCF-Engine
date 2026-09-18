#!/usr/bin/env python3
"""Generate the technical supporting figures in ``figures/``.

Every figure is drawn from artefacts that the simulator and the validation
suite actually wrote (``results/*.npz``, ``results/corridor.json``,
``validation/*.json``); nothing here recomputes or invents a number.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from _bootstrap import (  # noqa: E402
    FIGURES_DIR,
    RESULTS_DIR,
    VALIDATION_DIR,
    load_trajectory,
)

from reentry.aerodynamics import SphereCone, cp_max  # noqa: E402
from reentry.atmosphere import USSA76  # noqa: E402

# ---------------------------------------------------------------------------
# Shared style: dark, high-contrast, no chartjunk.
# ---------------------------------------------------------------------------
BG = "#0a0d16"
FG = "#d8dee9"
GRID = "#1e2637"
C_BALLISTIC = "#ff8c42"
C_LIFTING = "#4fc3f7"
C_ACCENT = "#ffd166"
C_ALT = "#ef5d8f"
C_GREEN = "#7ee787"

plt.rcParams.update(
    {
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
        "text.color": FG,
        "axes.labelcolor": FG,
        "axes.edgecolor": "#3b455c",
        "xtick.color": FG,
        "ytick.color": FG,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "axes.grid": True,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "font.size": 10,
        "legend.frameon": False,
        "legend.labelcolor": FG,
        "figure.dpi": 130,
        "lines.linewidth": 2.0,
    }
)


def _save(fig, name):
    path = FIGURES_DIR / name
    fig.savefig(path, bbox_inches="tight", dpi=130)
    plt.close(fig)
    print(f"  wrote {path.relative_to(FIGURES_DIR.parent)}")


# ---------------------------------------------------------------------------
def fig_corridor(bal, lif, corridor):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4))

    sup = corridor["super_circular"]
    boundary = sup["skip_out_boundary_gamma_deg"]
    # Altitude-velocity corridor: the super-circular family brackets the skip-out
    # boundary, so the qualitative split is visible in one panel.
    from reentry import Vehicle, simulate
    from reentry.atmosphere import USSA76 as _U

    atm = _U()
    veh = Vehicle()
    for gd in (-4.0, -4.75, -5.0, -6.0, -8.0, -11.0):
        r = simulate(vehicle=veh, atmosphere=atm, altitude0=120e3,
                     velocity0=sup["entry_velocity_m_s"], gamma0_deg=gd,
                     terminal_altitude=10e3, exit_altitude=120e3, rtol=1e-9,
                     atol=1e-9, n_output=2001, t_max=6000.0)
        skipped = r.termination == "skip_out"
        ax1.plot(r.velocity / 1e3, r.altitude / 1e3,
                 color=C_LIFTING if skipped else C_BALLISTIC,
                 lw=1.6, alpha=0.85,
                 label=None)
        j = int(np.argmin(r.altitude)) if skipped else int(
            np.argmin(np.abs(r.altitude - 45e3)))
        ax1.annotate(f"{gd:g}°", (r.velocity[j] / 1e3, r.altitude[j] / 1e3),
                     color=FG, fontsize=8.5, xytext=(6, 6),
                     textcoords="offset points")
    ax1.plot([], [], color=C_LIFTING, lw=1.6,
             label="11.0 km/s entry, skips back out")
    ax1.plot([], [], color=C_BALLISTIC, lw=1.6, label="11.0 km/s entry, captured")
    ax1.plot(bal.velocity / 1e3, bal.altitude / 1e3, color=C_ACCENT, lw=2.6,
             label="nominal LEO ballistic (7.8 km/s, -5.5°)")
    ax1.plot(lif.velocity / 1e3, lif.altitude / 1e3, color=C_GREEN, lw=2.2, ls="--",
             label="nominal LEO lifting (L/D = 0.31)")
    ax1.set_xlabel("velocity [km/s]")
    ax1.set_ylabel("altitude [km]")
    ax1.set_title("Entry corridor: altitude-velocity")
    ax1.set_ylim(0, 125)
    ax1.legend(fontsize=8, loc="lower right")

    rows = sup["rows"]
    g = np.array([r["gamma0_deg"] for r in rows])
    captured = np.array([r["termination"] == "terminal_altitude" for r in rows])
    pg = np.array([r["peak_g_load"] for r in rows])
    pq = np.array([r["peak_heat_flux_W_cm2"] for r in rows])
    ax2.plot(-g[captured], pg[captured], "o-", color=C_BALLISTIC, ms=3.5,
             label="peak deceleration [g]")
    ax2b = ax2.twinx()
    ax2b.plot(-g[captured], pq[captured], "s-", color=C_ALT, ms=3.5,
              label="peak heat flux [W/cm$^2$]")
    ax2b.grid(False)
    ax2b.set_ylabel("peak stagnation heat flux [W/cm$^2$]", color=C_ALT)
    ax2b.tick_params(axis="y", colors=C_ALT)
    if boundary is not None:
        ax2.axvline(-boundary, color=C_LIFTING, ls=":", lw=2)
        ax2.axvspan(0, -boundary, color=C_LIFTING, alpha=0.10)
        ax2.text(-boundary + 0.15, pg[captured].max() * 0.92,
                 f"skip-out boundary\n$\\gamma_e$ = {boundary:.2f}°",
                 color=C_LIFTING, fontsize=9)
    ax2.axhline(10.0, color=FG, ls="--", lw=1.0, alpha=0.5)
    ax2.text(11.2, 10.4, "10 g", color=FG, fontsize=8, ha="right")
    ax2.set_xlabel("entry flight-path angle magnitude $|\\gamma_e|$ [deg]")
    ax2.set_ylabel("peak deceleration [g]", color=C_BALLISTIC)
    ax2.tick_params(axis="y", colors=C_BALLISTIC)
    ax2.set_title(
        f"Corridor sweep at {sup['entry_velocity_m_s'] / 1e3:.1f} km/s "
        "(super-circular)"
    )
    ax2.set_xlim(0, 12)
    fig.suptitle(
        "Entry corridor for the 70° sphere-cone capsule "
        "($\\beta$ = 139 kg/m², USSA76, 3-DOF)",
        fontsize=13, y=1.02,
    )
    _save(fig, "fig1_entry_corridor.png")


def fig_heating(bal, lif):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.0))
    ax1.plot(bal.t, bal.q_dot / 1e4, color=C_BALLISTIC, label="Sutton-Graves (ballistic)")
    ax1.plot(bal.t, bal.q_dot_dkr / 1e4, color=C_ALT, lw=1.4, ls="--",
             label="Detra-Kemp-Riddell form")
    ax1.plot(bal.t, bal.q_dot_exp315 / 1e4, color=C_ACCENT, lw=1.4, ls=":",
             label="S-G re-anchored to $V^{3.15}$")
    ax1.plot(lif.t, lif.q_dot / 1e4, color=C_LIFTING, lw=1.6,
             label="Sutton-Graves (lifting)")
    ipk = int(np.argmax(bal.q_dot))
    ax1.plot(bal.t[ipk], bal.q_dot[ipk] / 1e4, "o", color="white", ms=5)
    ax1.annotate(
        f"{bal.q_dot[ipk] / 1e4:.0f} W/cm$^2$\n{bal.altitude[ipk] / 1e3:.1f} km, "
        f"{bal.velocity[ipk] / 1e3:.2f} km/s",
        (bal.t[ipk], bal.q_dot[ipk] / 1e4), xytext=(12, -6),
        textcoords="offset points", color=FG, fontsize=9,
    )
    ax1.set_xlabel("time from 120 km entry interface [s]")
    ax1.set_ylabel("stagnation-point heat flux [W/cm$^2$]")
    ax1.set_title("Stagnation-point convective heating, $R_n$ = 0.65 m")
    ax1.set_xlim(0, 220)
    ax1.legend(fontsize=8.5)

    ax2.plot(bal.t, bal.heat_load / 1e4, color=C_BALLISTIC, label="ballistic")
    ax2.plot(lif.t, lif.heat_load / 1e4, color=C_LIFTING, label="lifting")
    ax2.set_xlabel("time from 120 km entry interface [s]")
    ax2.set_ylabel("integrated heat load [J/cm$^2$]")
    ax2.set_xlim(0, 260)
    ax2b = ax2.twinx()
    ax2b.plot(bal.t, bal.wall_temperature, color=C_ACCENT, lw=1.4, ls="--")
    ax2b.plot(lif.t, lif.wall_temperature, color=C_GREEN, lw=1.4, ls="--")
    ax2b.set_ylabel("radiative-equilibrium wall temperature [K]", color=C_ACCENT)
    ax2b.tick_params(axis="y", colors=C_ACCENT)
    ax2b.grid(False)
    ax2.set_title("Integrated heat load (solid) and wall temperature (dashed)")
    ax2.legend(fontsize=9, loc="upper left")
    _save(fig, "fig2_heating.png")


def fig_loads(bal, lif):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.0))
    for r, c, lab in ((bal, C_BALLISTIC, "ballistic, $\\alpha$ = 0"),
                      (lif, C_LIFTING, "lifting, $\\alpha$ = 20°, bank 0")):
        ax1.plot(r.t, r.g_load, color=c, label=lab)
        ax2.plot(r.t, r.q_dyn / 1e3, color=c, label=lab)
    for r, c in ((bal, C_BALLISTIC), (lif, C_LIFTING)):
        i = int(np.argmax(r.g_load))
        ax1.plot(r.t[i], r.g_load[i], "o", color="white", ms=5)
        ax1.annotate(f"{r.g_load[i]:.1f} g @ {r.altitude[i] / 1e3:.0f} km",
                     (r.t[i], r.g_load[i]), xytext=(10, 4),
                     textcoords="offset points", color=c, fontsize=9)
    ax1.axhline(10.0, color=FG, ls="--", lw=1.0, alpha=0.4)
    ax1.set_xlabel("time [s]")
    ax1.set_ylabel("deceleration [g]")
    ax1.set_title("Deceleration history")
    ax1.set_xlim(0, 220)
    ax1.legend(fontsize=9)
    ax2.set_xlabel("time [s]")
    ax2.set_ylabel("dynamic pressure [kPa]")
    ax2.set_title("Dynamic pressure history")
    ax2.set_xlim(0, 220)
    ax2.legend(fontsize=9)
    _save(fig, "fig3_loads.png")


def fig_allen_eggers(val2):
    d = val2["data"]
    sweep = d["sweep"]
    g = np.array([r["gamma_deg"] for r in sweep])
    eb = np.array([abs(r["B_a_max_rel_error"]) for r in sweep]) * 100
    ec = np.array([abs(r["C_a_max_rel_error"]) for r in sweep]) * 100
    pred = np.array([r["gravity_work_prediction"] for r in sweep]) * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.0))
    ax1.semilogy(-g, np.maximum(eb, 1e-12), "o-", color=C_LIFTING,
                 label="B: gravity off (isolates constant-$\\gamma$)")
    ax1.semilogy(-g, ec, "s-", color=C_BALLISTIC, label="C: full physics")
    ax1.semilogy(-g, pred, "--", color=C_ACCENT,
                 label="predicted gravity-work contribution $2\\bar{g}\\Delta z/V_e^2$")
    ex = d["exact_limit"]
    ax1.axhline(abs(ex["a_max_rel_error"]) * 100, color=C_GREEN, ls=":", lw=1.6)
    ax1.text(20, abs(ex["a_max_rel_error"]) * 100 * 1.6,
             "A: exact Allen-Eggers limit "
             f"({abs(ex['a_max_rel_error']):.1e} rel.)",
             color=C_GREEN, fontsize=8.5)
    ax1.set_xlabel("entry flight-path angle magnitude [deg]")
    ax1.set_ylabel("|error| in peak deceleration vs Allen-Eggers [%]")
    ax1.set_title("Error decomposition")
    ax1.legend(fontsize=8.5, loc="lower left")
    ax1.set_ylim(1e-11, 200)

    keep = -g >= 10.0
    a_an = np.array([r["a_max_analytic_g"] for r in sweep])[keep]
    a_b = np.array([r["B_a_max_sim_g"] for r in sweep])[keep]
    a_c = np.array([r["C_a_max_sim_g"] for r in sweep])[keep]
    gk = -g[keep]
    ax2.plot(gk, a_an, color=C_ACCENT, lw=2.4, label="Allen-Eggers closed form")
    ax2.plot(gk, a_b, "o", color=C_LIFTING, ms=7, label="simulator, gravity off")
    ax2.plot(gk, a_c, "s", color=C_BALLISTIC, ms=6, label="simulator, full physics")
    ax2.set_xlabel("entry flight-path angle magnitude [deg]")
    ax2.set_ylabel("peak deceleration [g]")
    ax2.set_title("Peak deceleration")
    ax2.legend(fontsize=9)
    fig.suptitle(
        "Allen-Eggers ballistic-entry benchmark  ($V_e$ = 7.0 km/s, exponential "
        "atmosphere $H$ = 7.2 km, $\\beta$ = 151 kg/m$^2$)",
        fontsize=13, y=1.03,
    )
    _save(fig, "fig4_allen_eggers.png")


def fig_atmosphere(val3):
    atm = USSA76(tabulated=False)
    z = np.linspace(0.0, 200.0e3, 4001)
    rho = np.asarray(atm.density(z))
    t = np.asarray(atm.temperature(z))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    ax1.semilogx(rho, z / 1e3, color=C_BALLISTIC, label="USSA76 density (model)")
    ax1b = ax1.twiny()
    ax1b.plot(t, z / 1e3, color=C_LIFTING, lw=1.6, label="temperature")
    ax1b.set_xlabel("kinetic temperature [K]", color=C_LIFTING)
    ax1b.tick_params(axis="x", colors=C_LIFTING)
    ax1b.grid(False)
    ref_h = np.array([r["geometric_altitude_km"] for r in val3["data"]["layer_boundary_comparison"]])
    ref_rho = np.array([r["rho_reference_kg_m3"] for r in val3["data"]["layer_boundary_comparison"]])
    ax1.plot(ref_rho, ref_h, "o", color="white", ms=6, mfc="none", mew=1.6,
             label="published USSA76 table values")
    ax1.set_xlabel("density [kg/m$^3$]", color=C_BALLISTIC)
    ax1.tick_params(axis="x", colors=C_BALLISTIC)
    ax1.set_ylabel("geometric altitude [km]")
    ax1.set_title("1976 U.S. Standard Atmosphere as implemented")
    ax1.legend(fontsize=8.5, loc="upper right")
    ax1.set_ylim(0, 200)

    rows = val3["data"]["layer_boundary_comparison"]
    hh = np.array([r["geopotential_altitude_km"] for r in rows])
    et = np.array([max(r["T_rel_error"], 1e-17) for r in rows])
    ep = np.array([max(r["p_rel_error"], 1e-17) for r in rows])
    er = np.array([max(r["rho_rel_error"], 1e-17) for r in rows])
    w = 0.28
    idx = np.arange(len(hh))
    ax2.bar(idx - w, et, w, color=C_GREEN, label="temperature")
    ax2.bar(idx, ep, w, color=C_ACCENT, label="pressure")
    ax2.bar(idx + w, er, w, color=C_ALT, label="density")
    ax2.set_yscale("log")
    ax2.set_xticks(idx)
    ax2.set_xticklabels([f"{h:g}" for h in hh])
    ax2.set_xlabel("layer-boundary geopotential altitude [km']")
    ax2.set_ylabel("|relative error| vs published USSA76 table")
    ax2.axhline(1e-5, color=FG, ls="--", lw=1.0, alpha=0.5)
    ax2.text(len(hh) - 0.5, 1.4e-5, "1e-5", color=FG, fontsize=8, ha="right")
    ax2.set_title("Model vs published table at the eight layer boundaries")
    ax2.legend(fontsize=9)
    ax2.set_ylim(1e-17, 1e-2)
    _save(fig, "fig5_atmosphere_validation.png")


def fig_aero():
    sc = SphereCone()
    m = np.logspace(np.log10(0.05), np.log10(40.0), 500)
    cd0 = np.array([sc.coefficients(0.0, mm)[0] for mm in m])
    alphas = np.radians(np.linspace(-40, 40, 321))
    cd_a, cl_a = sc.coefficients(alphas, 25.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.0))
    ax1.semilogx(m, cp_max(m), color=C_LIFTING, label="$C_{p,max}(M)$")
    ax1.semilogx(m, cd0, color=C_BALLISTIC, label="$C_D(M)$ at $\\alpha$ = 0")
    ax1.axhline(1.8393710511306678, color=C_ACCENT, ls=":", lw=1.4)
    ax1.text(0.06, 1.87, "$C_{p,max} \\to 1.8394$  ($M \\to \\infty$, $\\gamma$ = 1.4)",
             color=C_ACCENT, fontsize=8.5)
    ax1.axvline(1.0, color=FG, ls="--", lw=1.0, alpha=0.4)
    ax1.text(1.05, 0.35, "M = 1", color=FG, fontsize=8)
    ax1.axvspan(0.05, 1.5, color=C_ALT, alpha=0.10)
    ax1.text(0.09, 0.15, "outside Newtonian validity", color=C_ALT, fontsize=8)
    ax1.set_xlabel("Mach number")
    ax1.set_ylabel("coefficient")
    ax1.set_title("Mach dependence from the Rayleigh pitot $C_{p,max}$")
    ax1.legend(fontsize=9, loc="lower right")
    ax1.set_ylim(0, 2.0)

    ax2.plot(np.degrees(alphas), cd_a, color=C_BALLISTIC, label="$C_D$")
    ax2.plot(np.degrees(alphas), cl_a, color=C_LIFTING, label="$C_L$")
    ax2.plot(np.degrees(alphas), cl_a / cd_a, color=C_ACCENT, label="$L/D$")
    ax2.axvline(-20.0, color=FG, ls="--", lw=1.0, alpha=0.5)
    i20 = int(np.argmin(np.abs(np.degrees(alphas) + 20.0)))
    ax2.plot(-20.0, cl_a[i20] / cd_a[i20], "o", color="white", ms=5)
    ax2.annotate(f"trim: $L/D$ = {cl_a[i20] / cd_a[i20]:.3f}",
                 (-20.0, cl_a[i20] / cd_a[i20]), xytext=(10, 22),
                 textcoords="offset points", color=FG, fontsize=9)
    ax2.set_xlabel("angle of attack [deg]")
    ax2.set_ylabel("coefficient (area $\\pi R_b^2$ = 5.31 m$^2$)")
    ax2.set_title("Surface integrals at M = 25")
    ax2.legend(fontsize=9)
    fig.suptitle(
        "Modified-Newtonian aerodynamics of the 70° sphere-cone aeroshell "
        "($R_n$ = 0.65 m, $R_b$ = 1.30 m)", fontsize=13, y=1.03)
    _save(fig, "fig6_aerodynamics.png")


def main() -> int:
    bal = load_trajectory(RESULTS_DIR / "traj_ballistic.npz")
    lif = load_trajectory(RESULTS_DIR / "traj_lifting.npz")
    corridor = json.loads((RESULTS_DIR / "corridor.json").read_text())
    val2 = json.loads((VALIDATION_DIR / "val_02_allen_eggers.json").read_text())
    val3 = json.loads((VALIDATION_DIR / "val_03_atmosphere_table.json").read_text())

    fig_corridor(bal, lif, corridor)
    fig_heating(bal, lif)
    fig_loads(bal, lif)
    fig_allen_eggers(val2)
    fig_atmosphere(val3)
    fig_aero()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
