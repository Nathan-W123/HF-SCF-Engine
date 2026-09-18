#!/usr/bin/env python3
"""Technical supporting figures (figures/*.png).

Six figures, all from artefacts written by ``run_all.py`` / the validation
scripts:

  fig1_separation.png   minimum pairwise separation vs time for every scenario,
                        against the R_min and R_collision thresholds
  fig2_topdown.png      top-down trajectories of the nominal transit through the
                        no-fly-zone field, with planned routes and zones
  fig3_tracking.png     L1 cross-track-error capture curves, straight + circular
  fig4_dryden.png       realised vs analytic Dryden PSD for u, v, w
  fig5_sweep.png        metrics across the wind x seed robustness sweep
  fig6_failure.png      the failure-case timeline (separation to the rogue
                        vehicle, avoidance activity, and who was affected)
"""

from __future__ import annotations

import json
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
sys.path.insert(0, os.path.join(HERE, "validation"))

from swarmsim import metrics              # noqa: E402

FIG = os.path.join(HERE, "figures")
RES = os.path.join(HERE, "results")
VAL = os.path.join(HERE, "validation")

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
    "figure.facecolor": "white",
})

PALETTE = {"transit": "#2b7bba", "swap": "#e2711d", "gust": "#3fa34d",
           "failure": "#b23a48"}


def _load(name):
    with open(os.path.join(RES, f"{name}.pkl"), "rb") as fh:
        return pickle.load(fh)


def _json(path):
    with open(path) as fh:
        return json.load(fh)


# --------------------------------------------------------------------------
def fig_separation(runs):
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    R_min = R_col = None
    for name, r in runs.items():
        d, _, _ = metrics.pairwise_min_distance(r.states)
        ax.plot(r.t, d.min(axis=1), lw=1.3, color=PALETTE.get(name, "#555"),
                label=f"{name} (min {d.min():.1f} m)")
        R_min = r.spec.avoid.R_min
        R_col = r.spec.avoid.R_collision
    ax.axhline(R_min, color="k", ls="--", lw=1.1)
    ax.text(0.995, R_min + 4, f"$R_{{min}}$ = {R_min:.0f} m", ha="right",
            va="bottom", transform=ax.get_yaxis_transform(), fontsize=8)
    ax.axhline(R_col, color="#a00", ls=":", lw=1.1)
    ax.text(0.995, R_col + 4, f"$R_{{collision}}$ = {R_col:.0f} m", ha="right",
            va="bottom", transform=ax.get_yaxis_transform(), fontsize=8,
            color="#a00")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("minimum pairwise separation [m]")
    ax.set_title("Minimum pairwise separation over time, all scenarios")
    ax.set_yscale("log")
    ax.set_ylim(10, 4000)
    ax.legend(loc="upper right", ncols=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_separation.png"))
    plt.close(fig)


def fig_topdown(runs):
    r = runs["transit"]
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    ax.set_facecolor("#f7f8fa")
    for z in r.spec.zones:
        ax.add_patch(plt.Circle((z.x, z.y), z.radius, color="#c0392b",
                                alpha=0.16, ec="#c0392b", lw=1.0, zorder=1))
        ax.add_patch(plt.Circle((z.x, z.y), z.radius + r.spec.clearance,
                                color="none", ec="#c0392b", ls="--", lw=0.7,
                                alpha=0.55, zorder=1))
    for wp in r.paths:
        ax.plot(wp[:, 0], wp[:, 1], color="#888", lw=0.6, alpha=0.55, zorder=2)
    alt = r.states[0, :, 2]
    norm = (alt - alt.min()) / max(alt.ptp(), 1e-9)
    cmap = plt.get_cmap("viridis")
    for i in range(r.states.shape[1]):
        ax.plot(r.states[:, i, 0], r.states[:, i, 1], lw=0.85,
                color=cmap(norm[i]), alpha=0.9, zorder=3)
    ax.scatter(r.spec.goals[:, 0], r.spec.goals[:, 1], s=9, marker="s",
               color="#222", zorder=4, label="goals")
    ax.scatter(r.spec.starts[:, 0], r.spec.starts[:, 1], s=9, marker="o",
               facecolor="none", edgecolor="#222", lw=0.6, zorder=4,
               label="starts")
    ax.set_aspect("equal")
    ax.set_xlabel("east [m]")
    ax.set_ylabel("north [m]")
    ax.set_title(f"Nominal transit: {r.states.shape[1]} vehicles, "
                 f"{len(r.spec.zones)} no-fly cylinders\n"
                 "grey = planned route (visibility graph + A*), "
                 "colour = flown track (hue = cruise altitude)")
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_topdown.png"))
    plt.close(fig)


def fig_tracking():
    z = np.load(os.path.join(VAL, "tracking_curves.npz"))
    offsets = z["offsets"]
    res = _json(os.path.join(VAL, "validate_tracking.json"))
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.5), sharey=True)
    for ax, kind, title in ((axes[0], "straight", "straight reference path"),
                            (axes[1], "circle", "circular path, R = 600 m")):
        cmap = plt.get_cmap("coolwarm")
        for i, off in enumerate(offsets):
            d, x = z[f"{kind}_{i}"]
            ax.plot(d, x, lw=1.1,
                    color=cmap((off - offsets.min()) /
                               max(offsets.ptp(), 1e-9)),
                    label=f"{off:+.0f} m")
        ax.axhline(0, color="k", lw=0.8)
        ss = res[f"worst_{kind}_steady_state_xte_m"]
        st = res[f"worst_{kind}_settling_distance_m"]
        ax.set_title(f"{title}\nworst settling {st:.0f} m, "
                     f"worst steady-state |xte| {ss:.2f} m")
        ax.set_xlabel("distance flown [m]")
        ax.set_xlim(0, 4000)
    axes[0].set_ylabel("cross-track error [m]")
    axes[0].legend(title="initial offset", ncols=2, loc="upper right")
    fig.suptitle("L1 nonlinear guidance: cross-track capture", y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_tracking.png"))
    plt.close(fig)


def fig_dryden():
    res = _json(os.path.join(VAL, "validate_turbulence.json"))
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.3), sharey=True)
    for ax, c in zip(axes, res["components"]):
        om = np.array(c["omega"])
        ax.loglog(om, c["psd_realised"], lw=1.6, color="#2b7bba",
                  label="realised (Welch, 48 runs)")
        ax.loglog(om, c["psd_analytic"], lw=1.2, ls="--", color="#b23a48",
                  label="analytic Dryden")
        ax.set_title(f"${c['component']}$-gust   "
                     f"$\\sigma$={c['sigma_target_mps']:.2f} m/s, "
                     f"$L$={c['scale_length_m']:.0f} m\n"
                     f"median ratio {c['median_psd_ratio']:.3f}, "
                     f"rms {c['log10_ratio_rms_decades']:.3f} dec")
        ax.set_xlabel(r"$\omega$ [rad/s]")
    axes[0].set_ylabel(r"PSD [(m/s)$^2$/(rad/s)]")
    axes[0].legend(loc="lower left")
    fig.suptitle("MIL-F-8785C Dryden turbulence: realised vs analytic spectrum",
                 y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig4_dryden.png"))
    plt.close(fig)


def fig_sweep():
    sw = _json(os.path.join(RES, "sweep.json"))
    rows = sw["runs"]
    levels = sorted({r["sigma_scale"] for r in rows})
    fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.3))

    def collect(key):
        return [[r[key] for r in rows if r["sigma_scale"] == L] for L in levels]

    sig_w = [sw["sigma_w_mps"][str(L)] for L in levels]

    ms = collect("min_separation_m")
    axes[0].boxplot(ms, positions=range(len(levels)), widths=0.55,
                    medianprops=dict(color="#2b7bba"))
    for i, v in enumerate(ms):
        axes[0].scatter(np.full(len(v), i) + np.random.uniform(-.12, .12, len(v)),
                        v, s=12, color="#2b7bba", alpha=0.65, zorder=3)
    axes[0].axhline(sw["R_min_m"], color="k", ls="--", lw=1.0)
    axes[0].set_ylabel("min separation [m]")
    axes[0].set_title("safety margin")

    axes[1].boxplot(collect("path_efficiency_mean"), positions=range(len(levels)),
                    widths=0.55, medianprops=dict(color="#e2711d"))
    axes[1].set_ylabel("flown / straight-line")
    axes[1].set_title("path efficiency")

    axes[2].boxplot(collect("goal_completion"), positions=range(len(levels)),
                    widths=0.55, medianprops=dict(color="#3fa34d"))
    axes[2].set_ylabel("fraction of goals reached")
    axes[2].set_ylim(-0.05, 1.08)
    axes[2].set_title("mission completion")

    for ax in axes:
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels([f"{s:.1f}" for s in sig_w])
        ax.set_xlabel(r"turbulence $\sigma_w$ [m/s]")
    fig.suptitle(f"Robustness sweep: {len(levels)} wind levels x "
                 f"{sw['n_seeds']} seeds ({len(rows)} runs)", y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig5_sweep.png"))
    plt.close(fig)


def fig_failure(runs):
    r = runs["failure"]
    fid = r.spec.fail_id
    pos = r.states[:, :, :3]
    d_rogue = np.linalg.norm(pos - pos[:, fid:fid + 1, :], axis=2)
    d_rogue[:, fid] = np.nan
    d_others, iu, ju = metrics.pairwise_min_distance(r.states)
    keep = (iu != fid) & (ju != fid)

    fig, axes = plt.subplots(2, 1, figsize=(7.6, 5.4), sharex=True,
                             height_ratios=[1.35, 1])
    ax = axes[0]
    ax.plot(r.t, np.nanmin(d_rogue, axis=1), lw=1.5, color="#b23a48",
            label=f"closest vehicle to the failed vehicle (#{fid})")
    ax.plot(r.t, d_others[:, keep].min(axis=1), lw=1.3, color="#2b7bba",
            label="closest cooperative pair")
    ax.axhline(r.spec.avoid.R_min, color="k", ls="--", lw=1.0)
    ax.axvline(r.spec.fail_time, color="#444", ls="-.", lw=1.0)
    ax.text(r.spec.fail_time + 2, 1500, "control failure\n+ non-cooperative",
            fontsize=8, color="#444")
    ax.set_yscale("log")
    ax.set_ylabel("separation [m]")
    ax.legend(loc="upper right")
    ax.set_title(f"Failure case: vehicle #{fid} locks "
                 f"{np.rad2deg(r.spec.fail_bank):.0f}$^\\circ$ of bank at "
                 f"t = {r.spec.fail_time:.0f} s and stops cooperating")

    ax = axes[1]
    ax.plot(r.t, r.deflection.mean(axis=1), lw=1.3, color="#e2711d",
            label="mean avoidance deflection")
    ax.plot(r.t, r.deflection.max(axis=1), lw=1.0, color="#e2711d", alpha=0.45,
            label="max avoidance deflection")
    n_engaged = (r.deflection > 0.5).sum(axis=1)
    ax2 = ax.twinx()
    ax2.fill_between(r.t, 0, n_engaged, color="#2b7bba", alpha=0.18, lw=0)
    ax2.set_ylabel("vehicles manoeuvring", color="#2b7bba")
    ax2.grid(False)
    ax.axvline(r.spec.fail_time, color="#444", ls="-.", lw=1.0)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("deflection |v$_{safe}$ - v$_{pref}$| [m/s]")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig6_failure.png"))
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    runs = {n: _load(n) for n in ("transit", "swap", "gust", "failure")}
    fig_separation(runs)
    fig_topdown(runs)
    fig_tracking()
    fig_dryden()
    fig_sweep()
    fig_failure(runs)
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
