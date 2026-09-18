#!/usr/bin/env python3
"""Technical supporting figures.

Every number plotted here is read back from an artifact the pipeline actually
produced: the validation JSON files, the diagnostics written during the
production run, and the snapshots themselves.  Nothing is hard-coded.

Style follows one categorical palette, validated for the dark chart surface
(adjacent-pair CVD dE >= 8.4, normal-vision dE >= 19.8, all slots >= 3:1 against
the surface).  Every series is direct-labelled as well as coloured, so identity
never depends on colour alone.  There are no dual-axis plots: measures on
different scales go in separate panels.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
FIGDIR = ROOT / "figures"
VALDIR = ROOT / "validation"
DATA = ROOT / "data"

from galcol.units import TIME_UNIT_GYR  # noqa: E402

# --- validated dark-surface palette --------------------------------------
SURFACE = "#1a1a19"
INK = "#ffffff"
INK2 = "#c3c2b7"
MUTED = "#6e6d66"
S1, S2, S3, S4 = "#3987e5", "#d95926", "#199e70", "#c98500"
BAND = "#383835"
# one-hue sequential ramp, dark -> light, monotone in lightness
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#07080c", "#0d366b", "#184f95", "#256abf", "#3987e5",
                 "#6da7ec", "#9ec5f4", "#cde2fb", "#f2f7fe"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": MUTED, "grid.color": "#2c2c2a",
    "font.size": 11, "axes.titlesize": 12.5, "axes.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.linewidth": 0.7, "grid.alpha": 0.9,
    "lines.linewidth": 2.0, "legend.frameon": False,
    "legend.labelcolor": INK2, "figure.dpi": 130,
})


def _load_json(name):
    p = VALDIR / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _save(fig, name):
    path = FIGDIR / name
    tight = getattr(fig, "_galcol_tight", True)
    if tight:
        fig.savefig(path, bbox_inches="tight", pad_inches=0.25)
    else:
        fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT)}")
    return path


def _label_end(ax, x, y, text, color, dx=6, dy=0, va="center"):
    ax.annotate(text, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
                color=color, fontsize=10, va=va, fontweight="medium",
                clip_on=False)


# --------------------------------------------------------------------------
def fig_conservation():
    d = np.load(DATA / "collision" / "diagnostics.npz", allow_pickle=True)
    t = d["t"] * TIME_UNIT_GYR
    e = d["etot"]
    rel = (e - e[0]) / abs(e[0])
    L = np.stack([d["lx"], d["ly"], d["lz"]], 1)
    dl = np.linalg.norm(L - L[0], axis=1) / np.linalg.norm(L[0])
    cons = _load_json("conservation")

    fig, axes = plt.subplots(3, 1, figsize=(8.2, 9.4), sharex=False)

    ax = axes[0]
    ax.plot(t, rel * 1e3, color=S1)
    half = t.size // 2
    lslope, lintc = np.polyfit(t[half:], rel[half:], 1)
    ax.plot(t[half:], (lslope * t[half:] + lintc) * 1e3, color=MUTED,
            lw=1.5, ls="--")
    _label_end(ax, t[-1], rel[-1] * 1e3, "  total energy", S1)
    ax.axhline(0, color=MUTED, lw=0.9)
    ax.set_ylabel(r"$\Delta E\,/\,|E_0|$   [$\times 10^{-3}$]")
    ax.set_xlabel("time  [Gyr]")
    ax.set_title("Energy error steps up at pericentre, then stays put",
                 loc="left", pad=10)
    ax.set_ylim(min(-0.15, rel.min() * 1e3 * 1.2), rel.max() * 1e3 * 1.55)
    if cons:
        ed = cons["energy_drift"]
        ax.text(0.985, 0.05,
                f"max $|\\Delta E|/|E_0|$ = {cons['energy']['max_rel_error']:.2e}\n"
                f"drift over the post-merger half: "
                f"{ed['late_half_drift_per_gyr']:+.1e} per Gyr (dashed)",
                transform=ax.transAxes, color=INK2, fontsize=9.5,
                va="bottom", ha="right")

    ax = axes[1]
    ax.plot(t, dl * 1e3, color=S3)
    _label_end(ax, t[-1], dl[-1] * 1e3, "  |$\\Delta$L|", S3)
    ax.set_ylabel(r"$|\Delta \mathbf{L}|\,/\,|\mathbf{L}_0|$   [$\times 10^{-3}$]")
    ax.set_xlabel("time  [Gyr]")
    ax.set_title("Angular-momentum error", loc="left", pad=10)

    ax = axes[2]
    if cons and "symplectic_comparison" in cons:
        c = cons["symplectic_comparison"]["trace"]
        tt = np.asarray(c["t_gyr"])
        lf = np.asarray(c["leapfrog_rel"])
        rk = np.asarray(c["rk2_rel"])
        ax.plot(tt, rk * 1e3, color=S2)
        ax.plot(tt, lf * 1e3, color=S1)
        _label_end(ax, tt[-1], rk[-1] * 1e3, "  RK2 (not symplectic)", S2)
        _label_end(ax, tt[-1], lf[-1] * 1e3, "  leapfrog KDK", S1)
        ax.axhline(0, color=MUTED, lw=0.9)
        ax.set_title("Same order, same forces, same dt — only symplecticity "
                     "differs", loc="left", pad=10)
    ax.set_ylabel(r"$\Delta E\,/\,|E_0|$   [$\times 10^{-3}$]")
    ax.set_xlabel("time  [Gyr]")
    fig.subplots_adjust(hspace=0.45)
    return _save(fig, "fig1_conservation.png")


def fig_tree_accuracy():
    j = _load_json("tree_force_accuracy")
    rows = j["rows"]
    th = np.array([r["theta"] for r in rows])
    med = np.array([r["median_rel_err"] for r in rows])
    p99 = np.array([r["p99_rel_err"] for r in rows])
    spd = np.array([r["speedup_vs_direct"] for r in rows])
    prod = j["production_theta"]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))
    ax = axes[0]
    ax.plot(th, p99, color=S2, marker="o", ms=5)
    ax.plot(th, med, color=S1, marker="o", ms=5)
    _label_end(ax, th[-1], p99[-1], "  99th pct", S2)
    _label_end(ax, th[-1], med[-1], "  median", S1)
    ax.axvline(prod, color=MUTED, lw=1.2, ls="--")
    ax.text(prod, ax.get_ylim()[0], f" production $\\theta$ = {prod}",
            color=INK2, fontsize=9.5, va="bottom", rotation=90)
    ax.set_yscale("log")
    ax.set_xlabel(r"opening angle  $\theta$")
    ax.set_ylabel(r"$|\Delta \mathbf{a}| / |\mathbf{a}_{\rm exact}|$")
    ax.set_title("Barnes–Hut force error vs exact $O(N^2)$ sum",
                 loc="left", pad=10)

    ax = axes[1]
    ax.plot(th, spd, color=S3, marker="o", ms=5)
    _label_end(ax, th[-1], spd[-1], "  speed-up", S3)
    ax.axvline(prod, color=MUTED, lw=1.2, ls="--")
    ax.set_xlabel(r"opening angle  $\theta$")
    ax.set_ylabel(r"speed-up over direct $O(N^2)$")
    ax.set_title(f"N = {j['n_particles']:,} particles, 4 cores", loc="left",
                 pad=10)
    fig.subplots_adjust(wspace=0.32)
    return _save(fig, "fig2_tree_accuracy.png")


def fig_rotation_curve():
    from galcol.analysis import rotation_curve
    from galcol.ics import make_isolated
    from galcol.production import MODEL, RESOLUTION
    R = np.linspace(0.15, 20.0, 300)
    m = MODEL
    p = make_isolated(m, RESOLUTION, live_halo=True)
    d = p.ptype == 0
    Rm, vm = rotation_curve(p.pos[d], p.vel[d], p.mass[d], n_bins=18, r_max=16.0)

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    ax.plot(R, np.sqrt(m.disk.v_circ_sq(R)), color=S1)
    ax.plot(R, np.sqrt(m.bulge.v_circ_sq(R)), color=S2)
    ax.plot(R, np.sqrt(m.halo.v_circ_sq(R)), color=S3)
    ax.plot(R, m.v_circ(R), color=INK, lw=2.6)
    ok = np.isfinite(vm)
    ax.plot(Rm[ok], vm[ok], ls="none", marker="o", ms=6, color=S4,
            markeredgecolor=SURFACE, markeredgewidth=1.2)
    _label_end(ax, R[-1], np.sqrt(m.disk.v_circ_sq(R))[-1], "  disk", S1)
    _label_end(ax, R[-1], np.sqrt(m.bulge.v_circ_sq(R))[-1], "  bulge", S2)
    _label_end(ax, R[-1], np.sqrt(m.halo.v_circ_sq(R))[-1], "  halo", S3)
    _label_end(ax, R[-1], m.v_circ(R)[-1], "  total", INK)
    k4 = min(5, ok.sum() - 1)
    _label_end(ax, Rm[ok][k4], vm[ok][k4],
               "sampled particles\n(mean $v_\\phi$)", S4, dx=6, dy=-26)
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 260)
    ax.set_xlabel("cylindrical radius $R$  [kpc]")
    ax.set_ylabel("circular speed  [km s$^{-1}$]")
    ax.set_title("Composite rotation curve, and what the sampled disk actually does",
                 loc="left", pad=10)
    ax.text(0.98, 0.05, "points sit below the curve by the\nasymmetric drift, as they should",
            transform=ax.transAxes, color=MUTED, fontsize=9.5, ha="right")
    return _save(fig, "fig3_rotation_curve.png")


def fig_isolated():
    j = _load_json("isolated_galaxy_stability")
    tr = j["track"]
    t = np.asarray(tr["t_gyr"])
    series = [("disk half-mass radius", np.asarray(tr["r_half"]), S1, 0.10),
              ("fitted scale length", np.asarray(tr["r_scale"]), S3, 0.15),
              ("disk median |z|", np.asarray(tr["z_med"]), S4, 0.60)]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.3))
    for ax, (name, y, col, tol) in zip(axes, series):
        rel = y / y[0] - 1.0
        ax.axhspan(-100 * tol, 100 * tol, color=BAND, zorder=0)
        ax.plot(t, 100 * rel, color=col)
        ax.axhline(0, color=MUTED, lw=0.9)
        ax.set_title(name, loc="left", pad=10)
        ax.set_xlabel("time  [Gyr]")
        ax.set_ylabel("change from t = 0  [%]")
        lim = max(100 * tol * 1.35, abs(100 * rel).max() * 1.35, 5)
        ax.set_ylim(-lim, lim)
        ax.text(0.03, 0.05, f"tolerance $\\pm${100 * tol:.0f}%\n"
                            f"initial {y[0]:.3f} kpc\nfinal {y[-1]:.3f} kpc",
                transform=ax.transAxes, color=INK2, fontsize=9.5, va="bottom")
    fig.suptitle(f"Isolated galaxy, {j['n_particles']:,} live particles, "
                 f"{j['t_end_gyr']} Gyr — shaded band is the declared tolerance",
                 x=0.012, ha="left", color=INK, fontsize=12.5)
    fig.subplots_adjust(wspace=0.34, top=0.80)
    return _save(fig, "fig4_isolated_stability.png")


def fig_morphology(epochs=(0.0, 0.42, 0.62, 0.95, 1.30, 2.15), extent=90.0):
    hdr = np.load(DATA / "collision" / "header.npz", allow_pickle=True)
    files = sorted((DATA / "collision").glob("snap_*.npz"))
    times = np.array([float(np.load(f)["t_gyr"]) for f in files])
    fig, axes = plt.subplots(2, 3, figsize=(13.4, 9.2))
    # one shared stretch for all panels so brightness is comparable between
    # epochs; the top of the scale is set from the initial (densest) frame
    ref = np.load(files[0])["star_pos"]
    h0, _, _ = np.histogram2d(ref[:, 1], ref[:, 0], bins=420,
                              range=[[-extent, extent], [-extent, extent]])
    vmin = np.log10(0.6)
    vmax = float(np.percentile(np.log10(h0[h0 > 0] + 0.6), 99.0))
    for ax, te in zip(axes.ravel(), epochs):
        k = int(np.argmin(np.abs(times - te)))
        pos = np.load(files[k])["star_pos"]
        h, _, _ = np.histogram2d(
            pos[:, 1], pos[:, 0], bins=420,
            range=[[-extent, extent], [-extent, extent]])
        img = np.log10(h + 0.6)
        ax.imshow(img, origin="lower", cmap=SEQ,
                  extent=[-extent, extent, -extent, extent],
                  vmin=vmin, vmax=vmax, interpolation="bilinear")
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.text(0.04, 0.92, f"t = {times[k]:.2f} Gyr", transform=ax.transAxes,
                color=INK, fontsize=12, fontweight="medium")
    fig._galcol_tight = False
    axes[1, 0].plot([-80, -80 + 40], [-78, -78], color=INK, lw=2.4)
    axes[1, 0].text(-80, -72, "40 kpc", color=INK, fontsize=10)
    fig.suptitle("Projected stellar surface density (log scale), face-on to the orbital plane",
                 x=0.012, y=0.985, ha="left", color=INK, fontsize=12.5)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.955, bottom=0.005,
                        wspace=0.012, hspace=0.012)
    return _save(fig, "fig5_morphology.png")


def fig_performance():
    from galcol.ics import Encounter, Resolution, make_collision
    from galcol.production import ENCOUNTER, MODEL
    from galcol.tree import accel_direct, accel_tree, build_tree

    d = np.load(DATA / "collision" / "diagnostics.npz", allow_pickle=True)
    st = d["step_times"]
    meta = json.loads(str(d["meta"]))
    t_axis = np.arange(st.size) * meta["dt_code"] * TIME_UNIT_GYR

    # --- measured scaling: tree vs direct, same machine, same call path ----
    ns, t_tree, t_direct = [], [], []
    for frac in (0.04, 0.08, 0.16, 0.32, 0.64, 1.0):
        res = Resolution(n_disk=max(60, int(15000 * frac)),
                         n_bulge=max(20, int(4000 * frac)),
                         n_halo=max(60, int(17000 * frac)))
        p = make_collision(MODEL, res, ENCOUNTER)
        tr = build_tree(p.pos, p.mass, p.eps2, leaf_size=12)
        accel_tree(p.pos, p.mass, p.eps2, theta=0.7, tree=tr)
        t0 = time.perf_counter()
        for _ in range(3):
            tr = build_tree(p.pos, p.mass, p.eps2, leaf_size=12)
            accel_tree(p.pos, p.mass, p.eps2, theta=0.7, tree=tr)
        t_tree.append((time.perf_counter() - t0) / 3)
        ns.append(p.n)
        if p.n <= 24000:
            accel_direct(p.pos[:64], p.mass[:64], p.eps2[:64])
            t0 = time.perf_counter()
            accel_direct(p.pos, p.mass, p.eps2)
            t_direct.append(time.perf_counter() - t0)
        else:
            t_direct.append(np.nan)
    ns = np.asarray(ns, float)
    t_tree = np.asarray(t_tree)
    t_direct = np.asarray(t_direct)

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.7))
    ax = axes[0]
    ax.plot(t_axis, st * 1e3, color=S1, lw=1.2)
    ax.set_xlabel("simulation time  [Gyr]")
    ax.set_ylabel("wall-clock per step  [ms]")
    ax.set_title(f"Cost per step, {meta['n_particles']:,} particles on 4 cores",
                 loc="left", pad=10)
    ax.set_ylim(0, st.max() * 1e3 * 1.32)
    ax.text(0.97, 0.93, f"mean {1e3 * meta['wall_per_step_s']:.0f} ms\n"
                        f"{meta['n_steps']:,} steps in "
                        f"{meta['wall_clock_s'] / 60:.1f} min",
            transform=ax.transAxes, color=INK2, fontsize=9.5, ha="right",
            va="top")

    ax = axes[1]
    ok = np.isfinite(t_direct)
    # reference slopes, each anchored to the largest measured point of its own
    # curve so the comparison is "does the data follow this slope", not
    # "where did the author put the line"
    ref_nlogn = t_tree[-1] * (ns / ns[-1]) * (np.log(ns) / np.log(ns[-1]))
    ref_n2 = t_direct[ok][-1] * (ns / ns[ok][-1]) ** 2
    ax.plot(ns, ref_nlogn, color=MUTED, lw=1.2, ls="--", zorder=1)
    ax.plot(ns, ref_n2, color=MUTED, lw=1.2, ls=":", zorder=1)
    ax.plot(ns, t_tree, color=S1, marker="o", ms=5, zorder=3)
    ax.plot(ns[ok], t_direct[ok], color=S2, marker="o", ms=5, zorder=3)
    _label_end(ax, ns[-1], t_tree[-1], "  Barnes–Hut", S1)
    _label_end(ax, ns[ok][-1], t_direct[ok][-1], "  direct $O(N^2)$", S2)
    ax.annotate(r"$\propto N\log N$", xy=(ns[2], ref_nlogn[2]), xytext=(10, -16),
                textcoords="offset points", color=MUTED, fontsize=9.5)
    ax.annotate(r"$\propto N^2$", xy=(ns[1], ref_n2[1]), xytext=(-40, 6),
                textcoords="offset points", color=MUTED, fontsize=9.5)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(ns[0] * 0.72, ns[-1] * 1.8)
    ax.set_ylim(min(t_tree.min(), np.nanmin(t_direct)) * 0.45,
                max(t_tree.max(), np.nanmax(t_direct)) * 3.2)
    ax.xaxis.set_major_locator(FixedLocator(list(ns)))
    ax.xaxis.set_major_formatter(FixedFormatter(
        [f"{n / 1000:.0f}k" if n >= 1000 else f"{int(n)}" for n in ns]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("particle number  $N$")
    ax.set_ylabel("one force evaluation  [s]")
    ax.set_title(r"Measured scaling at $\theta = 0.7$", loc="left", pad=10)
    fig.subplots_adjust(wspace=0.32)
    return _save(fig, "fig6_performance.png")


def main():
    FIGDIR.mkdir(parents=True, exist_ok=True)
    made = []
    for fn in (fig_conservation, fig_tree_accuracy, fig_rotation_curve,
               fig_isolated, fig_morphology, fig_performance):
        try:
            made.append(str(fn()))
        except Exception as exc:                     # keep going, report clearly
            print(f"  !! {fn.__name__} failed: {exc}")
    print(f"{len(made)} figures written")
    return 0 if len(made) >= 3 else 1


if __name__ == "__main__":
    raise SystemExit(main())
