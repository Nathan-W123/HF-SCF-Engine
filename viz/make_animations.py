#!/usr/bin/env python3
"""
Three short square videos built from real Hartree-Fock output.

    A  orbital-rotation   a converged MO turning in space, seamless loop
    B  scf-convergence    the density settling cycle by cycle, with the trace
    C  orbital-switch     density -> electrostatic potential -> HOMO-1/HOMO/LUMO

Every surface in these comes from backend/scf_engine.py: the SCF is run here,
the MO coefficients are evaluated on a grid, and the isosurfaces are extracted
from that grid.  Nothing is drawn by hand.

    python viz/make_animations.py                 # all three, 1080x1080
    python viz/make_animations.py --which a       # just one
    python viz/make_animations.py --preview       # small and fast, for checking
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import overlay as O            # noqa: E402
import render as R             # noqa: E402
import scene as S              # noqa: E402
from fields import (Grid, ao_grid, density_field, esp_field, mo_field,  # noqa: E402
                    sample_field, HARTREE_TO_KCAL)
from scf_engine import HARTREE_TO_EV  # noqa: E402

# ── helpers ───────────────────────────────────────────────────────────────────

def lerp(a, b, t):
    return a + (b - a) * t


def fnum(x, fmt="{:.6f}"):
    """Format a number with a typographic minus (never touches other text)."""
    return fmt.format(x).replace("-", "\u2212")


def orbital_label(res, index):
    """'HOMO', 'LUMO+1', … and the eigenvalue, from the engine's own list."""
    orb = res["orbitals"][index]
    return orb["label"], orb["energy_hartree"]


def eps_text(e):
    return f"ε = {fnum(e * HARTREE_TO_EV, '{:.2f}')} eV"


class Sequencer:
    """Named, back-to-back frame ranges."""

    def __init__(self):
        self.marks: list[tuple[str, int, int]] = []
        self.n = 0

    def add(self, name, frames):
        self.marks.append((name, self.n, self.n + frames))
        self.n += frames
        return self

    def where(self, i):
        for name, a, b in self.marks:
            if a <= i < b:
                return name, (i - a) / max(1, b - a - 1), i - a, b - a
        name, a, b = self.marks[-1]
        return name, 1.0, b - a - 1, b - a


def frame_to_image(cam, layers, out_size):
    return R.to_image(R.despeckle(R.compose(cam, layers)), out_size)


# ── A. orbital rotation ───────────────────────────────────────────────────────

def animation_a(out_dir, size, ss, fps, frames, quality):
    print("\n[A] orbital rotation — benzene HOMO")
    res, _, label, _ = S.calculate("benzene", "sto-3g")
    idx = res["homo_idx"]
    name, e = orbital_label(res, idx)

    grid = Grid.around(res["atoms"], margin_ang=2.8, spacing_ang=0.10)
    ao = ao_grid(res["_bfs"], grid)
    psi = mo_field(ao, res["_C"], idx)
    del ao

    W = int(size * ss)
    cam = S.fit_camera(res["atoms"], W, pad=1.04)
    ppa = quality * cam.scale ** 2
    iso = 0.04

    pos, neg = S.surface_for(psi, grid, iso, ppa)
    cam = S.fit_camera(res["atoms"], W, pad=1.12,
                       extra_points=[p.pos for p in (pos, neg) if p is not None])
    ppa = quality * cam.scale ** 2
    pos, neg = S.surface_for(psi, grid, iso, ppa)
    pts, cols = R.SurfacePoints.concat([(pos, R.PHASE_POS), (neg, R.PHASE_NEG)])
    print(f"  {pts.n / 1e6:.1f}M surface points, iso = ±{iso}")

    tilt = 0.52
    path = os.path.join(out_dir, "A_orbital_rotation.mp4")

    subtitle = f"{label} · RHF/STO-3G · {eps_text(e)}"

    def paint(img):
        # No edge scrim: the backdrop is flat, so the caption is already legible
        # and a gradient over a flat field would read as a smudge.
        layer, draw = O.draw_on(img, shade_edges=False)
        O.caption(draw, size, size, name, subtitle)
        return O.flatten(img, layer)

    with S.Video(path, fps=fps, poster_at=frames // 6) as vid:
        for i in range(frames):
            cam.R = R.rotation(2.0 * np.pi * i / frames, tilt)
            layers = [R.render_molecule(cam, res["atoms"]),
                      R.splat(cam, pts, None, vertex_rgb=cols)]
            R.fill_holes(layers[-1], 1)
            vid.add(paint(frame_to_image(cam, layers, size)), frames)
    return path


# ── B. SCF convergence ────────────────────────────────────────────────────────

def animation_b(out_dir, size, ss, fps, frames, quality):
    print("\n[B] SCF convergence — water")
    from scf_engine import compute_one_electron, parse_xyz_block, ANGSTROM_TO_BOHR

    res, hist, label, _ = S.calculate("water", "6-31g*", trace=True)
    xyz = S.MOLECULES["water"][0]
    atoms_bohr = [(s, x * ANGSTROM_TO_BOHR, y * ANGSTROM_TO_BOHR, z * ANGSTROM_TO_BOHR)
                  for s, x, y, z in parse_xyz_block(xyz)]
    Sm, _, _ = compute_one_electron(res["_bfs"], atoms_bohr, [0] * len(atoms_bohr))
    S.align_phases(hist, Sm)

    n_occ = res["homo_idx"] + 1
    grid = Grid.around(res["atoms"], margin_ang=2.7, spacing_ang=0.085)
    ao = ao_grid(res["_bfs"], grid)

    rho = [density_field(ao, h["C"], n_occ) for h in hist]
    rho_final = rho[-1]
    err = [r - rho_final for r in rho]
    psi_homo = mo_field(ao, res["_C"], res["homo_idx"])
    del ao

    energies = [h["energy"] for h in hist]
    deltas = [h["delta"] for h in hist]
    cycles = [h["cycle"] for h in hist]
    n_cyc = len(hist)

    RHO_ISO, ERR_ISO, MO_ISO = 0.055, 0.012, 0.08

    W = int(size * ss)
    probe = R.SurfacePoints.from_field(err[0], grid, ERR_ISO, points_per_A2=400.0)
    probe2 = R.SurfacePoints.from_field(psi_homo, grid, MO_ISO, points_per_A2=400.0)
    cam_wide = S.fit_camera(res["atoms"], W, pad=1.16,
                            extra_points=[probe.pos] if probe is not None else ())
    cam_tight = S.fit_camera(res["atoms"], W, pad=1.34,
                             extra_points=[probe2.pos] if probe2 is not None else ())
    r_wide, r_tight = cam_wide.radius, cam_tight.radius
    cam = cam_wide
    # Sample for the tightest framing, since the camera zooms in over the clip
    # and point density per pixel falls as the scale grows.
    ppa = quality * cam_tight.scale ** 2

    # Cycle timing: the visible change is over within a handful of cycles, so
    # give the early ones room and sweep the polishing tail.
    weights = np.array([1.0 / (1.0 + 0.55 * k) for k in range(n_cyc)])
    budget = int(frames * 0.60)
    per = np.maximum(4, np.round(weights / weights.sum() * budget)).astype(int)

    seq = Sequencer().add("intro", int(frames * 0.07))
    for k in range(n_cyc):
        seq.add(f"cycle{k}", int(per[k]))
    seq.add("converged", int(frames * 0.11))
    seq.add("reveal", int(frames * 0.10))
    seq.add("homo", max(24, frames - seq.n - int(frames * 0.10)))
    total = seq.n
    print(f"  {n_cyc} cycles, {total} frames")

    # Surfaces, one set per cycle (interpolated between neighbours while morphing).
    print("  extracting isosurfaces ...")
    surf_rho = [R.SurfacePoints.from_field(r, grid, RHO_ISO, points_per_A2=ppa)
                for r in rho]
    surf_err = []
    for e in err:
        p, n = S.surface_for(e, grid, ERR_ISO, ppa)
        surf_err.append(R.SurfacePoints.concat([(p, R.PHASE_POS), (n, R.PHASE_NEG)]))
    hp, hn = S.surface_for(psi_homo, grid, MO_ISO, ppa)
    surf_homo = R.SurfacePoints.concat([(hp, R.PHASE_POS), (hn, R.PHASE_NEG)])

    path = os.path.join(out_dir, "B_scf_convergence.mp4")
    plot_box = (int(0.520 * size), int(0.055 * size),
                int(0.945 * size), int(0.290 * size))

    with S.Video(path, fps=fps, poster_at=int(total * 0.35)) as vid:
        for i in range(total):
            stage, u, _, _ = seq.where(i)
            prog = i / max(1, total - 1)
            cam.R = R.rotation(-0.55 + 1.15 * prog, 0.30)
            # Pull in as the error surfaces shrink away.
            zoom = O.ease(np.clip((prog - 0.10) / 0.55, 0.0, 1.0))
            cam.radius = lerp(r_wide, r_tight, zoom)

            k = 0
            show_err = 0.0
            rho_pts = surf_rho[0]
            err_pts, err_cols = surf_err[0]
            fade_in = 1.0

            if stage == "intro":
                fade_in = O.ease(u)
                show_err = fade_in
            elif stage.startswith("cycle"):
                k = int(stage[5:])
                rho_pts = surf_rho[k]
                err_pts, err_cols = surf_err[k]
                show_err = 1.0
            else:
                k = n_cyc - 1
                rho_pts = surf_rho[k]
                err_pts, err_cols = surf_err[k]
                show_err = 0.0

            layers = [R.render_molecule(cam, res["atoms"])]

            if stage in ("reveal", "homo"):
                t = O.ease(u) if stage == "reveal" else 1.0
                lr = R.splat(cam, rho_pts, R.DENSITY_COLOR, alpha=0.34 * (1.0 - t))
                if not lr.empty:
                    R.fill_holes(lr, 1)
                    layers.append(lr)
                lh = R.splat(cam, surf_homo[0], None, vertex_rgb=surf_homo[1],
                             alpha=0.88 * t)
                R.fill_holes(lh, 1)
                layers.append(lh)
            else:
                lr = R.splat(cam, rho_pts, R.DENSITY_COLOR, alpha=0.34 * fade_in)
                if not lr.empty:
                    R.fill_holes(lr, 1)
                    layers.append(lr)
                if err_pts is not None and show_err > 0.01:
                    le = R.splat(cam, err_pts, None, vertex_rgb=err_cols,
                                 alpha=0.90 * show_err)
                    R.fill_holes(le, 1)
                    layers.append(le)

            img = frame_to_image(cam, layers, size)
            layer, draw = O.draw_on(img)

            shown = min(k + 1, n_cyc)
            method = f"{label} · RHF/6-31G*"
            if stage in ("reveal", "homo"):
                t = O.ease(u) if stage == "reveal" else 1.0
                O.convergence_panel(draw, plot_box, cycles, deltas, upto=n_cyc,
                                    alpha=1.0 - t, ylabel="max |ΔP|",
                                    value_text=f"{deltas[-1]:.1e}")
                nm, e = orbital_label(res, res["homo_idx"])
                O.caption_switch(
                    draw, size, size,
                    dict(title=f"{fnum(energies[-1])} Ha",
                         subtitle=f"converged in {n_cyc} cycles  ·  {method}"),
                    dict(title=nm, subtitle=f"{method} · {eps_text(e)}"),
                    t if stage == "reveal" else 1.0)
            else:
                O.convergence_panel(draw, plot_box, cycles, deltas, upto=shown,
                                    ylabel="max |ΔP|",
                                    value_text=f"{deltas[shown - 1]:.1e}")
                if stage == "converged":
                    O.caption(draw, size, size, f"{fnum(energies[-1])} Ha",
                              f"converged in {n_cyc} cycles  ·  {method}")
                else:
                    O.caption(draw, size, size, f"{fnum(energies[shown - 1])} Ha",
                              f"SCF cycle {shown} of {n_cyc}  ·  {method}")

            vid.add(O.flatten(img, layer), total)
    return path


# ── C. orbital switch ─────────────────────────────────────────────────────────

def animation_c(out_dir, size, ss, fps, frames, quality):
    print("\n[C] orbital switch — benzene")
    res, _, label, _ = S.calculate("benzene", "sto-3g")
    homo = res["homo_idx"]
    n_occ = homo + 1

    grid = Grid.around(res["atoms"], margin_ang=3.0, spacing_ang=0.10)
    ao = ao_grid(res["_bfs"], grid)
    rho = density_field(ao, res["_C"], n_occ)
    fields = {k: mo_field(ao, res["_C"], i)
              for k, i in (("homo1", homo - 1), ("homo", homo), ("lumo", homo + 1))}
    del ao

    print("  solving Poisson equation for the electrostatic potential ...")
    esp = esp_field(rho.astype(np.float64), grid, res["atoms"]) * HARTREE_TO_KCAL

    W = int(size * ss)
    MO_ISO = 0.04
    RHO_TIGHT, RHO_VDW = 0.09, 0.002

    # Each object in the sequence has a different extent, so the camera is
    # framed per segment and eased between them; sampling density follows the
    # framing so a surface is never under-sampled for the scale it is shown at.
    def probe_pts(field, iso):
        return R.SurfacePoints.from_field(field, grid, iso, points_per_A2=300.0)

    p_rho = probe_pts(rho, RHO_TIGHT)
    p_vdw = probe_pts(rho, RHO_VDW)
    p_mo = [probe_pts(f, MO_ISO) for f in fields.values()]

    r_rho = S.fit_camera(res["atoms"], W, pad=1.30, extra_points=[p_rho.pos]).radius
    r_vdw = S.fit_camera(res["atoms"], W, pad=1.26, extra_points=[p_vdw.pos]).radius
    r_mo = S.fit_camera(res["atoms"], W, pad=1.26,
                        extra_points=[q.pos for q in p_mo if q is not None]).radius
    cam = S.fit_camera(res["atoms"], W, pad=1.26, extra_points=[p_vdw.pos])

    def ppa_at(radius):
        return quality * (0.5 * W / radius) ** 2

    print("  extracting isosurfaces ...")
    ppa = ppa_at(min(r_rho, r_mo))
    rho_tight = R.SurfacePoints.from_field(rho, grid, RHO_TIGHT,
                                           points_per_A2=ppa_at(r_rho))
    rho_vdw = R.SurfacePoints.from_field(rho, grid, RHO_VDW,
                                         points_per_A2=ppa_at(r_vdw))

    esp_on_vdw = sample_field(esp, grid, rho_vdw.pos * (1.0 / 0.529177210903))
    vmax = float(np.percentile(np.abs(esp_on_vdw), 97))
    esp_cols = S.esp_colors(esp_on_vdw, vmax)
    teal_cols = np.broadcast_to(np.asarray(R.DENSITY_COLOR, np.float32),
                                (rho_vdw.n, 3)).copy()
    print(f"  ESP on ρ = {RHO_VDW} a.u. surface: ±{vmax:.0f} kcal/mol/e")

    mo_surf = {}
    for key, f in fields.items():
        p, n = S.surface_for(f, grid, MO_ISO, ppa_at(r_mo))
        mo_surf[key] = R.SurfacePoints.concat([(p, R.PHASE_POS), (n, R.PHASE_NEG)])

    MO_INFO = {"homo1": homo - 1, "homo": homo, "lumo": homo + 1}

    f = frames / 440.0
    seq = (Sequencer()
           .add("rho", int(58 * f)).add("inflate", int(34 * f))
           .add("esp", int(86 * f)).add("to_homo1", int(34 * f))
           .add("homo1", int(46 * f)).add("to_homo", int(30 * f))
           .add("homo", int(50 * f)).add("to_lumo", int(30 * f))
           .add("lumo", int(72 * f)))
    total = seq.n

    method = f"{label} · RHF/STO-3G"

    def mo_title(key):
        nm, e = orbital_label(res, MO_INFO[key])
        return dict(title=nm, subtitle=f"{method} · {eps_text(e)}")

    T_RHO = dict(title="Electron density",
                 subtitle=f"{method} · {res['n_electrons']} electrons")
    T_ESP = dict(title="Electrostatic potential",
                 subtitle="on the ρ = 0.002 a.u. surface")
    CAPTION_HOLD = {"rho": T_RHO, "esp": T_ESP, "homo1": mo_title("homo1"),
                    "homo": mo_title("homo"), "lumo": mo_title("lumo")}
    CAPTION_MOVE = {"inflate": (T_RHO, T_ESP),
                    "to_homo1": (T_ESP, mo_title("homo1")),
                    "to_homo": (mo_title("homo1"), mo_title("homo")),
                    "to_lumo": (mo_title("homo"), mo_title("lumo"))}

    path = os.path.join(out_dir, "C_orbital_switch.mp4")
    bar_box = (int(0.600 * size), int(0.128 * size),
               int(0.925 * size), int(0.158 * size))

    with S.Video(path, fps=fps, poster_at=int(total * 0.22)) as vid:
        for i in range(total):
            stage, u, _, _ = seq.where(i)
            cam.R = R.rotation(0.45 + 2.0 * np.pi * 1.25 * i / total, 0.46)
            cam.radius = {
                "rho":      lambda t: r_rho,
                "inflate":  lambda t: lerp(r_rho, r_vdw, O.ease(t)),
                "esp":      lambda t: r_vdw,
                "to_homo1": lambda t: lerp(r_vdw, r_mo, O.ease(t)),
            }.get(stage, lambda t: r_mo)(u)
            layers = [R.render_molecule(cam, res["atoms"])]

            mo_alpha = {"homo1": 0.0, "homo": 0.0, "lumo": 0.0}
            surf_pts, surf_cols, surf_alpha = None, None, 0.0
            esp_mix, bar_alpha = 0.0, 0.0

            if stage == "rho":
                surf_pts, surf_cols, surf_alpha = rho_tight, R.DENSITY_COLOR, 0.80
            elif stage == "inflate":
                t = O.ease(u)
                iso = float(np.exp(lerp(np.log(RHO_TIGHT), np.log(RHO_VDW), t)))
                surf_pts = R.SurfacePoints.from_field(
                    rho, grid, iso, points_per_A2=ppa_at(cam.radius))
                mix = O.ease(max(0.0, (t - 0.35) / 0.65))
                cols = S.esp_colors(
                    sample_field(esp, grid, surf_pts.pos * (1.0 / 0.529177210903)), vmax)
                surf_cols = lerp(np.broadcast_to(
                    np.asarray(R.DENSITY_COLOR, np.float32), cols.shape), cols, mix)
                surf_alpha, esp_mix, bar_alpha = 0.82, mix, mix
            elif stage == "esp":
                surf_pts, surf_cols = rho_vdw, esp_cols
                surf_alpha, esp_mix, bar_alpha = 0.86, 1.0, 1.0
            elif stage == "to_homo1":
                t = O.ease(u)
                surf_pts, surf_cols = rho_vdw, esp_cols
                surf_alpha, esp_mix = 0.86 * (1.0 - t), 1.0
                bar_alpha = 1.0 - t
                mo_alpha["homo1"] = 0.88 * t
            elif stage in ("homo1", "homo", "lumo"):
                mo_alpha[stage] = 0.88
            elif stage == "to_homo":
                t = O.ease(u)
                mo_alpha["homo1"], mo_alpha["homo"] = 0.88 * (1 - t), 0.88 * t
            elif stage == "to_lumo":
                t = O.ease(u)
                mo_alpha["homo"], mo_alpha["lumo"] = 0.88 * (1 - t), 0.88 * t

            if surf_pts is not None and surf_alpha > 0.01:
                l = R.splat(cam, surf_pts,
                            surf_cols if isinstance(surf_cols, tuple) else None,
                            vertex_rgb=None if isinstance(surf_cols, tuple) else surf_cols,
                            alpha=surf_alpha, rim=0.24 if esp_mix > 0.5 else 0.30)
                R.fill_holes(l, 1)
                layers.append(l)
            for key, a in mo_alpha.items():
                if a > 0.01:
                    pts, cols = mo_surf[key]
                    l = R.splat(cam, pts, None, vertex_rgb=cols, alpha=a)
                    R.fill_holes(l, 1)
                    layers.append(l)

            img = frame_to_image(cam, layers, size)
            layer, draw = O.draw_on(img)
            if stage in CAPTION_HOLD:
                O.caption(draw, size, size, **CAPTION_HOLD[stage])
            else:
                before, after = CAPTION_MOVE[stage]
                O.caption_switch(draw, size, size, before, after, u)

            if bar_alpha > 0.02:
                pad_x, pad_y = int(0.026 * size), int(0.046 * size)
                O.panel(draw, (bar_box[0] - pad_x, bar_box[1] - pad_y,
                               bar_box[2] + pad_x, bar_box[3] + pad_y),
                        alpha=0.72 * bar_alpha, radius=int(0.018 * size))
                O.colorbar(draw, bar_box, S.esp_ramp, alpha=bar_alpha,
                           title="kcal/mol",
                           labels=(f"−{vmax:.0f}", f"+{vmax:.0f}"),
                           ticks=[(0.5, "0")])

            vid.add(O.flatten(img, layer), total)
    return path


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--which", nargs="+", default=["a", "b", "c"],
                    choices=["a", "b", "c"])
    ap.add_argument("--out", default=os.path.join(_HERE, "..", "renders"))
    ap.add_argument("--size", type=int, default=1080, help="output edge, px")
    ap.add_argument("--ss", type=float, default=1.4, help="supersampling factor")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--quality", type=float, default=4.5,
                    help="surface points per pixel of projected area")
    ap.add_argument("--preview", action="store_true",
                    help="small and short, for checking composition")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(message)s")
    logging.getLogger("viz").setLevel(logging.INFO)

    if args.preview:
        args.size, args.ss, args.quality = 480, 1.3, 2.5

    os.makedirs(args.out, exist_ok=True)
    plan = {
        "a": (animation_a, 80 if args.preview else 240),
        "b": (animation_b, 90 if args.preview else 360),
        "c": (animation_c, 110 if args.preview else 440),
    }
    t0 = time.time()
    made = []
    for key in args.which:
        fn, frames = plan[key]
        made.append(fn(args.out, args.size, args.ss, args.fps, frames, args.quality))
    print(f"\ndone in {time.time() - t0:.0f}s")
    for p in made:
        print("  " + os.path.relpath(p))
    return made


if __name__ == "__main__":
    main()
