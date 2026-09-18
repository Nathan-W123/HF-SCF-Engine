#!/usr/bin/env python3
"""Render ``media/hero.png`` and ``media/hero.mp4`` from the simulated entry.

Both are produced by :mod:`reentry.render.hero` from ``results/traj_ballistic.npz``
-- the trajectory, the heat flux and the atmospheric density profile are the ones
the simulator computed; see the README section "How the hero visual is derived
from the physics".

``--quick`` renders a short, half-length video for smoke testing.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from _bootstrap import MEDIA_DIR, RESULTS_DIR, load_trajectory  # noqa: E402

from reentry.dynamics import state_to_cartesian  # noqa: E402
from reentry.render import CameraPath, HeroScene  # noqa: E402
from reentry.render.compose import open_writer, tonemap  # noqa: E402

# --- the shot -------------------------------------------------------------
T_START = 30.0  # s after the 120 km entry interface: heating has just begun
T_END = 195.0  # s: the fireball has faded and the capsule is subsonic
FPS = 30
DURATION_S = 18.0

CAMERA = CameraPath(
    start_offset=(1.25e6, 0.45e6, 3.60e6),
    end_offset=(0.72e6, 0.05e6, 2.50e6),
    start_fov_deg=17.5,
    end_fov_deg=12.5,
    start_roll_deg=-27.0,
    end_roll_deg=-16.0,
    start_aim_blend=0.58,
    end_aim_blend=0.68,
    start_yaw_deg=-7.0,
    end_yaw_deg=3.0,
    anchor_lag_s=16.0,
)

SCENE_KW = dict(
    sun_intensity=2.6,
    sun_elevation_deg=-11.0,
    sun_azimuth_deg=-78.0,
    ambient_night=0.0035,
    city_gain=0.12,
    wake_halo_gain=2.8,
    persistence_s=55.0,
    head_gain=55.0,
    n_stars=8000,
)


def build_scene():
    tr = load_trajectory(RESULTS_DIR / "traj_ballistic.npz")
    pos, _ = state_to_cartesian(tr.radius, tr.lon, tr.lat, tr.velocity, tr.gamma,
                                tr.psi)
    scene = HeroScene(
        positions=pos, times=tr.t, q_dot=tr.q_dot, altitude=tr.altitude,
        camera=CAMERA, **SCENE_KW,
    )
    return tr, scene


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="short, lower-quality video for smoke testing")
    ap.add_argument("--still-only", action="store_true", help="skip the video")
    args = ap.parse_args()

    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    tr, scene = build_scene()

    # ---------------- hero still: the peak-heating instant ----------------
    i_peak = int(np.argmax(tr.q_dot))
    t_peak = float(tr.t[i_peak])
    s_peak = float(np.clip((t_peak - T_START) / (T_END - T_START), 0.0, 1.0))
    t0 = time.time()
    still = scene.render_image(
        t_peak, s_peak, 1920, 1080, n_steps=72, supersample=2,
    )
    from PIL import Image

    Image.fromarray(still).save(MEDIA_DIR / "hero.png")
    print(
        f"hero.png  t = {t_peak:.1f} s, altitude = {tr.altitude[i_peak] / 1e3:.1f} km, "
        f"V = {tr.velocity[i_peak] / 1e3:.2f} km/s, qdot = "
        f"{tr.q_dot[i_peak] / 1e4:.1f} W/cm^2, T_wall = "
        f"{tr.wall_temperature[i_peak]:.0f} K   [{time.time() - t0:.1f} s, 2x SSAA]"
    )

    if args.still_only:
        return 0

    # ---------------- hero video ----------------
    duration = DURATION_S * (0.25 if args.quick else 1.0)
    n_frames = int(round(duration * FPS))
    steps = 16 if args.quick else 24
    path = MEDIA_DIR / "hero.mp4"
    writer = open_writer(path, size=(1920, 1080), fps=FPS,
                         crf=18 if args.quick else 15,
                         preset="fast" if args.quick else "slow")
    t0 = time.time()
    try:
        from tqdm import tqdm

        it = tqdm(range(n_frames), desc="hero.mp4", unit="frame")
    except Exception:  # pragma: no cover
        it = range(n_frames)
    for k in it:
        s = k / max(n_frames - 1, 1)
        t_sim = T_START + (T_END - T_START) * s
        buf, _ = scene.render_frame(t_sim, s, 1920, 1080, n_steps=steps)
        writer.send(np.ascontiguousarray(tonemap(buf)))
    writer.close()
    dt = time.time() - t0
    print(f"hero.mp4  {n_frames} frames, {duration:.0f} s at {FPS} fps "
          f"[{dt:.0f} s total, {dt / n_frames:.2f} s/frame]")

    (MEDIA_DIR / "hero_manifest.json").write_text(json.dumps({
        "source_trajectory": "results/traj_ballistic.npz",
        "still_time_s": t_peak,
        "still_altitude_km": float(tr.altitude[i_peak] / 1e3),
        "still_velocity_m_s": float(tr.velocity[i_peak]),
        "still_heat_flux_W_cm2": float(tr.q_dot[i_peak] / 1e4),
        "still_wall_temperature_K": float(tr.wall_temperature[i_peak]),
        "video_window_s": [T_START, T_END],
        "video_duration_s": duration,
        "video_fps": FPS,
        "video_frames": n_frames,
        "resolution": [1920, 1080],
        "camera": CAMERA.__dict__,
        "scene": SCENE_KW,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
