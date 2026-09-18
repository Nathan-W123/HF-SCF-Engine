"""Rendering subpackage: analytic Earth/atmosphere ray marching, additive
splatting, bloom/tone mapping and the hero scene composer."""

from .compose import bloom, open_writer, splat, tonemap
from .hero import CameraPath, HeroScene
from .raymarch import RAYLEIGH_BETA, build_noise_volumes, render_background

__all__ = [
    "splat", "bloom", "tonemap", "open_writer",
    "HeroScene", "CameraPath",
    "render_background", "build_noise_volumes", "RAYLEIGH_BETA",
]
