"""Entry-vehicle definition."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .aerodynamics import SphereCone

__all__ = ["Vehicle", "GENERIC_CAPSULE"]


@dataclass
class Vehicle:
    """A point-mass entry vehicle with a sphere-cone aeroshell.

    Attributes
    ----------
    mass : float
        Entry mass [kg].
    geometry : SphereCone
        Forebody geometry; supplies the reference area and the Newtonian
        aerodynamic shape integrals.
    alpha_trim_rad : float
        Trim angle of attack [rad].  With the sign convention of
        :mod:`reentry.aerodynamics` (freestream at ``+alpha`` toward ``+z_body``)
        a *negative* trim angle puts the Newtonian lift vector along ``+z_body``,
        i.e. "lift up" at zero bank.  Zero for a purely ballistic entry.
    bank_rad : float
        Bank (roll) angle about the velocity vector [rad].  0 = lift vector in
        the vertical plane pointing up, pi = lift down.
    emissivity : float
        TPS surface emissivity used for the radiative-equilibrium wall
        temperature.  0.85 is a representative value for a coated ceramic /
        charred ablator surface; it is a modelling choice, documented as such.
    """

    mass: float = 1200.0
    geometry: SphereCone = field(default_factory=SphereCone)
    alpha_trim_rad: float = 0.0
    bank_rad: float = 0.0
    emissivity: float = 0.85
    name: str = "generic 70deg sphere-cone capsule"

    @property
    def area_ref(self) -> float:
        return self.geometry.area_ref

    @property
    def nose_radius(self) -> float:
        return self.geometry.nose_radius

    def aero(self, mach):
        """Return ``(C_D, C_L)`` at the trim angle of attack."""
        return self.geometry.coefficients(self.alpha_trim_rad, mach)

    def ballistic_coefficient(self, mach=25.0) -> float:
        """``m / (C_D A)`` [kg/m^2] at the given Mach number."""
        cd, _ = self.aero(mach)
        return float(self.mass / (cd * self.area_ref))

    def lift_to_drag(self, mach=25.0) -> float:
        cd, cl = self.aero(mach)
        return float(cl / cd)

    def describe(self) -> dict:
        d = self.geometry.describe()
        d.update(
            {
                "name": self.name,
                "mass_kg": self.mass,
                "alpha_trim_deg": float(np.degrees(self.alpha_trim_rad)),
                "bank_deg": float(np.degrees(self.bank_rad)),
                "emissivity": self.emissivity,
                "ballistic_coefficient_mach25_kg_m2": self.ballistic_coefficient(25.0),
                "lift_to_drag_mach25": self.lift_to_drag(25.0),
            }
        )
        return d


#: Default vehicle used across the project.
GENERIC_CAPSULE = Vehicle()
