"""
Integration tests for the SCF engine.
Tests against known literature values.

H2O cc-pVDZ: expected RHF energy ≈ -76.0268 Ha (literature)
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scf_engine import run_rhf, parse_xyz_block, strip_internal

# Standard H2O geometry (Å, experimental equilibrium)
H2O_XYZ = """
O  0.000000  0.000000  0.117176
H  0.000000  0.757001 -0.468704
H  0.000000 -0.757001 -0.468704
""".strip()

# Literature RHF/cc-pVDZ energy for H2O (Hartree)
H2O_CCPVDZ_REF = -76.0268


class TestParseXYZ:
    def test_parse_raw_atoms(self):
        atoms = parse_xyz_block(H2O_XYZ)
        assert len(atoms) == 3
        assert atoms[0][0] == "O"
        assert atoms[1][0] == "H"

    def test_parse_full_xyz_format(self):
        full = "3\nwater molecule\n" + H2O_XYZ
        atoms = parse_xyz_block(full)
        assert len(atoms) == 3

    def test_parse_empty_raises(self):
        with pytest.raises(ValueError, match="No valid atom"):
            parse_xyz_block("")

    def test_parse_bad_coords_raises(self):
        with pytest.raises(ValueError):
            parse_xyz_block("O 0.0 notanumber 0.0")


class TestRHFWater:
    """H2O RHF calculations — requires PySCF installed."""

    def test_h2o_ccpvdz_energy(self):
        """RHF/cc-pVDZ energy should match literature within 1 mHa."""
        result = run_rhf(H2O_XYZ, zeta_level="D", calendar_prefix=None, base_family="cc-pV")

        assert result.get("converged"), "SCF did not converge"
        energy = result["total_energy"]
        assert abs(energy - H2O_CCPVDZ_REF) < 0.001, (
            f"Energy {energy:.6f} Ha deviates >1 mHa from literature {H2O_CCPVDZ_REF} Ha"
        )

    def test_h2o_converged(self):
        result = run_rhf(H2O_XYZ, zeta_level="D", calendar_prefix=None, base_family="cc-pV")
        assert result["converged"] is True

    def test_h2o_homo_lumo(self):
        result = run_rhf(H2O_XYZ, zeta_level="D", calendar_prefix=None, base_family="cc-pV")
        assert result["homo_energy"] is not None
        assert result["lumo_energy"] is not None
        # HOMO of water is negative (bound state), LUMO may be positive
        assert result["homo_energy"] < 0.0

    def test_h2o_dipole(self):
        """H2O dipole moment should be ~1.85 Debye (experimental ~1.85 D)."""
        result = run_rhf(H2O_XYZ, zeta_level="D", calendar_prefix=None, base_family="cc-pV")
        dip = result["dipole"]["total"]
        # RHF/cc-pVDZ overestimates slightly, expect 1.6–2.2 D
        assert 1.5 < dip < 2.5, f"Dipole {dip:.3f} D out of expected range"

    def test_h2o_orbital_count(self):
        """cc-pVDZ gives 24 basis functions for H2O."""
        result = run_rhf(H2O_XYZ, zeta_level="D", calendar_prefix=None, base_family="cc-pV")
        assert result["n_basis"] == 24

    def test_h2o_electron_count(self):
        result = run_rhf(H2O_XYZ, zeta_level="D", calendar_prefix=None, base_family="cc-pV")
        assert result["n_electrons"] == 10

    def test_h2o_orbital_labels(self):
        result = run_rhf(H2O_XYZ, zeta_level="D", calendar_prefix=None, base_family="cc-pV")
        orb_labels = [o["label"] for o in result["orbitals"]]
        assert "HOMO" in orb_labels
        assert "LUMO" in orb_labels

    def test_strip_internal(self):
        result = run_rhf(H2O_XYZ, zeta_level="D", calendar_prefix=None, base_family="cc-pV")
        stripped = strip_internal(result)
        assert "_mol" not in stripped
        assert "_mf" not in stripped


class TestRHFCalendarBasis:
    def test_h2o_aug_ccpvdz(self):
        """aug-cc-pVDZ should give lower energy than cc-pVDZ (more functions)."""
        base_result = run_rhf(H2O_XYZ, zeta_level="D", calendar_prefix=None, base_family="cc-pV")
        aug_result = run_rhf(H2O_XYZ, zeta_level="D", calendar_prefix="aug", base_family="cc-pV")

        assert aug_result["converged"]
        # aug basis has more functions → lower (more negative) energy by variational principle
        assert aug_result["total_energy"] < base_result["total_energy"]

    def test_h2o_jul_ccpvdz(self):
        """jul-cc-pVDZ energy should be between cc-pVDZ and aug-cc-pVDZ."""
        base = run_rhf(H2O_XYZ, zeta_level="D", calendar_prefix=None, base_family="cc-pV")
        aug = run_rhf(H2O_XYZ, zeta_level="D", calendar_prefix="aug", base_family="cc-pV")
        jul = run_rhf(H2O_XYZ, zeta_level="D", calendar_prefix="jul", base_family="cc-pV")

        assert jul["converged"]
        # jul should have energy between base and aug
        assert aug["total_energy"] <= jul["total_energy"] <= base["total_energy"] + 1e-6


class TestRHFErrors:
    def test_invalid_xyz(self):
        result = run_rhf("not valid xyz", zeta_level="D")
        assert "error" in result

    def test_unknown_basis(self):
        result = run_rhf(H2O_XYZ, zeta_level="D", base_family="unknown-family")
        assert "error" in result
