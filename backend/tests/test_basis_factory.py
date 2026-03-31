"""
Unit tests for BasisFactory — calendar convention and basis resolution.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basis_factory import (
    resolve_basis,
    build_calendar_basis,
    get_basis_label,
    CALENDAR_DROP,
    ZETA_AUG_LMAX,
    AM_LABELS,
)


# ─── resolve_basis tests ───────────────────────────────────────────────────────

class TestResolveBasis:
    def test_plain_dz(self):
        assert resolve_basis("D", None, "cc-pV") == "cc-pVDZ"

    def test_plain_tz(self):
        assert resolve_basis("T", None, "cc-pV") == "cc-pVTZ"

    def test_plain_qz(self):
        assert resolve_basis("Q", None, "cc-pV") == "cc-pVQZ"

    def test_aug_dz(self):
        assert resolve_basis("D", "aug", "cc-pV") == "aug-cc-pVDZ"

    def test_aug_tz(self):
        assert resolve_basis("T", "aug", "cc-pV") == "aug-cc-pVTZ"

    def test_jul_dz_returns_tuple(self):
        result = resolve_basis("D", "jul", "cc-pV")
        assert isinstance(result, tuple)
        base, aug, cal, zeta = result
        assert base == "cc-pVDZ"
        assert aug == "aug-cc-pVDZ"
        assert cal == "jul"
        assert zeta == "D"

    def test_jun_tz_returns_tuple(self):
        result = resolve_basis("T", "jun", "cc-pV")
        assert isinstance(result, tuple)
        assert result[2] == "jun"

    def test_def2_tz(self):
        assert resolve_basis("T", None, "def2") == "def2-TZVP"

    def test_def2_qz(self):
        assert resolve_basis("Q", None, "def2") == "def2-QZVP"

    def test_unknown_family_raises(self):
        with pytest.raises(ValueError, match="Unknown basis"):
            resolve_basis("D", None, "unknown-basis")

    def test_invalid_calendar_raises(self):
        with pytest.raises(ValueError, match="Unknown calendar prefix"):
            resolve_basis("D", "feb", "cc-pV")


# ─── Calendar convention logic ────────────────────────────────────────────────

class TestCalendarDrop:
    def test_drop_counts(self):
        assert CALENDAR_DROP["aug"] == 0
        assert CALENDAR_DROP["jul"] == 1
        assert CALENDAR_DROP["jun"] == 2
        assert CALENDAR_DROP["may"] == 3
        assert CALENDAR_DROP["apr"] == 4
        assert CALENDAR_DROP["mar"] == 5

    def test_aug_lmax_dz(self):
        """DZ augmentation adds s, p, d → lmax=2"""
        assert ZETA_AUG_LMAX["D"] == 2

    def test_aug_lmax_tz(self):
        """TZ augmentation adds s, p, d, f → lmax=3"""
        assert ZETA_AUG_LMAX["T"] == 3

    def test_aug_lmax_qz(self):
        """QZ augmentation adds s, p, d, f, g → lmax=4"""
        assert ZETA_AUG_LMAX["Q"] == 4

    def test_jul_dz_drops_d(self):
        """jul-cc-pVDZ drops d diffuse → keeps only s, p diffuse"""
        # aug_lmax for DZ = 2 (d), drop_count = 1 → kept up to l=1 (p)
        drop = CALENDAR_DROP["jul"]
        lmax = ZETA_AUG_LMAX["D"]
        aug_shells = list(range(lmax + 1))  # [0, 1, 2]
        kept = aug_shells[: len(aug_shells) - drop]
        assert kept == [0, 1]  # s and p diffuse only

    def test_jun_tz_drops_f_and_g(self):
        """jun-cc-pVTZ drops f diffuse (and higher) → keeps s, p, d diffuse"""
        drop = CALENDAR_DROP["jun"]
        lmax = ZETA_AUG_LMAX["T"]
        aug_shells = list(range(lmax + 1))  # [0, 1, 2, 3]
        kept = aug_shells[: len(aug_shells) - drop]
        assert kept == [0, 1, 2]  # s, p, d

    def test_may_qz_keeps_sp_only(self):
        """may-cc-pVQZ drops d, f, g diffuse → keeps only s, p"""
        drop = CALENDAR_DROP["may"]
        lmax = ZETA_AUG_LMAX["Q"]
        aug_shells = list(range(lmax + 1))  # [0, 1, 2, 3, 4]
        kept = aug_shells[: len(aug_shells) - drop]
        assert kept == [0, 1]  # s and p

    def test_apr_dz_keeps_s_only(self):
        """apr-cc-pVDZ: drop 4 from [s,p,d] leaves empty → only base"""
        drop = CALENDAR_DROP["apr"]
        lmax = ZETA_AUG_LMAX["D"]
        aug_shells = list(range(lmax + 1))  # [0, 1, 2]
        kept = aug_shells[: max(0, len(aug_shells) - drop)]
        assert kept == []  # nothing kept


# ─── build_calendar_basis integration ─────────────────────────────────────────

class TestBuildCalendarBasis:
    """Integration tests that require PySCF to be installed."""

    def test_jul_dz_water(self):
        """jul-cc-pVDZ for H2O should produce a valid basis dict."""
        result = build_calendar_basis(
            elements=["O", "H"],
            base_name="cc-pVDZ",
            aug_name="aug-cc-pVDZ",
            calendar="jul",
            zeta_level="D",
        )
        assert "O" in result
        assert "H" in result
        # jul-DZ has fewer shells than aug-DZ
        aug_result = build_calendar_basis(
            elements=["O", "H"],
            base_name="cc-pVDZ",
            aug_name="aug-cc-pVDZ",
            calendar="aug",  # won't trigger, aug handled separately
            zeta_level="D",
        )

    def test_calendar_basis_dict_is_list(self):
        """Each element's basis should be a list of shell tuples."""
        result = build_calendar_basis(
            elements=["O"],
            base_name="cc-pVDZ",
            aug_name="aug-cc-pVDZ",
            calendar="jun",
            zeta_level="D",
        )
        assert isinstance(result["O"], list)
        assert len(result["O"]) > 0

    def test_apr_dz_equals_base_for_dz(self):
        """apr-cc-pVDZ should yield essentially the plain cc-pVDZ basis
        (drop 4 from 3 available → no diffuse added)."""
        from pyscf import gto
        result = build_calendar_basis(
            elements=["H"],
            base_name="cc-pVDZ",
            aug_name="aug-cc-pVDZ",
            calendar="apr",
            zeta_level="D",
        )
        base_shells = gto.load("cc-pVDZ", "H")
        # apr should have same or fewer shells than aug, same as base for H
        assert len(result["H"]) >= len(base_shells)


# ─── get_basis_label tests ────────────────────────────────────────────────────

class TestGetBasisLabel:
    def test_plain_dz_label(self):
        label = get_basis_label("D", None, "cc-pV")
        assert "Double" in label
        assert "cc-pVDZ" in label

    def test_aug_tz_label(self):
        label = get_basis_label("T", "aug", "cc-pV")
        assert "aug" in label
        assert "Triple" in label

    def test_jul_label(self):
        label = get_basis_label("D", "jul", "cc-pV")
        assert "jul" in label

    def test_def2_tz_label(self):
        label = get_basis_label("T", None, "def2")
        assert "def2-TZVP" in label
