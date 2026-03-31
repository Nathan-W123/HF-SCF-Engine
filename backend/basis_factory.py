"""
Basis factory — resolves basis set names and provides human-readable labels.
No external quantum chemistry library dependency.
"""
from __future__ import annotations

from basis_data import AVAILABLE_BASES, BASIS_LABELS


def resolve_basis(basis: str) -> str:
    """Validate and normalize a basis set name."""
    b = basis.strip().lower()
    if b not in AVAILABLE_BASES:
        raise ValueError(
            f"Unknown basis '{basis}'. Available: {AVAILABLE_BASES}"
        )
    return b


def get_basis_label(basis: str) -> str:
    """Human-readable label for a basis set."""
    return BASIS_LABELS.get(basis.lower(), basis)
