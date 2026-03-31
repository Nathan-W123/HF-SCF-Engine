"""Pydantic request/response schemas for the FastAPI endpoints."""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator
from basis_data import AVAILABLE_BASES


# ─── Request Schemas ──────────────────────────────────────────────────────────

class CalculateRequest(BaseModel):
    xyz_input: str = Field(
        ...,
        description="Molecular geometry in XYZ format (with or without count/comment header)",
        example="O 0.000 0.000 0.117\nH 0.000 0.757 -0.471\nH 0.000 -0.757 -0.471",
    )
    basis: str = Field(
        "sto-3g",
        description=f"Basis set name. Available: {AVAILABLE_BASES}",
    )
    charge: int = Field(0, ge=-10, le=10, description="Molecular charge")
    spin: int = Field(0, ge=0, le=20, description="2S (number of unpaired electrons)")
    max_cycles: int = Field(200, ge=1, le=1000)
    conv_tol: float = Field(1e-9, ge=1e-14, le=1e-4)
    molecule_name: Optional[str] = Field(None, description="Optional label for this molecule")
    basis_recipe_id: Optional[int] = Field(None, description="(legacy, ignored)")

    @field_validator("basis")
    @classmethod
    def validate_basis(cls, v):
        b = v.strip().lower()
        if b not in AVAILABLE_BASES:
            raise ValueError(f"Unknown basis '{v}'. Available: {AVAILABLE_BASES}")
        return b


class EstimateRequest(BaseModel):
    xyz_input: str
    basis: str = Field("sto-3g", description=f"Basis set name. Available: {AVAILABLE_BASES}")
    charge: int = Field(0, ge=-10, le=10)
    spin: int = Field(0, ge=0, le=20)

    @field_validator("basis")
    @classmethod
    def validate_basis(cls, v):
        b = v.strip().lower()
        if b not in AVAILABLE_BASES:
            raise ValueError(f"Unknown basis '{v}'. Available: {AVAILABLE_BASES}")
        return b


class EstimateResponse(BaseModel):
    n_basis: int
    n_electrons: int
    n_dominant_integrals: int
    method: str
    estimated_seconds_low: float
    estimated_seconds_mid: float
    estimated_seconds_high: float
    memory_mb: float
    warning: Optional[str] = None


class OrbitalRequest(BaseModel):
    calculation_id: int
    orbital_idx: int = Field(..., description="0-based MO index")
    nx: int = Field(60, ge=20, le=150)
    ny: int = Field(60, ge=20, le=150)
    nz: int = Field(60, ge=20, le=150)


class BasisRecipeCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    basis: str = "sto-3g"
    extra_primitives: Optional[Dict[str, Any]] = None


class BasisRecipeUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    extra_primitives: Optional[Dict[str, Any]] = None


# ─── Response Schemas ─────────────────────────────────────────────────────────

class OrbitalInfo(BaseModel):
    index: int
    energy_hartree: float
    energy_ev: float
    occupation: float
    label: str


class DipoleInfo(BaseModel):
    x: float
    y: float
    z: float
    total: float


class AtomInfo(BaseModel):
    symbol: str
    x: float
    y: float
    z: float


class CalculateResponse(BaseModel):
    id: int
    converged: bool
    total_energy: Optional[float]
    homo_energy: Optional[float]
    lumo_energy: Optional[float]
    homo_lumo_gap: Optional[float]
    homo_idx: Optional[int]
    dipole: Optional[DipoleInfo]
    orbitals: List[OrbitalInfo]
    basis_label: str
    n_electrons: Optional[int]
    n_basis: Optional[int]
    atoms: List[AtomInfo]
    error: Optional[str] = None


class BasisRecipeResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    basis: str
    extra_primitives: Optional[Dict[str, Any]]
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True


class CalculationSummary(BaseModel):
    id: int
    molecule_name: Optional[str]
    basis_label: str
    total_energy: Optional[float]
    converged: bool
    n_electrons: Optional[int]
    n_basis: Optional[int]
    homo_energy: Optional[float]
    lumo_energy: Optional[float]
    created_at: str

    class Config:
        from_attributes = True
