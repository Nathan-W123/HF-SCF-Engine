from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Text, DateTime, JSON
from database import Base


class BasisRecipe(Base):
    """User-defined basis set presets."""
    __tablename__ = "basis_recipes"

    id          = Column(Integer, primary_key=True, index=True)
    name        = Column(String(255), unique=True, nullable=False)
    description = Column(Text, nullable=True)
    basis       = Column(String(50), nullable=False)   # "sto-3g", "6-31g", etc.
    extra_primitives = Column(JSON, nullable=True)
    created_at  = Column(DateTime, default=datetime.utcnow)
    updated_at  = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class CalculationResult(Base):
    """Stored SCF calculation results."""
    __tablename__ = "calculation_results"

    id              = Column(Integer, primary_key=True, index=True)
    molecule_name   = Column(String(255), nullable=True)
    xyz_input       = Column(Text, nullable=False)
    charge          = Column(Integer, default=0)
    spin            = Column(Integer, default=0)
    basis_label     = Column(String(255), nullable=False)

    total_energy    = Column(Float, nullable=True)
    converged       = Column(Integer, default=0)
    n_iterations    = Column(Integer, nullable=True)
    homo_energy     = Column(Float, nullable=True)
    lumo_energy     = Column(Float, nullable=True)
    dipole_x        = Column(Float, nullable=True)
    dipole_y        = Column(Float, nullable=True)
    dipole_z        = Column(Float, nullable=True)
    dipole_total    = Column(Float, nullable=True)
    orbital_metadata= Column(JSON, nullable=True)

    created_at      = Column(DateTime, default=datetime.utcnow)
