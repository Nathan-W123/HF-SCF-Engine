"""
Hartree-Fock SCF Molecular Calculator — FastAPI Backend
Custom HF engine (no PySCF).
"""

import logging
from contextlib import asynccontextmanager
from typing import List, Optional
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from database import get_db, init_db
from models import BasisRecipe, CalculationResult
from schemas import (
    CalculateRequest, CalculateResponse,
    EstimateRequest, EstimateResponse,
    OrbitalRequest,
    BasisRecipeCreate, BasisRecipeUpdate, BasisRecipeResponse,
    CalculationSummary,
    OrbitalInfo, DipoleInfo, AtomInfo,
)
from scf_engine import run_rhf, generate_cube, strip_internal, estimate_rhf
from basis_factory import get_basis_label, resolve_basis
from basis_data import AVAILABLE_BASES, BASIS_LABELS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_CACHE_MAX = 20
_result_cache: dict[int, dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    logger.info("Database initialized")
    yield
    _result_cache.clear()


app = FastAPI(
    title="Hartree-Fock SCF Molecular Calculator",
    description="Custom RHF engine with Gaussian basis sets and 3D orbital visualization",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_STATIC = Path(__file__).parent / "static"


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def serve_ui():
    html = (_STATIC / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(content=html)


# ─── Calculation Endpoints ────────────────────────────────────────────────────

@app.post("/estimate", response_model=EstimateResponse, tags=["SCF"])
def estimate(req: EstimateRequest):
    """
    Estimate RHF calculation time without running it.

    Returns basis function count, unique ERI count, estimated time range,
    and memory requirement for the ERI tensor.
    """
    basis = resolve_basis(req.basis)
    try:
        result = estimate_rhf(req.xyz_input, basis, req.charge, req.spin)
    except (ValueError, KeyError) as e:
        raise HTTPException(422, detail=str(e))
    return EstimateResponse(**result)


@app.post("/calculate", response_model=CalculateResponse, tags=["SCF"])
def calculate(req: CalculateRequest, db: Session = Depends(get_db)):
    """
    Run an RHF SCF calculation.

    Accepts XYZ geometry, basis set name, charge, and spin.
    Returns total energy, orbital energies (HOMO/LUMO), dipole moment,
    and convergence status.
    """
    basis = resolve_basis(req.basis)
    basis_label = get_basis_label(basis)

    logger.info(f"RHF calculation: basis={basis_label}, charge={req.charge}, spin={req.spin}")

    result = run_rhf(
        xyz_block=req.xyz_input,
        basis=basis,
        charge=req.charge,
        spin=req.spin,
        max_cycles=req.max_cycles,
        conv_tol=req.conv_tol,
    )

    if "error" in result and result.get("total_energy") is None:
        raise HTTPException(422, detail=result["error"])

    dip = result.get("dipole", {})
    db_calc = CalculationResult(
        molecule_name=req.molecule_name,
        xyz_input=req.xyz_input,
        charge=req.charge,
        spin=req.spin,
        basis_label=basis_label,
        total_energy=result.get("total_energy"),
        converged=1 if result.get("converged") else 0,
        n_iterations=result.get("n_iterations"),
        homo_energy=result.get("homo_energy"),
        lumo_energy=result.get("lumo_energy"),
        dipole_x=dip.get("x"),
        dipole_y=dip.get("y"),
        dipole_z=dip.get("z"),
        dipole_total=dip.get("total"),
        orbital_metadata=result.get("orbitals"),
    )
    db.add(db_calc)
    db.commit()
    db.refresh(db_calc)

    _result_cache[db_calc.id] = result
    if len(_result_cache) > _CACHE_MAX:
        _result_cache.pop(next(iter(_result_cache)))

    orbitals    = [OrbitalInfo(**o) for o in (result.get("orbitals") or [])]
    dipole_resp = DipoleInfo(**dip) if dip else None
    atoms       = [AtomInfo(**a) for a in (result.get("atoms") or [])]

    return CalculateResponse(
        id=db_calc.id,
        converged=result.get("converged", False),
        total_energy=result.get("total_energy"),
        homo_energy=result.get("homo_energy"),
        lumo_energy=result.get("lumo_energy"),
        homo_lumo_gap=result.get("homo_lumo_gap"),
        homo_idx=result.get("homo_idx"),
        dipole=dipole_resp,
        orbitals=orbitals,
        basis_label=basis_label,
        n_electrons=result.get("n_electrons"),
        n_basis=result.get("n_basis"),
        atoms=atoms,
        error=result.get("error"),
    )


@app.post("/calculate/{calc_id}/orbital", tags=["SCF"])
def get_orbital_cube(calc_id: int, req: OrbitalRequest, db: Session = Depends(get_db)):
    """Generate a Gaussian .cube file for a specific molecular orbital."""
    db_calc = _get_or_404(db, CalculationResult, calc_id, "Calculation")
    if not db_calc.converged:
        raise HTTPException(400, "SCF did not converge — cannot generate orbital cube")

    cached = _result_cache.get(calc_id)
    if not cached:
        raise HTTPException(
            410,
            "Calculation data no longer in memory. Re-run the calculation to generate orbitals.",
        )

    orbital_list = db_calc.orbital_metadata or []
    max_orb = len(orbital_list) - 1
    if req.orbital_idx < 0 or req.orbital_idx > max_orb:
        raise HTTPException(400, f"orbital_idx must be 0–{max_orb}")

    try:
        cube_data = generate_cube(cached, req.orbital_idx, req.nx, req.ny, req.nz)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Cube generation failed: {e}")

    return Response(
        content=cube_data,
        media_type="text/plain",
        headers={"Content-Disposition": f'attachment; filename="mo_{req.orbital_idx}.cube"'},
    )


# ─── Calculation History ──────────────────────────────────────────────────────

@app.get("/calculations", response_model=List[CalculationSummary], tags=["History"])
def list_calculations(skip: int = 0, limit: int = 50, db: Session = Depends(get_db)):
    """List all stored calculation results, newest first."""
    results = (
        db.query(CalculationResult)
        .order_by(CalculationResult.created_at.desc())
        .offset(skip).limit(limit).all()
    )
    return [
        CalculationSummary(
            id=r.id,
            molecule_name=r.molecule_name,
            basis_label=r.basis_label,
            total_energy=r.total_energy,
            converged=bool(r.converged),
            n_electrons=None,
            n_basis=len(r.orbital_metadata) if r.orbital_metadata else None,
            homo_energy=r.homo_energy,
            lumo_energy=r.lumo_energy,
            created_at=r.created_at.isoformat(),
        )
        for r in results
    ]


@app.get("/calculations/{calc_id}", response_model=CalculateResponse, tags=["History"])
def get_calculation(calc_id: int, db: Session = Depends(get_db)):
    """Retrieve a specific stored calculation result."""
    r = _get_or_404(db, CalculationResult, calc_id, "Calculation")
    orbitals = [OrbitalInfo(**o) for o in (r.orbital_metadata or [])]
    dip = DipoleInfo(x=r.dipole_x, y=r.dipole_y, z=r.dipole_z, total=r.dipole_total) \
          if r.dipole_x is not None else None
    return CalculateResponse(
        id=r.id,
        converged=bool(r.converged),
        total_energy=r.total_energy,
        homo_energy=r.homo_energy,
        lumo_energy=r.lumo_energy,
        homo_lumo_gap=(r.lumo_energy - r.homo_energy)
            if (r.homo_energy and r.lumo_energy) else None,
        homo_idx=None,
        dipole=dip,
        orbitals=orbitals,
        basis_label=r.basis_label,
        n_electrons=None,
        n_basis=len(r.orbital_metadata) if r.orbital_metadata else None,
        atoms=[],
    )


@app.delete("/calculations/{calc_id}", tags=["History"])
def delete_calculation(calc_id: int, db: Session = Depends(get_db)):
    """Delete a stored calculation result."""
    r = _get_or_404(db, CalculationResult, calc_id, "Calculation")
    db.delete(r)
    db.commit()
    _result_cache.pop(calc_id, None)
    return {"deleted": calc_id}


# ─── Basis Recipe CRUD ────────────────────────────────────────────────────────

@app.post("/basis-recipes", response_model=BasisRecipeResponse, tags=["Basis Recipes"])
def create_basis_recipe(req: BasisRecipeCreate, db: Session = Depends(get_db)):
    if db.query(BasisRecipe).filter(BasisRecipe.name == req.name).first():
        raise HTTPException(409, f"Basis recipe '{req.name}' already exists")
    try:
        basis = resolve_basis(req.basis)
    except ValueError as e:
        raise HTTPException(422, str(e))
    recipe = BasisRecipe(
        name=req.name,
        description=req.description,
        basis=basis,
        extra_primitives=req.extra_primitives,
    )
    db.add(recipe)
    db.commit()
    db.refresh(recipe)
    return _recipe_to_response(recipe)


@app.get("/basis-recipes", response_model=List[BasisRecipeResponse], tags=["Basis Recipes"])
def list_basis_recipes(db: Session = Depends(get_db)):
    return [_recipe_to_response(r) for r in db.query(BasisRecipe).order_by(BasisRecipe.created_at.desc()).all()]


@app.get("/basis-recipes/{recipe_id}", response_model=BasisRecipeResponse, tags=["Basis Recipes"])
def get_basis_recipe(recipe_id: int, db: Session = Depends(get_db)):
    return _recipe_to_response(_get_or_404(db, BasisRecipe, recipe_id, "Recipe"))


@app.put("/basis-recipes/{recipe_id}", response_model=BasisRecipeResponse, tags=["Basis Recipes"])
def update_basis_recipe(recipe_id: int, req: BasisRecipeUpdate, db: Session = Depends(get_db)):
    r = _get_or_404(db, BasisRecipe, recipe_id, "Recipe")
    if req.name is not None:
        r.name = req.name
    if req.description is not None:
        r.description = req.description
    if req.extra_primitives is not None:
        r.extra_primitives = req.extra_primitives
    r.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(r)
    return _recipe_to_response(r)


@app.delete("/basis-recipes/{recipe_id}", tags=["Basis Recipes"])
def delete_basis_recipe(recipe_id: int, db: Session = Depends(get_db)):
    r = _get_or_404(db, BasisRecipe, recipe_id, "Recipe")
    db.delete(r)
    db.commit()
    return {"deleted": recipe_id}


# ─── Utilities ────────────────────────────────────────────────────────────────

@app.get("/basis-options", tags=["Utilities"])
def basis_options():
    """Return valid basis set options for the frontend."""
    return {
        "bases": [
            {"value": k, "label": v}
            for k, v in BASIS_LABELS.items()
        ]
    }


@app.get("/health", tags=["Utilities"])
def health():
    """Health check."""
    import numpy as np
    import scipy
    return {
        "status": "ok",
        "engine": "custom HF (McMurchie-Davidson)",
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
    }


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _get_or_404(db: Session, model, record_id: int, label: str):
    obj = db.query(model).filter(model.id == record_id).first()
    if not obj:
        raise HTTPException(404, f"{label} {record_id} not found")
    return obj


def _recipe_to_response(r: BasisRecipe) -> BasisRecipeResponse:
    return BasisRecipeResponse(
        id=r.id,
        name=r.name,
        description=r.description,
        basis=r.basis,
        extra_primitives=r.extra_primitives,
        created_at=r.created_at.isoformat(),
        updated_at=r.updated_at.isoformat(),
    )
