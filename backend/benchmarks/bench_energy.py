"""
Accuracy + performance benchmark for the HF-SCF-Engine RHF implementation.

Reference: PySCF 2.14 RHF, same geometry, same basis, same Cartesian/spherical
convention (the engine uses Cartesian Gaussians throughout, so PySCF is run with
mol.cart = True), converged to 1e-12.  Any remaining difference is attributable
to the engine.
"""
import argparse
import json
import logging
import os
import sys
import time

import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
logging.basicConfig(level=logging.ERROR)

import numpy as np
from pyscf import gto, scf
from scf_engine import run_rhf

HARTREE_TO_KCAL = 627.5094740631

MOLECULES = {
    # name: (xyz in Angstrom, charge)
    "H2":   ("H 0.0 0.0 0.0\nH 0.0 0.0 0.74", 0),
    "LiH":  ("Li 0.0 0.0 0.0\nH 0.0 0.0 1.5949", 0),
    "H2O":  ("O 0.000000 0.000000 0.117176\n"
             "H 0.000000 0.757001 -0.468704\n"
             "H 0.000000 -0.757001 -0.468704", 0),
    "NH3":  ("N  0.000000  0.000000  0.112607\n"
             "H  0.000000  0.938169 -0.262782\n"
             "H  0.812405 -0.469084 -0.262782\n"
             "H -0.812405 -0.469084 -0.262782", 0),
    "CH4":  ("C  0.000000  0.000000  0.000000\n"
             "H  0.629118  0.629118  0.629118\n"
             "H -0.629118 -0.629118  0.629118\n"
             "H -0.629118  0.629118 -0.629118\n"
             "H  0.629118 -0.629118 -0.629118", 0),
    "HF":   ("H 0.0 0.0 0.0\nF 0.0 0.0 0.9168", 0),
    "CO":   ("C 0.0 0.0 0.0\nO 0.0 0.0 1.128", 0),
    "N2":   ("N 0.0 0.0 0.0\nN 0.0 0.0 1.0977", 0),
    "HCl":  ("H 0.0 0.0 0.0\nCl 0.0 0.0 1.2746", 0),
    "C2H4": ("C  0.000000  0.000000  0.667500\n"
             "C  0.000000  0.000000 -0.667500\n"
             "H  0.000000  0.922600  1.237800\n"
             "H  0.000000 -0.922600  1.237800\n"
             "H  0.000000  0.922600 -1.237800\n"
             "H  0.000000 -0.922600 -1.237800", 0),
    "C6H6": ("\n".join(
        [f"C {1.3970*np.cos(np.pi/3*i):.6f} {1.3970*np.sin(np.pi/3*i):.6f} 0.000000" for i in range(6)] +
        [f"H {2.4810*np.cos(np.pi/3*i):.6f} {2.4810*np.sin(np.pi/3*i):.6f} 0.000000" for i in range(6)]
    ), 0),
}
MOLECULES["H2CO"] = ("C  0.000000  0.000000 -0.533500\n"
                     "O  0.000000  0.000000  0.669500\n"
                     "H  0.000000  0.934800 -1.081900\n"
                     "H  0.000000 -0.934800 -1.081900", 0)


def pyscf_reference(xyz, basis, charge=0):
    mol = gto.M(atom=xyz, basis=basis, unit="Angstrom", charge=charge, verbose=0)
    mol.cart = True          # engine uses Cartesian Gaussians
    mol.build()
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.verbose = 0
    t0 = time.perf_counter()
    e = mf.kernel()
    t = time.perf_counter() - t0
    if not mf.converged:
        raise RuntimeError("reference SCF did not converge")
    dip = mf.dip_moment(unit="Debye")
    homo = mf.mo_energy[mol.nelectron // 2 - 1]
    lumo = mf.mo_energy[mol.nelectron // 2]
    return dict(e=float(e), nao=int(mol.nao), t=t,
                dip=float(np.linalg.norm(dip)),
                homo=float(homo), lumo=float(lumo))


def run_case(name, basis):
    xyz, charge = MOLECULES[name]
    ref = pyscf_reference(xyz, basis, charge)
    t0 = time.perf_counter()
    r = run_rhf(xyz, basis, charge=charge, conv_tol=1e-10, max_cycles=300)
    t = time.perf_counter() - t0
    if r.get("error"):
        return dict(molecule=name, basis=basis, error=r["error"])
    d = dict(
        molecule=name, basis=basis,
        nbf=r["n_basis"], nao_ref=ref["nao"],
        converged=bool(r.get("converged")),
        e=float(r["total_energy"]), e_ref=ref["e"],
        resid_mHa=(float(r["total_energy"]) - ref["e"]) * 1e3,
        t_engine=t, t_ref=ref["t"],
        point_group=r.get("point_group"),
    )
    d["resid_per_e_mHa"] = d["resid_mHa"] / max(r.get("n_electrons", 1), 1)
    d["resid_kcal"] = (d["e"] - d["e_ref"]) * HARTREE_TO_KCAL
    if r.get("homo_energy") is not None:
        d["homo"] = float(r["homo_energy"]); d["homo_ref"] = ref["homo"]
        d["homo_err_eV"] = (d["homo"] - ref["homo"]) * 27.211386245
    dp = r.get("dipole")
    if isinstance(dp, dict) and "total" in dp:
        d["dipole"] = float(dp["total"]); d["dipole_ref"] = ref["dip"]
        d["dipole_err"] = d["dipole"] - ref["dip"]
    return d


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True,
                    help="semicolon list of molecule:basis, e.g. 'H2O:sto-3g;CH4:6-31g'")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    results = []
    for spec in a.cases.split(";"):
        if not spec.strip():
            continue
        mol, basis = spec.split(":")
        try:
            res = run_case(mol.strip(), basis.strip())
        except Exception as exc:
            res = dict(molecule=mol, basis=basis, error=f"{type(exc).__name__}: {exc}")
        results.append(res)
        if "error" in res:
            print(f"{res['molecule']:<6} {res['basis']:<12} ERROR {res['error']}", flush=True)
        else:
            print(f"{res['molecule']:<6} {res['basis']:<12} N={res['nbf']:<4} "
                  f"conv={res['converged']!s:<5} "
                  f"E={res['e']:.9f}  ref={res['e_ref']:.9f}  "
                  f"resid={res['resid_mHa']:+.5f} mHa  "
                  f"t={res['t_engine']:.2f}s (ref {res['t_ref']:.2f}s)", flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(results, f, indent=2)
