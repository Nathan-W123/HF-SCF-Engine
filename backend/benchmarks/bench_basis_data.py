"""Decisive basis-data check: is the engine's basis table the SAME BASIS as the
reference, regardless of how it is segmented?

Both are handed to PySCF and run through the identical SCF code path, so any
energy difference is attributable purely to the basis data.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pyscf import gto, scf

MOLS = {
    "H2O": """O 0.000000 0.000000 0.117176
H 0.000000 0.757001 -0.468704
H 0.000000 -0.757001 -0.468704""",
    "CH4": """C  0.000000  0.000000  0.000000
H  0.629118  0.629118  0.629118
H -0.629118 -0.629118  0.629118
H -0.629118  0.629118 -0.629118
H  0.629118 -0.629118 -0.629118""",
    "HCl": """H 0.0 0.0 0.0
Cl 0.0 0.0 1.274""",
}


def engine_basis_to_pyscf(name):
    from basis_fetcher import get_basis
    bd = get_basis(name)
    out = {}
    for el, shells in bd.items():
        lst = []
        for sh in shells:
            if sh[0] == "SP":
                _, e, sc, pc = sh
                lst.append([0] + [[float(a), float(c)] for a, c in zip(e, sc)])
                lst.append([1] + [[float(a), float(c)] for a, c in zip(e, pc)])
            else:
                l, e, c = sh
                lst.append([int(l)] + [[float(a), float(cc)] for a, cc in zip(e, c)])
        out[el] = lst
    return out


def energy(mol_xyz, basis_spec, cart):
    m = gto.M(atom=mol_xyz, basis=basis_spec, unit="Angstrom", verbose=0)
    m.cart = cart
    m.build()
    mf = scf.RHF(m)
    mf.conv_tol = 1e-12
    mf.verbose = 0
    return mf.kernel(), m.nao


print(f"{'basis':<12}{'molecule':<8}{'E[engine table]':>20}{'E[reference]':>20}{'Δ / mHa':>14}{'nao':>8}")
print("-" * 84)
for bname in ["sto-3g", "6-31g", "6-31g*", "6-31g**", "cc-pvdz", "cc-pvtz", "aug-cc-pvdz"]:
    try:
        eb = engine_basis_to_pyscf(bname)
    except Exception as exc:
        print(f"{bname:<12} could not load engine table: {exc}")
        continue
    for mname, xyz in MOLS.items():
        els = set(l.split()[0] for l in xyz.strip().splitlines())
        if not els <= set(eb):
            continue
        try:
            e1, n1 = energy(xyz, eb, cart=True)
            e2, n2 = energy(xyz, bname, cart=True)
        except Exception as exc:
            print(f"{bname:<12}{mname:<8} FAILED: {exc}")
            continue
        d = (e1 - e2) * 1000
        flag = "" if abs(d) < 1e-3 else "   <<< basis data differs"
        print(f"{bname:<12}{mname:<8}{e1:>20.10f}{e2:>20.10f}{d:>14.5f}{n1:>5}/{n2:<3}{flag}")
