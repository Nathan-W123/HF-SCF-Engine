"""
Integral-level validation: compare the engine's one- and two-electron integral
matrices element-by-element against PySCF (libcint) in the same Cartesian basis.

The two codes order and scale AOs differently (the engine emits s,p together for
a Pople SP shell; both normalise Cartesian d/f components by their own
convention), so AOs are matched by (atom, lmn, exponents) and each engine AO is
rescaled by sqrt(S_pyscf_ii / S_engine_ii) before comparison.  That scaling is a
diagonal similarity transform and leaves the RHF energy invariant.
"""
import sys
import numpy as np

import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import logging
logging.basicConfig(level=logging.ERROR)

from pyscf import gto
from scf_engine import (build_basis, parse_xyz_block, compute_one_electron,
                        compute_eri, ANGSTROM_TO_BOHR, ATOMIC_NUMBERS)

MOLS = {
    "H2O": """O 0.000000 0.000000 0.117176
H 0.000000 0.757001 -0.468704
H 0.000000 -0.757001 -0.468704""",
    "CH4": """C  0.000000  0.000000  0.000000
H  0.629118  0.629118  0.629118
H -0.629118 -0.629118  0.629118
H -0.629118  0.629118 -0.629118
H  0.629118 -0.629118 -0.629118""",
    "HCl": "H 0.0 0.0 0.0\nCl 0.0 0.0 1.2746",
}


def engine_basis_for_pyscf(name):
    """The engine's own basis table, in PySCF's internal format."""
    from basis_fetcher import get_basis
    out = {}
    for el, shells in get_basis(name).items():
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


def engine_ao_keys(bfs):
    keys = []
    for b in bfs:
        keys.append((b.atom_idx, tuple(b.lmn),
                     tuple(round(float(e), 6) for e in b.exponents)))
    return keys


def pyscf_ao_keys(mol):
    """(atom_idx, lmn, exponents) for each Cartesian AO, in PySCF order."""
    CART = {0: [(0, 0, 0)],
            1: [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
            2: [(2, 0, 0), (1, 1, 0), (1, 0, 1), (0, 2, 0), (0, 1, 1), (0, 0, 2)],
            3: [(3, 0, 0), (2, 1, 0), (2, 0, 1), (1, 2, 0), (1, 1, 1), (1, 0, 2),
                (0, 3, 0), (0, 2, 1), (0, 1, 2), (0, 0, 3)]}
    keys = []
    for sh in range(mol.nbas):
        l = mol.bas_angular(sh)
        at = mol.bas_atom(sh)
        exps = mol.bas_exp(sh)
        ncontr = mol.bas_nctr(sh)
        for c in range(ncontr):
            coefs = mol.bas_ctr_coeff(sh)[:, c]
            nz = tuple(round(float(e), 6) for e, cc in zip(exps, coefs) if cc != 0.0)
            for lmn in CART[l]:
                keys.append((at, lmn, nz))
    return keys


def build_perm(bfs, mol):
    ek, pk = engine_ao_keys(bfs), pyscf_ao_keys(mol)
    if len(ek) != len(pk):
        raise RuntimeError(f"AO count differs: engine {len(ek)} pyscf {len(pk)}")
    pool = {}
    for i, k in enumerate(pk):
        pool.setdefault(k, []).append(i)
    perm = np.empty(len(ek), dtype=int)
    for i, k in enumerate(ek):
        if k not in pool or not pool[k]:
            raise RuntimeError(f"engine AO {i} {k} has no PySCF counterpart")
        perm[i] = pool[k].pop(0)
    return perm


print(f"{'molecule':<6}{'basis':<10}{'N':>5}"
      f"{'max|ΔS|':>12}{'max|ΔT|':>12}{'max|ΔV|':>12}{'max|ΔERI|':>12}")
print("-" * 69)
rows = []
for basis in ["sto-3g", "6-31g", "6-31g*", "cc-pvdz"]:
    for name, xyz in MOLS.items():
        atoms_bohr = [(s, x * ANGSTROM_TO_BOHR, y * ANGSTROM_TO_BOHR, z * ANGSTROM_TO_BOHR)
                      for s, x, y, z in parse_xyz_block(xyz)]
        Z = [ATOMIC_NUMBERS[s] for s, *_ in atoms_bohr]
        bfs = build_basis(atoms_bohr, basis)
        # give PySCF exactly the engine's geometry (Bohr) AND the engine's own
        # basis table, so the only thing under test is the integral code
        mol = gto.M(atom=[[s, (x, y, z)] for s, x, y, z in atoms_bohr],
                    basis=engine_basis_for_pyscf(basis), unit="Bohr", verbose=0)
        mol.cart = True
        mol.build()
        try:
            perm = build_perm(bfs, mol)
        except RuntimeError as exc:
            print(f"{name:<6}{basis:<10}  skipped: {exc}")
            continue

        S, T, V = compute_one_electron(bfs, atoms_bohr, Z)
        Sp = mol.intor("int1e_ovlp")[np.ix_(perm, perm)]
        Tp = mol.intor("int1e_kin")[np.ix_(perm, perm)]
        Vp = mol.intor("int1e_nuc")[np.ix_(perm, perm)]

        # rescale engine AOs to PySCF's normalisation (diagonal, energy-invariant)
        sc = np.sqrt(np.diag(Sp) / np.diag(S))
        D = np.outer(sc, sc)
        S, T, V = S * D, T * D, V * D

        E = compute_eri(bfs)
        Ep = mol.intor("int2e")[np.ix_(perm, perm, perm, perm)]
        E = E * D[:, :, None, None] * D[None, None, :, :]

        r = (name, basis, len(bfs),
             np.abs(S - Sp).max(), np.abs(T - Tp).max(),
             np.abs(V - Vp).max(), np.abs(E - Ep).max())
        rows.append(r)
        print(f"{r[0]:<6}{r[1]:<10}{r[2]:>5}{r[3]:>12.2e}{r[4]:>12.2e}{r[5]:>12.2e}{r[6]:>12.2e}")

if rows:
    print("-" * 69)
    print(f"{'WORST':<21}{max(r[3] for r in rows):>12.2e}"
          f"{max(r[4] for r in rows):>12.2e}{max(r[5] for r in rows):>12.2e}"
          f"{max(r[6] for r in rows):>12.2e}")
