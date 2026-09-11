"""
Basis set registry.

Shell format (as produced by basis_fetcher.get_basis):
  (l, exponents, coefficients)          -- pure angular momentum shell
  ("SP", exponents, s_coeffs, p_coeffs) -- combined SP shell (Pople basis sets)

All basis sets are obtained from the Basis Set Exchange
(https://www.basissetexchange.org) via basis_fetcher and cached in
basis_cache.json, so the network is only touched the first time a given set is
requested.

The Pople sets (STO-3G, 6-31G, 6-31G*, 6-31G**) used to be hardcoded tables in
this module.  Those tables were transcribed incorrectly: only H, He and C
matched the published values.  Oxygen 6-31G was wrong in the third decimal of
every contraction coefficient (~10 mHa in the water RHF energy) and the whole
Na-Ar row was wrong outright, both in the coefficients and in the number of
p primitives (6-31G needs six, the table carried three) — HCl/6-31G came out
28 Ha above the correct energy.  Fetching them from BSE like every other set
removes the transcription risk entirely.
"""

# fmt: off

# ── Registry ──────────────────────────────────────────────────────────────────

# Every basis set is fetched from BSE on first use and then served from
# basis_cache.json.
FETCHED_BASES = [
    # Pople
    "sto-3g",
    "6-31g",
    "6-31g*",
    "6-31g**",
    # Dunning correlation-consistent
    "cc-pvdz",
    "cc-pvtz",
    "cc-pvqz",
    "aug-cc-pvdz",
    "aug-cc-pvtz",
    "aug-cc-pvqz",
    # Calendar (partial augmentation) — Papajak et al.
    "jul-cc-pvdz", "jul-cc-pvtz", "jul-cc-pvqz",
    "jun-cc-pvdz", "jun-cc-pvtz", "jun-cc-pvqz",
    "may-cc-pvdz", "may-cc-pvtz",
    "apr-cc-pvdz", "apr-cc-pvtz",
    "mar-cc-pvdz",
]

AVAILABLE_BASES = list(FETCHED_BASES)

BASIS_LABELS = {
    # Pople
    "sto-3g":      "STO-3G (minimal)",
    "6-31g":       "6-31G (split-valence)",
    "6-31g*":      "6-31G* (+ d polarization)",
    "6-31g**":     "6-31G** (+ d and p polarization)",
    # Dunning correlation-consistent
    "cc-pvdz":     "cc-pVDZ (double zeta)",
    "cc-pvtz":     "cc-pVTZ (triple zeta)",
    "cc-pvqz":     "cc-pVQZ (quad zeta)",
    # Full augmentation
    "aug-cc-pvdz": "aug-cc-pVDZ (full aug DZ)",
    "aug-cc-pvtz": "aug-cc-pVTZ (full aug TZ)",
    "aug-cc-pvqz": "aug-cc-pVQZ (full aug QZ)",
    # Calendar — partial augmentation (Papajak et al.)
    "jul-cc-pvdz": "jul-cc-pVDZ (drop highest-AM diffuse)",
    "jul-cc-pvtz": "jul-cc-pVTZ (drop highest-AM diffuse)",
    "jul-cc-pvqz": "jul-cc-pVQZ (drop highest-AM diffuse)",
    "jun-cc-pvdz": "jun-cc-pVDZ (drop 2 highest-AM diffuse)",
    "jun-cc-pvtz": "jun-cc-pVTZ (drop 2 highest-AM diffuse)",
    "jun-cc-pvqz": "jun-cc-pVQZ (drop 2 highest-AM diffuse)",
    "may-cc-pvdz": "may-cc-pVDZ (s+p diffuse only)",
    "may-cc-pvtz": "may-cc-pVTZ (s+p diffuse only)",
    "apr-cc-pvdz": "apr-cc-pVDZ (s diffuse only on heavy, none on H)",
    "apr-cc-pvtz": "apr-cc-pVTZ (s diffuse only on heavy, none on H)",
    "mar-cc-pvdz": "mar-cc-pVDZ (s diffuse only)",
}
