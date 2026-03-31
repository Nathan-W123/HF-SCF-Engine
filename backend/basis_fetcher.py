"""
Fetch basis sets from the Basis Set Exchange (BSE) REST API.
https://www.basissetexchange.org

Results are cached in basis_cache.json so internet is only needed once per basis set.
Shell format returned: {symbol: [(l, exps, coeffs), ...] or ("SP", exps, s_c, p_c)}

Calendar (partial-augmentation) basis sets (jul/jun/may/apr/mar-cc-pVXZ) are NOT
hosted on BSE. They are constructed here by fetching aug-cc-pVXZ + cc-pVXZ and
dropping the N highest-AM diffuse shells (Papajak et al., J. Chem. Theory Comput. 2011).
"""

import json
import logging
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

CACHE_PATH = Path(__file__).parent / "basis_cache.json"
BSE_URL = "https://www.basissetexchange.org/api/basis/{name}/format/nwchem/"

_SHELL_AM = {"S": 0, "P": 1, "D": 2, "F": 3, "G": 4, "H": 5}

_VALID_ELEMENTS = {
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
}

# Calendar prefix → number of highest-AM diffuse shells to drop per atom
_CALENDAR_N_DROP: dict[str, int] = {
    "jul": 1, "jun": 2, "may": 3, "apr": 4, "mar": 5,
}


def _shell_key(shell: tuple) -> tuple:
    """Hashable key for a shell tuple so we can identify aug-added shells."""
    if shell[0] == "SP":
        return ("SP", tuple(shell[1]), tuple(shell[2]), tuple(shell[3]))
    return (int(shell[0]), tuple(shell[1]), tuple(shell[2]))


def _build_calendar_basis(name: str) -> dict[str, list]:
    """
    Construct a calendar basis (e.g. jul-cc-pvdz) from aug-cc-pVXZ + cc-pVXZ.

    Algorithm per element:
      1. diffuse_shells = aug_shells - cc_shells  (set difference by exponents)
      2. Sort diffuse groups by angular momentum, descending.
      3. Drop the N highest-AM diffuse groups (N = 1 for jul, 2 for jun, …).
      4. Result = cc_shells + remaining diffuse shells.
    """
    prefix, base_name = name.split("-", 1)   # "jul", "cc-pvdz"
    n_drop = _CALENDAR_N_DROP[prefix]
    aug_name = f"aug-{base_name}"            # "aug-cc-pvdz"

    aug_data = get_basis(aug_name)
    cc_data  = get_basis(base_name)

    result: dict[str, list] = {}
    for elem, aug_shells in aug_data.items():
        cc_shells = cc_data.get(elem, [])
        cc_keys   = {_shell_key(s) for s in cc_shells}

        # Shells present in aug but absent from cc are the diffuse functions
        diffuse = [s for s in aug_shells if _shell_key(s) not in cc_keys]

        # Gather unique AMs that have diffuse functions, sorted highest-first
        diffuse_ams = sorted({s[0] if s[0] != "SP" else 1 for s in diffuse}, reverse=True)
        drop_ams    = set(diffuse_ams[:n_drop])

        kept_diffuse = [
            s for s in diffuse
            if (s[0] if s[0] != "SP" else 1) not in drop_ams
        ]
        result[elem] = list(cc_shells) + kept_diffuse

    return result


def get_basis(name: str) -> dict[str, list]:
    """
    Return basis data for elements 1-18.
    Calendar bases (jul/jun/may/apr/mar-cc-pVXZ) are built programmatically.
    All others are fetched from BSE on first call; subsequent calls use local cache.
    Raises RuntimeError if fetch fails.
    """
    cache = _load_cache()
    key = name.lower()
    if key in cache:
        return cache[key]

    # Calendar bases are not on BSE — construct from aug + plain cc
    prefix = key.split("-")[0]
    if prefix in _CALENDAR_N_DROP:
        logger.info(f"Building calendar basis '{key}' from aug + cc components...")
        data = _build_calendar_basis(key)
        cache[key] = data
        _save_cache(cache)
        return data

    logger.info(f"Fetching basis '{name}' from BSE (one-time download)...")
    url = BSE_URL.format(name=key)
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "hf-scf-calculator/2.0"}
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            text = resp.read().decode("utf-8")
    except Exception as exc:
        raise RuntimeError(
            f"Could not fetch basis '{name}' from Basis Set Exchange. "
            f"Check your internet connection. Details: {exc}"
        ) from exc

    data = _parse_nwchem(text)
    if not data:
        raise RuntimeError(
            f"Basis '{name}' not found on BSE or returned no data."
        )

    cache[key] = data
    _save_cache(cache)
    logger.info(f"Cached basis '{name}' for {sorted(data.keys())}")
    return data


# ── NWChem format parser ──────────────────────────────────────────────────────

def _parse_nwchem(text: str) -> dict[str, list]:
    """Parse NWChem-format basis output from BSE into shell tuples."""
    result: dict[str, list] = {}

    cur_elem: str | None = None
    cur_am: int = 0
    cur_is_sp: bool = False
    cur_exps: list[float] = []
    cur_c0: list[float] = []   # s-coeffs (or only coeffs for pure shells)
    cur_c1: list[float] = []   # p-coeffs (SP shells only)

    def _flush():
        nonlocal cur_elem
        if cur_elem is None or not cur_exps:
            return
        if cur_elem not in _VALID_ELEMENTS:
            cur_elem = None
            return
        shells = result.setdefault(cur_elem, [])
        if cur_is_sp:
            shells.append(("SP", list(cur_exps), list(cur_c0), list(cur_c1)))
        else:
            shells.append((cur_am, list(cur_exps), list(cur_c0)))
        cur_elem = None

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        upper = line.upper()
        # Skip NWChem block delimiters
        if upper.startswith("BASIS") or upper == "END":
            continue

        parts = line.split()

        # Shell header: "C S", "H P", "O SP", "N D" …
        if (
            len(parts) == 2
            and parts[0].capitalize() in _VALID_ELEMENTS
            and parts[1].upper() in {*_SHELL_AM, "SP"}
        ):
            _flush()
            cur_elem = parts[0].capitalize()
            stype = parts[1].upper()
            cur_is_sp = stype == "SP"
            cur_am = _SHELL_AM.get(stype, 0)
            cur_exps, cur_c0, cur_c1 = [], [], []
            continue

        # Data row (exponent + coefficient(s))
        if cur_elem is not None and len(parts) >= 2:
            try:
                exp = float(parts[0].replace("D", "E").replace("d", "e"))
                c0  = float(parts[1].replace("D", "E").replace("d", "e"))
            except ValueError:
                continue
            cur_exps.append(exp)
            cur_c0.append(c0)
            if cur_is_sp and len(parts) >= 3:
                try:
                    cur_c1.append(float(parts[2].replace("D","E").replace("d","e")))
                except ValueError:
                    cur_c1.append(0.0)

    _flush()
    return result


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _load_cache() -> dict:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.write_text(
        json.dumps(cache, separators=(",", ":")), encoding="utf-8"
    )
