"""
Accuracy benchmark for the Boys function F_n(x) = ∫₀¹ t^{2n} exp(−x t²) dt.

F_n is the kernel of every Coulomb integral in the code, so its relative error
propagates directly into V and the ERIs.  Reference values come from mpmath
quadrature at 40 decimal digits.

Run:  python benchmarks/bench_boys.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from integrals import boys, _boys_array

try:
    import mpmath as mp
except ImportError:
    sys.exit("this benchmark needs mpmath:  pip install mpmath")

mp.mp.dps = 40

# n up to 10 covers g functions (l=4) in a four-centre integral;
# x spans the full range seen in practice, including both branch boundaries.
N_VALUES = [0, 1, 2, 3, 4, 6, 8, 10]
X_VALUES = [0.0, 1e-9, 1e-4, 0.01, 0.25, 0.5, 0.99, 1.0, 1.5, 5.0,
            12.0, 24.9, 25.0, 25.1, 60.0, 200.0]


def reference(n, x):
    return float(mp.quad(lambda t: t ** (2 * n) * mp.e ** (-x * t * t), [0, 1]))


def main():
    worst_scalar = worst_array = 0.0
    worst_case = None
    print(f"{'n':>3}{'x':>10}{'boys()':>16}{'_boys_array()':>18}{'reference':>18}"
          f"{'rel.err':>12}")
    print("-" * 77)
    for n in N_VALUES:
        for x in X_VALUES:
            ref = reference(n, x)
            a = boys(n, x)
            b = float(_boys_array(n, x)[n])
            ra = abs(a - ref) / abs(ref)
            rb = abs(b - ref) / abs(ref)
            if ra > worst_scalar:
                worst_scalar, worst_case = ra, (n, x)
            worst_array = max(worst_array, rb)
            if max(ra, rb) > 1e-13:
                print(f"{n:>3}{x:>10.4g}{a:>16.9e}{b:>18.9e}{ref:>18.9e}"
                      f"{max(ra, rb):>12.2e}   <<<")
    print("-" * 77)
    print(f"  grid: n ∈ {N_VALUES}, x ∈ [0, 200]  ({len(N_VALUES) * len(X_VALUES)} points)")
    print(f"  worst relative error, boys()        : {worst_scalar:.3e}  at n={worst_case[0]}, x={worst_case[1]}")
    print(f"  worst relative error, _boys_array() : {worst_array:.3e}")
    ok = max(worst_scalar, worst_array) < 1e-13
    print(f"  RESULT: {'PASS' if ok else 'FAIL'} (tolerance 1e-13)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
