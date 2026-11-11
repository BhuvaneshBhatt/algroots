"""Search rational compositions and reconstruct a supplied algebraic candidate.

Guarantee: Exact identities and rational-search completeness over QQ.
Run from the repository root: python examples/functional_decomposition.py
"""

import sympy as sp

from algroots.functional_decomposition import algebraic_decompose, rational_decompose
from algroots.quotient import QuotientAlgebra


def main():
    x, z, a, t = sp.symbols("x z a t")
    rational = rational_decompose((x + 1 / x) ** 2, x, parameter=t)
    assert rational.complete and rational.verify() and len(rational.decompositions) == 3
    print("Rational decompositions up to intermediate Möbius changes:")
    for d in rational.decompositions:
        print("  outer:", d.outer, "inner:", d.inner)
    limited = rational_decompose(x**6, x, max_candidates=0)
    assert limited.status == "incomplete" and limited.verify()
    A = QuotientAlgebra.from_polynomials([a * a - z * a + z**3], (a,), domain=sp.QQ.frac_field(z))
    F = x**4 - z * x * x * (x + 1) + z**3 * (x + 1) ** 2
    result = algebraic_decompose(F, (x, z), candidate=x * x - a * (x + 1), ambient=A, parameter=t)
    assert result.verify() and not result.complete
    assert not result.coefficient_field.contains(z)
    print("Algebraic components:", result.decomposition.outer, result.decomposition.inner)
    print("Resultant power:", result.decomposition.exponent)


if __name__ == "__main__":
    main()
