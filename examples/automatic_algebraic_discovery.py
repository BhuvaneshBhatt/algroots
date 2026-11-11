"""Discover algebraic factor extensions and verify a functional decomposition.

Guarantee: Exact factor-orbit coverage and explicit search completeness.
Run from the repository root: python examples/automatic_algebraic_discovery.py
"""

import sympy as sp

from algroots.functional_decomposition import algebraic_decompose, discover_algebraic_factors


def main():
    x, z, t = sp.symbols("x z t")
    F = x**4 - z * x * x * (x + 1) + z**3 * (x + 1) ** 2
    discovery = discover_algebraic_factors(F, (x, z), degrees=(2,))
    assert discovery.complete and discovery.verify()
    assert [o.extension_degree for o in discovery.orbits] == [2, 4]
    print("All six quadratic factors occur in extension orbits of sizes 2 and 4.")
    search = algebraic_decompose(F, (x, z), parameter=t)
    assert search.status == "decomposable" and search.verify()
    assert not search.complete  # Default: stop at the first verified answer.
    (D,) = search.decompositions
    print("Automatically discovered components:", D.outer, D.inner)
    complete = algebraic_decompose(x**3 - 3 * x - z, (x, z), parameter=t, exhaustive=True)
    assert complete.complete and complete.verify()
    assert complete.decompositions[0].exponent == 2
    print("Exhaustive cubic search; resultant power:", complete.decompositions[0].exponent)
    negative = algebraic_decompose(x * x + z * x + z, (x, z), exhaustive=True)
    assert negative.complete and negative.status == "indecomposable" and negative.verify()
    print(
        "Certified absence of a nontrivial minimal coefficient-field decomposition:",
        negative.relation,
    )


if __name__ == "__main__":
    main()
