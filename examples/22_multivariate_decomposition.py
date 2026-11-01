"""Search multivariate rational compositions and replay exact coverage.

Guarantee: Nontrivial two-factor classes over QQ, up to Möbius changes.
Run from the repository root: python examples/22_multivariate_decomposition.py
"""

import sympy as sp

from algroots.functional_decomposition import multivariate_rational_decompose, rational_decompose


def main():
    x, y, z, t = sp.symbols("x y z t")
    result = multivariate_rational_decompose((x * x + y * y) ** 2, (x, y), parameter=t)
    assert result.complete and result.verify()
    (pair,) = result.decompositions
    assert pair.degrees == (2, 2) and pair.verify()
    print("Components:", pair.outer, pair.inner)
    rational = rational_decompose((x / y + y / x) ** 2, (x, y), parameter=t)
    assert rational.complete and rational.verify() and len(rational.decompositions) == 3
    print("Three distinct intermediate fields:", [d.inner for d in rational.decompositions])
    partial = multivariate_rational_decompose((x + y) ** 6, (x, y), max_candidates=0)
    assert partial.status == "incomplete" and not partial.complete and partial.verify()
    negative = multivariate_rational_decompose(x**4 + y, (x, y))
    assert negative.complete and negative.status == "indecomposable" and negative.verify()
    print("Budget exhaustion and exhaustive negative claims replay correctly.")
    a, b = x * y + x * z * z - y**5, 9 * x + 25 * y * z
    paper = multivariate_rational_decompose(
        (a * a - b * b) / (a * a - 2 * b * b), (x, y, z), parameter=t
    )
    assert paper.complete and paper.verify()
    assert [d.degrees for d in paper.decompositions] == [(2, 5)]
    print("GRS02 Example 2.1:", paper.decompositions[0].degrees)


if __name__ == "__main__":
    main()
