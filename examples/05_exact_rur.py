"""Construct an RUR and extract exact points.

Guarantee: Exact algebraic coordinates.
Run from the repository root: python examples/05_exact_rur.py
"""

import sympy as sp

from algroots.rational_univariate import (
    compute_rational_univariate_representation,
    solve_rur_representation,
)


def main():
    x, y = sp.symbols("x y")
    representation = compute_rational_univariate_representation((x - y**2, y**2 - 2), (x, y))
    points = solve_rur_representation(representation)
    assert len(points) == 2
    assert all(sp.simplify(a - b**2) == 0 and sp.simplify(b**2 - 2) == 0 for a, b in points)
    print("Defining polynomial:", representation.defining_polynomial)
    print("Exact points:", points)


if __name__ == "__main__":
    main()
