"""Compare exact border-basis constructions.

Guarantee: Exact quotient relations and commutation.
Run from the repository root: python examples/10_border_bases.py
"""

import sympy as sp

from algroots.border_basis import compute_border_basis, compute_border_basis_linear


def main():
    x, y = sp.symbols("x y")
    equations = (x**2 - 1, y - x)
    groebner = compute_border_basis(equations, (x, y))
    linear = compute_border_basis_linear(equations, (x, y))
    assert groebner.dimension == linear.dimension == 2
    assert (
        groebner.has_commuting_multiplication_matrices()
        and linear.has_commuting_multiplication_matrices()
    )
    assert groebner.normal_form(x**2 + y**2) == linear.normal_form(x**2 + y**2) == 2
    print("Groebner border basis:", groebner)
    print("Linear border basis:", linear)


if __name__ == "__main__":
    main()
