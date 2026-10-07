"""Inspect exact quotient structure and bounded caches.

Guarantee: Exact finite algebra.
Run from the repository root: python examples/04_quotient_and_cache.py
"""

import sympy as sp

from algroots.quotient import QuotientAlgebra


def main():
    x, y = sp.symbols("x y")
    quotient = QuotientAlgebra.from_polynomials(
        (x**2 - 2, y**2 - 3), (x, y), normal_form_cache_size=4
    )
    assert quotient.dimension == quotient.geometric_solution_count == 4
    assert quotient.normal_form(x**4 + y**2) == 7
    assert quotient.normal_form(x**4 + y**2) == 7
    assert quotient.normal_form_cache_info["hits"] >= 1
    mx, my = quotient.variable_multiplication_matrices
    assert mx * my == my * mx
    print("Basis:", quotient.standard_monomials)
    print("Cache:", quotient.normal_form_cache_info)
    print("Operations:", quotient.operation_diagnostics)


if __name__ == "__main__":
    main()
