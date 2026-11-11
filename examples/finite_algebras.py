"""Inspect finite-field quotients, nilpotents, and exact primary factors."""

import sympy as sp

from algroots.finite_decomposition import (
    nilradical_basis,
    primary_decomposition,
    reduced_algebra,
)
from algroots.finite_fields import field_elements, field_order, finite_field
from algroots.quotient import QuotientAlgebra


def main():
    x, y = sp.symbols("x y")
    field = finite_field(2, 2)
    assert field_order(field) == 4
    assert len(set(field_elements(field))) == 4

    nonreduced = QuotientAlgebra.from_polynomials((x**2,), (x,), domain=sp.GF(2))
    assert nonreduced.dimension == 2
    assert len(nilradical_basis(nonreduced)) == 1
    assert reduced_algebra(nonreduced).dimension == 1
    ((component, residue),) = primary_decomposition(nonreduced)
    assert component.dimension == 2 and residue.dimension == 1 and residue.is_field
    print("One doubled point: quotient dimension 2, residue field dimension 1")

    product = QuotientAlgebra.from_polynomials((x**2 - x, y**2 - y), (x, y), domain=sp.GF(2))
    factors = primary_decomposition(product)
    assert len(factors) == 4
    assert sum(factor.dimension for factor, _ in factors) == product.dimension
    assert all(factor.dimension == residue.dimension == 1 for factor, residue in factors)
    print("Four reduced local factors over GF(2)")


if __name__ == "__main__":
    main()
