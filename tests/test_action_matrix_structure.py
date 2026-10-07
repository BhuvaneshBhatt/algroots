import sympy as sp

from algroots.quotient import QuotientAlgebra
from algroots.solver import (
    _groebner_basis,
    _normalize_equations,
)


def test_separator_matrix_and_coordinate_normal_forms() -> None:
    x, y = sp.symbols("x y")
    variables = (x, y)
    normalized = _normalize_equations((x**2 - 2, y**2 - 3), variables)
    basis = _groebner_basis(normalized, variables)
    quotient = QuotientAlgebra.from_groebner_basis(basis, variables, max_dimension=16)
    monomials = quotient.standard_exponents

    columns = quotient.separator_matrix_columns((1, 2))
    coordinate_forms = quotient.coordinate_normal_forms

    assert len(monomials) == 4
    assert len(columns) == 4
    assert all(len(column) == 4 for column in columns)
    assert len(coordinate_forms) == 2

    # Both coordinate variables are already standard monomials for this ideal,
    # so their normal forms are exact unit vectors in the quotient basis.
    x_index = monomials.index((1, 0))
    y_index = monomials.index((0, 1))
    assert coordinate_forms[0][x_index] == 1
    assert coordinate_forms[1][y_index] == 1
