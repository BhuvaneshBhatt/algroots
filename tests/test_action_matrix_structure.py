import sympy as sp

from algroots.solver import (
    _coordinate_normal_forms,
    _groebner_basis,
    _normalize_equations,
    _separator_matrix_coeffs,
    _standard_monomials,
)


def test_separator_matrix_and_coordinate_normal_forms() -> None:
    x, y = sp.symbols("x y")
    variables = (x, y)
    normalized = _normalize_equations((x**2 - 2, y**2 - 3), variables)
    basis = _groebner_basis(normalized, variables)
    monomials = _standard_monomials(basis, variables, 16)

    columns = _separator_matrix_coeffs(basis, variables, monomials, (1, 2))
    coordinate_forms = _coordinate_normal_forms(basis, variables, monomials)

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
