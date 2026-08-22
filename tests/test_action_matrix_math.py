import sympy as sp

from algroots.solver import (
    _coordinate_normal_forms,
    _groebner_basis,
    _monomial_expr,
    _normal_form_coeffs,
    _separator_matrix_coeffs,
    _standard_monomials,
)

x, y = sp.symbols("x y")


def _matrix_from_columns(columns):
    d = len(columns)
    return sp.Matrix(d, d, lambda row, col: columns[col][row])


def test_separator_matrix_equals_linear_combination_of_coordinate_matrices():
    variables = (x, y)
    equations = (x**2 - 2, y**2 - 3)
    basis = _groebner_basis(equations, variables)
    monomials = _standard_monomials(basis, variables, 16)
    index = {exp: i for i, exp in enumerate(monomials)}

    coordinate_matrices = []
    for variable in variables:
        columns = []
        for exponent in monomials:
            product = variable * _monomial_expr(exponent, variables)
            columns.append(_normal_form_coeffs(basis, product, variables, index))
        coordinate_matrices.append(_matrix_from_columns(columns))

    coeffs = (2, 5)
    separator_columns = _separator_matrix_coeffs(basis, variables, monomials, coeffs)
    separator = _matrix_from_columns(separator_columns)
    reference = sum(
        (c * matrix for c, matrix in zip(coeffs, coordinate_matrices, strict=True)),
        sp.zeros(len(monomials)),
    )
    assert separator == reference


def test_coordinate_normal_forms_recover_variable_values_from_evaluation_vector():
    variables = (x, y)
    equations = (x**2 - 2, y**2 - 3)
    basis = _groebner_basis(equations, variables)
    monomials = _standard_monomials(basis, variables, 16)
    forms = _coordinate_normal_forms(basis, variables, monomials)

    root = (sp.sqrt(2), -sp.sqrt(3))
    evaluation = sp.Matrix(
        [
            sp.prod(value**power for value, power in zip(root, exponent, strict=True))
            for exponent in monomials
        ]
    )
    recovered = [
        sp.expand(sum(coeff * evaluation[i] for i, coeff in enumerate(form))) for form in forms
    ]
    assert all(
        sp.simplify(value - expected) == 0 for value, expected in zip(recovered, root, strict=True)
    )
