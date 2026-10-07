import sympy as sp

from algroots.quotient import QuotientAlgebra, monomial_from_exponent
from algroots.solver import (
    _groebner_basis,
)

x, y = sp.symbols("x y")


def _reference_coordinates(basis, expression, variables, monomials):
    """Independent Groebner reduction oracle for quotient coordinates."""
    remainder = basis.reduce(expression)[1]
    polynomial = sp.Poly(remainder, *variables, extension=True)
    return tuple(polynomial.coeff_monomial(exponent) for exponent in monomials)


def _matrix_from_columns(columns):
    d = len(columns)
    return sp.Matrix(d, d, lambda row, col: columns[col][row])


def test_separator_matrix_equals_linear_combination_of_coordinate_matrices():
    variables = (x, y)
    equations = (x**2 - 2, y**2 - 3)
    basis = _groebner_basis(equations, variables)
    quotient = QuotientAlgebra.from_groebner_basis(basis, variables, max_dimension=16)
    monomials = quotient.standard_exponents

    coordinate_matrices = []
    for variable in variables:
        columns = []
        for exponent in monomials:
            product = variable * monomial_from_exponent(variables, exponent)
            columns.append(_reference_coordinates(basis, product, variables, monomials))
        coordinate_matrices.append(_matrix_from_columns(columns))

    coeffs = (2, 5)
    separator_columns = quotient.separator_matrix_columns(coeffs)
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
    quotient = QuotientAlgebra.from_groebner_basis(basis, variables, max_dimension=16)
    monomials = quotient.standard_exponents
    forms = quotient.coordinate_normal_forms

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
