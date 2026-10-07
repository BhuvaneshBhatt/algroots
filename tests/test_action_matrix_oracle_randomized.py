import random

import sympy as sp

from algroots.quotient import QuotientAlgebra, monomial_from_exponent
from algroots.solver import (
    _groebner_basis,
    _normalize_equations,
)

x, y = sp.symbols("x y")


def _reference_coordinates(basis, expression, variables, monomials):
    """Independent Groebner reduction oracle for quotient coordinates."""
    remainder = basis.reduce(expression)[1]
    polynomial = sp.Poly(remainder, *variables, extension=True)
    return tuple(polynomial.coeff_monomial(exponent) for exponent in monomials)


def _reference_multiplication_matrix(basis, variables, monomials, variable):
    columns = []
    for exponent in monomials:
        product = variable * monomial_from_exponent(variables, exponent)
        columns.append(_reference_coordinates(basis, product, variables, monomials))
    return sp.Matrix.hstack(*(sp.Matrix(column) for column in columns))


def test_separator_and_coordinate_normal_forms_match_reference_matrices_randomized():
    rng = random.Random(20260821)
    for _ in range(6):
        a = rng.randint(1, 5)
        b = rng.randint(1, 5)
        shear = rng.randint(-2, 2)
        equations = ((x + shear * y) ** 2 - a, y**2 - b)
        variables = (x, y)
        normalized = _normalize_equations(equations, variables)
        basis = _groebner_basis(normalized, variables)
        quotient = QuotientAlgebra.from_groebner_basis(basis, variables, max_dimension=32)
        monomials = quotient.standard_exponents

        mx = _reference_multiplication_matrix(basis, variables, monomials, x)
        my = _reference_multiplication_matrix(basis, variables, monomials, y)
        coeffs = (2, 3)
        columns = quotient.separator_matrix_columns(coeffs)
        ml = sp.Matrix.hstack(*(sp.Matrix(column) for column in columns))
        assert ml == 2 * mx + 3 * my

        forms = quotient.coordinate_normal_forms
        for form, variable in zip(forms, variables, strict=True):
            expected = _reference_coordinates(basis, variable, variables, monomials)
            assert form == expected


def test_randomized_coordinate_forms_remain_exact_under_affine_coupling():
    """Broaden the exact matrix oracle beyond the original square/square family."""
    rng = random.Random(918273)
    for _ in range(10):
        degree = rng.choice((2, 3))
        constant = rng.choice((1, 2, 3, 5))
        shear = rng.randint(-3, 3)
        offset = rng.randint(-2, 2)
        equations = (x - shear * y - offset, y**degree - constant)
        variables = (x, y)
        normalized = _normalize_equations(equations, variables)
        basis = _groebner_basis(normalized, variables)
        quotient = QuotientAlgebra.from_groebner_basis(basis, variables, max_dimension=32)
        monomials = quotient.standard_exponents
        forms = quotient.coordinate_normal_forms

        for form, variable in zip(forms, variables, strict=True):
            expected = _reference_coordinates(basis, variable, variables, monomials)
            assert form == expected

        coeffs = (rng.randint(1, 4), rng.randint(1, 4))
        columns = quotient.separator_matrix_columns(coeffs)
        matrix = sp.Matrix.hstack(*(sp.Matrix(column) for column in columns))
        reference = sum(
            (
                coefficient
                * _reference_multiplication_matrix(
                    basis,
                    variables,
                    monomials,
                    variable,
                )
                for coefficient, variable in zip(coeffs, variables, strict=True)
            ),
            sp.zeros(len(monomials)),
        )
        assert matrix == reference
