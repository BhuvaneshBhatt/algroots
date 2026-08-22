import random

import sympy as sp

from algroots.solver import (
    _coordinate_normal_forms,
    _groebner_basis,
    _monomial_expr,
    _normal_form_coeffs,
    _normalize_equations,
    _separator_matrix_coeffs,
    _standard_monomials,
)

x, y = sp.symbols("x y")


def _reference_multiplication_matrix(basis, variables, monomials, variable):
    basis_index = {exponent: index for index, exponent in enumerate(monomials)}
    columns = []
    for exponent in monomials:
        product = variable * _monomial_expr(exponent, variables)
        columns.append(_normal_form_coeffs(basis, product, variables, basis_index))
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
        monomials = _standard_monomials(basis, variables, 32)

        mx = _reference_multiplication_matrix(basis, variables, monomials, x)
        my = _reference_multiplication_matrix(basis, variables, monomials, y)
        coeffs = (2, 3)
        columns = _separator_matrix_coeffs(basis, variables, monomials, coeffs)
        ml = sp.Matrix.hstack(*(sp.Matrix(column) for column in columns))
        assert ml == 2 * mx + 3 * my

        forms = _coordinate_normal_forms(basis, variables, monomials)
        basis_index = {exponent: index for index, exponent in enumerate(monomials)}
        for form, variable in zip(forms, variables, strict=True):
            expected = _normal_form_coeffs(basis, variable, variables, basis_index)
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
        monomials = _standard_monomials(basis, variables, 32)
        forms = _coordinate_normal_forms(basis, variables, monomials)
        basis_index = {exponent: index for index, exponent in enumerate(monomials)}

        for form, variable in zip(forms, variables, strict=True):
            expected = _normal_form_coeffs(basis, variable, variables, basis_index)
            assert form == expected

        coeffs = (rng.randint(1, 4), rng.randint(1, 4))
        columns = _separator_matrix_coeffs(basis, variables, monomials, coeffs)
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
