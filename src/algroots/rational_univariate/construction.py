from __future__ import annotations

from collections.abc import Iterable, Sequence

import sympy as sp

from .quotient import (
    _as_exact_polynomial,
    _basis_multiplication_matrices,
    _exact_coefficient_domain,
    _leading_exponent_grevlex,
    _select_separating_linear_form,
    _standard_exponents,
    _variable_multiplication_matrices,
)
from .representation import RationalUnivariateError, RationalUnivariateRepresentation


def compute_rational_univariate_representation(
    polynomials: Iterable[sp.Expr],
    variables: Sequence[sp.Symbol],
    parameter: sp.Symbol | None = None,
    *,
    max_separating_attempts: int = 64,
) -> RationalUnivariateRepresentation:
    """Compute an exact RUR for a zero-dimensional polynomial system.

    Coefficients may be rational numbers or exact algebraic numbers. Algebraic
    coefficients are embedded in one compositum number field over ``QQ`` and
    all quotient-algebra operations are performed exactly in that field.
    """

    variable_tuple = tuple(variables)
    raw_polynomials = tuple(sp.sympify(poly) for poly in polynomials)
    if not variable_tuple:
        raise RationalUnivariateError("at least one variable is required")
    if parameter is None:
        used_names = {str(symbol) for symbol in variable_tuple}
        name = "_rur_t"
        while name in used_names:
            name = "_" + name
        parameter = sp.Symbol(name)
    if parameter in variable_tuple:
        raise RationalUnivariateError("parameter must be distinct from system variables")

    domain = _exact_coefficient_domain(raw_polynomials, variable_tuple)
    polys = [
        _as_exact_polynomial(poly, variable_tuple, domain).as_expr() for poly in raw_polynomials
    ]
    if len(polys) < len(variable_tuple):
        raise RationalUnivariateError(
            "at least as many equations as variables are required for rational univariate solving"
        )

    groebner_basis = sp.groebner(polys, *variable_tuple, order="grevlex", domain=domain)
    unit = sp.Poly(1, *variable_tuple, domain=domain)
    if groebner_basis.polys == [unit]:
        result = RationalUnivariateRepresentation(
            variables=variable_tuple,
            parameter=parameter,
            defining_polynomial=sp.Poly(1, parameter, domain=domain),
            coordinate_denominator=sp.Poly(1, parameter, domain=domain),
            coordinate_numerators=tuple(
                sp.Poly(0, parameter, domain=domain) for _ in variable_tuple
            ),
            separating_linear_form=sp.Integer(0),
            standard_exponents=tuple(),
            quotient_dimension=0,
            geometric_solution_count=0,
        )
        return result

    leading_exponents = [_leading_exponent_grevlex(poly) for poly in groebner_basis.polys]
    basis_exponents = _standard_exponents(leading_exponents, len(variable_tuple))
    variable_matrices = _variable_multiplication_matrices(
        groebner_basis, variable_tuple, basis_exponents
    )
    basis_matrices = _basis_multiplication_matrices(basis_exponents, variable_matrices)
    linear_form, defining_poly, denominator_poly, trace_vector, powers, geometric_count = (
        _select_separating_linear_form(
            groebner_basis,
            variable_tuple,
            parameter,
            basis_exponents,
            basis_matrices,
            max_attempts=max_separating_attempts,
        )
    )

    degree = defining_poly.degree()
    coeffs_ascending = list(reversed(defining_poly.all_coeffs()))
    horner_vectors: list[sp.Matrix] = []
    for power_index in range(degree):
        vector = sp.zeros(len(basis_exponents), 1)
        for shift in range(power_index + 1):
            coeff = coeffs_ascending[degree - shift]
            vector += coeff * powers[power_index - shift]
        horner_vectors.append(sp.simplify(vector))

    coordinate_numerators: list[sp.Poly] = []
    for variable_index in range(len(variable_tuple)):
        variable_mult = variable_matrices[variable_index]
        trace_variable_products = variable_mult.T * trace_vector
        numerator = sp.Integer(0)
        for power_index, horner_vector in enumerate(horner_vectors):
            numerator += (horner_vector.T * trace_variable_products)[0] * parameter ** (
                degree - power_index - 1
            )
        coordinate_numerators.append(sp.Poly(sp.expand(numerator), parameter, domain=domain))

    result = RationalUnivariateRepresentation(
        variables=variable_tuple,
        parameter=parameter,
        defining_polynomial=defining_poly,
        coordinate_denominator=denominator_poly,
        coordinate_numerators=tuple(coordinate_numerators),
        separating_linear_form=linear_form,
        standard_exponents=tuple(tuple(e) for e in basis_exponents),
        quotient_dimension=len(basis_exponents),
        geometric_solution_count=geometric_count,
    )
    return result
