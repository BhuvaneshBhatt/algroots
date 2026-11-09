from __future__ import annotations

from collections.abc import Iterable, Sequence

import sympy as sp

from ..errors import QuotientAlgebraError
from ..quotient import QuotientAlgebra
from .representation import RationalUnivariateError, RationalUnivariateRepresentation


def _as_rur_error(exc: QuotientAlgebraError) -> RationalUnivariateError:
    return RationalUnivariateError(str(exc))


def compute_rational_univariate_representation(
    polynomials: Iterable[sp.Expr],
    variables: Sequence[sp.Symbol],
    parameter: sp.Symbol | None = None,
    *,
    max_separating_attempts: int = 64,
    domain=None,
    groebner_backend="native",
    groebner_limits=None,
) -> RationalUnivariateRepresentation:
    """Compute an exact RUR for a zero-dimensional polynomial system.

    ``groebner_backend`` and ``groebner_limits`` select verified basis construction.
    ``domain`` may be QQ, a number field, or a characteristic-zero rational
    function field. Such a representation describes the generic field problem;
    exceptional fibers need separate specialization conditions. The
    equality-only quotient structure is provided by :class:`algroots.quotient.QuotientAlgebra`;
    this function adds the rational-univariate representation layer.
    """

    variable_tuple = tuple(variables)
    raw_polynomials = tuple(sp.sympify(poly) for poly in polynomials)
    if not variable_tuple:
        raise RationalUnivariateError("at least one variable is required")
    if len(raw_polynomials) < len(variable_tuple):
        raise RationalUnivariateError(
            "at least as many equations as variables are required for rational univariate solving"
        )
    if domain is not None:
        from ..domains import coefficient_domain

        domain = coefficient_domain(domain)
        if domain.characteristic():
            raise RationalUnivariateError(
                "RUR construction requires characteristic-zero coefficients"
            )
    try:
        quotient = QuotientAlgebra.from_polynomials(
            raw_polynomials,
            variable_tuple,
            order="grevlex",
            domain=domain,
            groebner_backend=groebner_backend,
            groebner_limits=groebner_limits,
        )
    except QuotientAlgebraError as exc:
        raise _as_rur_error(exc) from exc
    return rur_from_quotient(quotient, parameter, max_separating_attempts=max_separating_attempts)


def rur_from_quotient(quotient, parameter=None, *, max_separating_attempts=64):
    """Build an RUR of precisely the supplied algebra without rebuilding its ideal."""
    variable_tuple = quotient.variables
    domain = quotient.domain
    if domain.characteristic():
        raise RationalUnivariateError("RUR construction requires characteristic-zero coefficients")
    coefficient_symbols = tuple(domain.symbols) if domain.is_FractionField else ()
    coefficient_symbols += (
        *getattr(domain, "parameters", ()),
        *getattr(domain, "tower_symbols", ()),
    )
    if getattr(domain, "is_AlgebraicFunctionField", False):
        coefficient_symbols += (domain.symbol,)
    used = {str(v) for v in (*variable_tuple, *coefficient_symbols)}
    if parameter is None:
        name = "_rur_t"
        while name in used:
            name = "_" + name
        parameter = sp.Symbol(name)
    if not isinstance(parameter, sp.Symbol) or str(parameter) in used:
        raise RationalUnivariateError(
            "RUR parameter must be a Symbol distinct from coordinates and coefficient parameters"
        )
    if quotient.dimension == 0:
        return RationalUnivariateRepresentation(
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

    try:
        separator = quotient.separating_element(
            parameter,
            max_attempts=max_separating_attempts,
        )
    except QuotientAlgebraError as exc:
        raise _as_rur_error(exc) from exc

    defining_poly = separator.defining_polynomial
    degree = defining_poly.degree()
    from sympy.polys.matrices import DomainMatrix

    from ..domains import domain_matrix

    coefficients = [domain.convert(c) for c in defining_poly.rep.to_list()]
    action = quotient._linear_combination_domain_matrix(separator.coefficients)
    unit = DomainMatrix({0: {0: domain.one}}, (quotient.dimension, 1), domain)
    horner = [unit]
    for i in range(1, degree):
        horner.append(action * horner[-1] + unit.scalarmul(coefficients[i]))
    coordinate_numerators = []
    traces = domain_matrix(separator.trace_vector, domain)
    for variable in quotient._variable_domain_matrices:
        functional = variable.transpose() * traces
        values = [
            (vector.transpose() * functional).to_dod().get(0, {}).get(0, domain.zero)
            for vector in horner
        ]
        coordinate_numerators.append(sp.Poly.from_list(values, parameter, domain=domain))

    return RationalUnivariateRepresentation(
        variables=variable_tuple,
        parameter=parameter,
        defining_polynomial=defining_poly,
        coordinate_denominator=separator.coordinate_denominator,
        coordinate_numerators=tuple(coordinate_numerators),
        separating_linear_form=separator.expression,
        standard_exponents=quotient.standard_exponents,
        quotient_dimension=quotient.dimension,
        geometric_solution_count=separator.geometric_solution_count,
    )
