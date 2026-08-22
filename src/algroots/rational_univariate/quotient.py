from __future__ import annotations

from collections.abc import Iterable, Sequence
from itertools import product

import sympy as sp
from sympy.polys.domains import Domain
from sympy.polys.orderings import grevlex

from .representation import RationalUnivariateError


def _exact_coefficient_domain(
    expressions: Iterable[sp.Expr], variables: Sequence[sp.Symbol]
) -> Domain:
    """Infer QQ or a single exact algebraic number field for ``expressions``."""

    algebraic_coefficients: list[sp.Expr] = []
    for expr in expressions:
        try:
            poly = sp.Poly(sp.expand(expr), *variables)
        except (sp.PolynomialError, ValueError, TypeError) as exc:
            raise RationalUnivariateError(
                "RUR requires polynomial equations with exact rational or algebraic coefficients"
            ) from exc
        if poly.total_degree() < 0:
            raise RationalUnivariateError("zero polynomial is not a valid defining equation")
        for coefficient in poly.coeffs():
            coefficient = sp.sympify(coefficient)
            if coefficient.is_Rational:
                continue
            if coefficient.is_algebraic is not True or coefficient.has(sp.Float):
                raise RationalUnivariateError(
                    "RUR requires exact rational or algebraic coefficients"
                )
            algebraic_coefficients.append(coefficient)

    if not algebraic_coefficients:
        return sp.QQ
    try:
        return sp.QQ.algebraic_field(*algebraic_coefficients)
    except (sp.PolynomialError, ValueError, TypeError) as exc:
        raise RationalUnivariateError(
            "could not construct a common algebraic number field for the coefficients"
        ) from exc


def _as_exact_polynomial(
    expr: sp.Expr,
    variables: Sequence[sp.Symbol],
    domain: Domain,
) -> sp.Poly:
    try:
        poly = sp.Poly(sp.expand(expr), *variables, domain=domain)
    except (sp.PolynomialError, sp.polys.polyerrors.CoercionFailed, ValueError, TypeError) as exc:
        raise RationalUnivariateError(
            "RUR requires polynomial equations over the inferred exact coefficient field"
        ) from exc
    if poly.total_degree() < 0:
        raise RationalUnivariateError("zero polynomial is not a valid defining equation")
    return poly


def _as_rational_polynomial(expr: sp.Expr, variables: Sequence[sp.Symbol]) -> sp.Poly:
    """Rational-only helper retained for the exact border-basis implementation."""

    return _as_exact_polynomial(expr, variables, sp.QQ)


def _leading_exponent_grevlex(poly: sp.Poly) -> tuple[int, ...]:
    terms = poly.terms(order=grevlex)
    if not terms:
        raise RationalUnivariateError("zero polynomial has no leading monomial")
    return tuple(int(v) for v in terms[0][0])


def _componentwise_leq(left: Sequence[int], right: Sequence[int]) -> bool:
    return all(a <= b for a, b in zip(left, right, strict=True))


def _is_not_divisible_by_any_leading_monomial(
    candidate: Sequence[int], leading_exponents: Sequence[Sequence[int]]
) -> bool:
    return not any(_componentwise_leq(leading, candidate) for leading in leading_exponents)


def _standard_exponents(
    leading_exponents: Sequence[Sequence[int]], variable_count: int
) -> tuple[tuple[int, ...], ...]:
    """Return exponent vectors of monomials outside the leading ideal.

    The standard monomial basis is finite when each variable direction has a
    pure-power leading monomial bound. This condition is sufficient for the
    quotient basis used here and rejects positive-dimensional inputs early.
    """

    bounds: list[int] = []
    for index in range(variable_count):
        pure_power = None
        for exponent in leading_exponents:
            if all(value == 0 for pos, value in enumerate(exponent) if pos != index):
                pure_power = int(exponent[index])
                break
        if pure_power is None or pure_power <= 0:
            raise RationalUnivariateError("system does not expose a finite standard-monomial basis")
        bounds.append(pure_power)

    candidates = product(*(range(bound) for bound in bounds))
    basis = [
        tuple(candidate)
        for candidate in candidates
        if _is_not_divisible_by_any_leading_monomial(candidate, leading_exponents)
    ]
    if not basis or basis[0] != tuple(0 for _ in range(variable_count)):
        basis.sort(key=lambda exp: (sum(exp), exp))
    return tuple(basis)


def _monomial_from_exponent(variables: Sequence[sp.Symbol], exponent: Sequence[int]) -> sp.Expr:
    monomial = sp.Integer(1)
    for variable, power in zip(variables, exponent, strict=True):
        monomial *= variable ** int(power)
    return monomial


def _normal_form(groebner_basis: sp.polys.polytools.GroebnerBasis, expr: sp.Expr) -> sp.Expr:
    try:
        _, remainder = groebner_basis.reduce(sp.expand(expr))
    except (
        sp.PolynomialError,
        ValueError,
        TypeError,
    ) as exc:  # pragma: no cover - defensive SymPy boundary
        raise RationalUnivariateError(f"Groebner reduction failed: {exc}") from exc
    return sp.expand(remainder)


def _coefficient_vector(
    expr: sp.Expr,
    variables: Sequence[sp.Symbol],
    basis_exponents: Sequence[Sequence[int]],
    domain: Domain = sp.QQ,
) -> sp.Matrix:
    poly = sp.Poly(sp.expand(expr), *variables, domain=domain)
    coefficient_rules = {tuple(mon): coeff for mon, coeff in poly.terms()}
    return sp.Matrix(
        [coefficient_rules.get(tuple(exponent), sp.Integer(0)) for exponent in basis_exponents]
    )


def _multiplication_tensor(
    groebner_basis: sp.polys.polytools.GroebnerBasis,
    variables: Sequence[sp.Symbol],
    basis_exponents: Sequence[Sequence[int]],
) -> list[list[sp.Matrix]]:
    basis_monomials = [_monomial_from_exponent(variables, exponent) for exponent in basis_exponents]
    tensor: list[list[sp.Matrix]] = []
    for left in basis_monomials:
        row: list[sp.Matrix] = []
        for right in basis_monomials:
            remainder = _normal_form(groebner_basis, left * right)
            row.append(
                _coefficient_vector(remainder, variables, basis_exponents, groebner_basis.domain)
            )
        tensor.append(row)
    return tensor


def _variable_multiplication_matrices(
    groebner_basis: sp.polys.polytools.GroebnerBasis,
    variables: Sequence[sp.Symbol],
    basis_exponents: Sequence[Sequence[int]],
) -> tuple[sp.Matrix, ...]:
    """Build coordinate multiplication matrices with n*D Groebner reductions."""
    basis_monomials = [_monomial_from_exponent(variables, exponent) for exponent in basis_exponents]
    matrices: list[sp.Matrix] = []
    for variable in variables:
        columns = []
        for basis_monomial in basis_monomials:
            remainder = _normal_form(groebner_basis, variable * basis_monomial)
            columns.append(
                _coefficient_vector(remainder, variables, basis_exponents, groebner_basis.domain)
            )
        matrices.append(sp.Matrix.hstack(*columns) if columns else sp.zeros(0, 0))
    return tuple(matrices)


def _basis_multiplication_matrices(
    basis_exponents: Sequence[Sequence[int]],
    variable_matrices: Sequence[sp.Matrix],
) -> tuple[sp.Matrix, ...]:
    """Construct multiplication by each basis monomial from coordinate actions."""
    dimension = len(basis_exponents)
    identity = sp.eye(dimension)
    matrices: list[sp.Matrix] = []
    power_cache: list[dict[int, sp.Matrix]] = [
        {0: identity, 1: matrix} for matrix in variable_matrices
    ]
    for exponent in basis_exponents:
        matrix = identity
        for index, power in enumerate(exponent):
            if not power:
                continue
            cache = power_cache[index]
            if power not in cache:
                cache[power] = variable_matrices[index] ** int(power)
            matrix = matrix * cache[power]
        matrices.append(matrix)
    return tuple(matrices)


def _multiplication_matrix(
    coordinates: sp.Matrix, basis_matrices: Sequence[sp.Matrix]
) -> sp.Matrix:
    """Build multiplication by an element from its quotient-basis coordinates."""
    if not basis_matrices:
        return sp.zeros(0, 0)
    dimension = basis_matrices[0].rows
    result = sp.zeros(dimension, dimension)
    for coefficient, matrix in zip(coordinates, basis_matrices, strict=True):
        if coefficient != 0:
            result += coefficient * matrix
    return result


def _monic_polynomial(poly: sp.Poly) -> sp.Poly:
    if poly.is_zero:
        raise RationalUnivariateError("zero polynomial cannot be normalized to monic form")
    return sp.Poly(poly.as_expr() / poly.LC(), *poly.gens, domain=poly.domain)


def _squarefree_part(poly: sp.Poly) -> sp.Poly:
    derivative = poly.diff()
    gcd = sp.gcd(poly, derivative)
    return _monic_polynomial(
        sp.Poly(
            sp.cancel(poly.as_expr() / gcd.as_expr()),
            *poly.gens,
            domain=poly.domain,
        )
    )


def _select_separating_linear_form(
    groebner_basis: sp.polys.polytools.GroebnerBasis,
    variables: Sequence[sp.Symbol],
    parameter: sp.Symbol,
    basis_exponents: Sequence[Sequence[int]],
    basis_matrices: Sequence[sp.Matrix],
    *,
    max_attempts: int = 64,
) -> tuple[sp.Expr, sp.Poly, sp.Poly, sp.Matrix, list[sp.Matrix], int]:
    dimension = len(basis_exponents)
    domain = groebner_basis.domain
    trace_of_basis_mult = sp.Matrix([matrix.trace() for matrix in basis_matrices])
    trace_pairing = sp.zeros(dimension, dimension)
    for left in range(dimension):
        for right in range(dimension):
            trace_pairing[left, right] = (basis_matrices[left] * basis_matrices[right]).trace()
    expected_distinct_roots = trace_pairing.rank()

    if max_attempts < 1:
        raise RationalUnivariateError("max_attempts must be positive")

    for attempt in range(1, max_attempts + 1):
        linear_form = sum((attempt**idx) * variable for idx, variable in enumerate(variables))
        remainder = _normal_form(groebner_basis, linear_form)
        coordinate_vector = _coefficient_vector(remainder, variables, basis_exponents, domain)
        multiplication = _multiplication_matrix(coordinate_vector, basis_matrices)
        characteristic = sp.Poly(
            multiplication.charpoly(parameter).as_expr(), parameter, domain=domain
        )
        squarefree = _squarefree_part(characteristic)
        if squarefree.degree() == expected_distinct_roots:
            derivative = characteristic.diff()
            gcd = sp.gcd(characteristic, derivative)
            denominator = sp.Poly(
                sp.cancel(derivative.as_expr() / gcd.as_expr()), parameter, domain=domain
            )
            powers = [sp.eye(dimension).col(0)]
            for _ in range(1, squarefree.degree()):
                powers.append(sp.simplify(multiplication * powers[-1]))
            return (
                linear_form,
                squarefree,
                denominator,
                trace_of_basis_mult,
                powers,
                expected_distinct_roots,
            )
    raise RationalUnivariateError("could not find a separating linear form")
