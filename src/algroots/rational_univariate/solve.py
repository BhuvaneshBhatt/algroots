from __future__ import annotations

from collections.abc import Iterable, Sequence

import sympy as sp

from .._validation import validate_integer
from ..errors import NumericalRootError
from ..univariate import numerical_univariate_roots
from .construction import compute_rational_univariate_representation
from .representation import (
    RationalUnivariateError,
    RationalUnivariatePoint,
    RationalUnivariateRepresentation,
)


def _require_constant_coefficients(representation):
    if representation.defining_polynomial.domain.is_FractionField or getattr(
        representation.defining_polynomial.domain, "is_AlgebraicFunctionField", False
    ):
        raise RationalUnivariateError(
            "specialize coefficient parameters and rebuild the RUR before extracting roots"
        )


def _all_parameter_roots(
    representation: RationalUnivariateRepresentation,
    *,
    real: bool,
) -> tuple[sp.Expr, ...]:
    _require_constant_coefficients(representation)
    defining = representation.defining_polynomial
    if defining.degree() <= 0:
        return tuple()
    try:
        roots = tuple(defining.all_roots())
    except (NotImplementedError, sp.PolynomialError, ValueError, TypeError) as exc:
        raise RationalUnivariateError(
            "exact enumeration of the RUR parameter roots failed"
        ) from exc
    if not real or not _real_coefficient_field(defining.domain):
        return roots
    return tuple(root for root in roots if root.is_real is True or sp.simplify(sp.im(root)) == 0)


def _real_coefficient_field(domain):
    return domain.is_QQ or domain.is_AlgebraicField and domain.ext.is_real is True


def _real_coordinates(coordinates):
    """Decide reality of algebraic coordinates without numerical tolerances."""
    for value in coordinates:
        if value.is_real is True:
            continue
        if value.is_real is False:
            return False
        imaginary = sp.simplify(sp.im(value))
        if imaginary.is_zero is True:
            continue
        if imaginary.is_zero is False:
            return False
        try:
            if not sp.to_number_field(imaginary).rep.is_zero:
                return False
        except (NotImplementedError, sp.PolynomialError, TypeError, ValueError) as exc:
            raise RationalUnivariateError("could not decide whether a coordinate is real") from exc
    return True


def _solution_coordinates(representation, real):
    """Evaluate parameter roots and enforce reality in the original coordinates.

    Over a real coefficient field a real parameter has real coordinate maps.
    Over a complex field neither implication holds: complex coordinates can
    have a real separator, and a complex separator can encode a real point.
    """
    maps = representation.normalized_coordinate_polynomials()
    check_coordinates = real and not _real_coefficient_field(
        representation.defining_polynomial.domain
    )
    for root in _all_parameter_roots(representation, real=real):
        coordinates = tuple(
            sp.cancel(poly.as_expr().subs(representation.parameter, root)) for poly in maps
        )
        if not check_coordinates or _real_coordinates(coordinates):
            yield root, coordinates


def numerical_rur_roots(
    representation: RationalUnivariateRepresentation,
    *,
    digits: int,
    maxsteps: int = 200,
    max_precision_digits: int | None = None,
) -> tuple[tuple[sp.Expr, ...], ...]:
    """Evaluate a RUR numerically without constructing exact algebraic roots.

    The exact RUR is retained, but its defining polynomial is solved numerically
    and the cached exact coordinate maps are evaluated at those parameter roots.
    Final multivariate verification belongs to the caller.
    """
    validate_integer(digits, "digits", minimum=15)
    validate_integer(maxsteps, "maxsteps")
    if max_precision_digits is not None:
        validate_integer(max_precision_digits, "max_precision_digits", minimum=digits)
    _require_constant_coefficients(representation)
    defining = representation.defining_polynomial
    if defining.degree() <= 0:
        return tuple()
    try:
        parameter_roots = tuple(
            numerical_univariate_roots(
                defining.as_expr(),
                representation.parameter,
                digits,
                maxsteps,
                max_precision_digits,
            )
        )
    except (sp.PolynomialError, ValueError, TypeError, NumericalRootError) as exc:
        raise RationalUnivariateError(
            "numerical solution of the RUR parameter polynomial failed"
        ) from exc
    t = representation.parameter
    coordinate_polys = representation.normalized_coordinate_polynomials()
    return tuple(
        tuple(sp.N(poly.as_expr().subs(t, root), digits) for poly in coordinate_polys)
        for root in parameter_roots
    )


def solve_rur_representation(
    representation: RationalUnivariateRepresentation,
    *,
    real: bool = True,
) -> tuple[tuple[sp.Expr, ...], ...]:
    """Return distinct exact solutions from an existing RUR."""

    _require_constant_coefficients(representation)
    if representation.defining_polynomial.degree() <= 0:
        return tuple()

    solutions: list[tuple[sp.Expr, ...]] = []
    seen_keys: set[tuple[str, ...]] = set()
    for _, point in _solution_coordinates(representation, real):
        key = tuple(sp.sstr(coord) for coord in point)
        if key in seen_keys:
            continue
        if any(
            all(sp.simplify(a - b) == 0 for a, b in zip(point, old, strict=True))
            for old in solutions
        ):
            continue
        seen_keys.add(key)
        solutions.append(point)
    return tuple(solutions)


def solve_zero_dimensional_system_with_rur(
    polynomials: Iterable[sp.Expr],
    variables: Sequence[sp.Symbol],
    *,
    real: bool = True,
    parameter: sp.Symbol | None = None,
    max_separating_attempts: int = 64,
    domain=None,
    groebner_backend="native",
    groebner_limits=None,
) -> tuple[tuple[sp.Expr, ...], ...]:
    """Return exact RUR solutions using native F4 or explicit portable SymPy construction."""

    representation = compute_rational_univariate_representation(
        polynomials,
        variables,
        parameter,
        max_separating_attempts=max_separating_attempts,
        domain=domain,
        groebner_backend=groebner_backend,
        groebner_limits=groebner_limits,
    )
    return solve_rur_representation(representation, real=real)


def solve_rur_points(
    representation: RationalUnivariateRepresentation,
    *,
    real: bool = True,
) -> tuple[RationalUnivariatePoint, ...]:
    """Return distinct solutions as RUR parameter-root points."""

    _require_constant_coefficients(representation)
    if representation.defining_polynomial.degree() <= 0:
        return tuple()
    points: list[RationalUnivariatePoint] = []
    seen: set[tuple[str, ...]] = set()
    for root, coordinates in _solution_coordinates(representation, real):
        point = RationalUnivariatePoint(representation, root)
        key = tuple(sp.sstr(coord) for coord in coordinates)
        if key in seen:
            continue
        seen.add(key)
        points.append(point)
    return tuple(points)
