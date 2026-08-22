from __future__ import annotations

from collections.abc import Iterable, Sequence

import sympy as sp
from mpmath.libmp.libhyper import NoConvergence

from .construction import compute_rational_univariate_representation
from .representation import (
    RationalUnivariateError,
    RationalUnivariatePoint,
    RationalUnivariateRepresentation,
)


def _all_parameter_roots(
    representation: RationalUnivariateRepresentation,
    *,
    real: bool,
) -> tuple[sp.Expr, ...]:
    defining = representation.defining_polynomial
    if defining.degree() <= 0:
        return tuple()
    try:
        roots = tuple(defining.all_roots())
    except (NotImplementedError, sp.PolynomialError, ValueError, TypeError) as exc:
        raise RationalUnivariateError(
            "exact enumeration of the RUR parameter roots failed"
        ) from exc
    if not real:
        return roots
    return tuple(root for root in roots if root.is_real is True or sp.simplify(sp.im(root)) == 0)


def numerical_rur_roots(
    representation: RationalUnivariateRepresentation,
    *,
    digits: int,
    maxsteps: int = 200,
) -> tuple[tuple[sp.Expr, ...], ...]:
    """Evaluate a RUR numerically without constructing exact algebraic roots.

    The exact RUR is retained, but its defining polynomial is solved numerically
    and the cached exact coordinate maps are evaluated at those parameter roots.
    Final multivariate verification belongs to the caller.
    """
    if not isinstance(digits, int) or digits < 15:
        raise ValueError("digits must be an integer >= 15")
    defining = representation.defining_polynomial
    if defining.degree() <= 0:
        return tuple()
    try:
        parameter_roots = tuple(sp.nroots(defining.as_expr(), n=digits, maxsteps=maxsteps))
    except (sp.PolynomialError, ValueError, TypeError, NoConvergence) as exc:
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

    if representation.defining_polynomial.degree() <= 0:
        return tuple()

    t = representation.parameter
    coordinate_polys = representation.normalized_coordinate_polynomials()
    roots = _all_parameter_roots(representation, real=real)
    solutions: list[tuple[sp.Expr, ...]] = []
    seen_keys: set[tuple[str, ...]] = set()
    for root in roots:
        point = tuple(sp.cancel(poly.as_expr().subs(t, root)) for poly in coordinate_polys)
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
) -> tuple[tuple[sp.Expr, ...], ...]:
    """Return distinct exact solutions obtained from a RUR representation."""

    representation = compute_rational_univariate_representation(
        polynomials, variables, parameter, max_separating_attempts=max_separating_attempts
    )
    return solve_rur_representation(representation, real=real)


def solve_rur_points(
    representation: RationalUnivariateRepresentation,
    *,
    real: bool = True,
) -> tuple[RationalUnivariatePoint, ...]:
    """Return distinct solutions as RUR parameter-root points."""

    if representation.defining_polynomial.degree() <= 0:
        return tuple()
    roots = _all_parameter_roots(representation, real=real)
    points: list[RationalUnivariatePoint] = []
    seen: set[tuple[str, ...]] = set()
    for root in roots:
        point = RationalUnivariatePoint(representation, root)
        key = tuple(sp.sstr(coord) for coord in point.coordinates)
        if key in seen:
            continue
        seen.add(key)
        points.append(point)
    return tuple(points)
