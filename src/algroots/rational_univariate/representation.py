from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property

import sympy as sp

from ..domains import polynomial_domain
from ..errors import PolynomialSystemError


class RationalUnivariateError(PolynomialSystemError):
    """Raised when a rational univariate representation cannot be computed."""


@dataclass(frozen=True)
class RationalUnivariateRepresentation:
    """Rational univariate representation of a zero-dimensional system.

    If ``parameter`` is named ``t``, each solution is represented by
    ``defining_polynomial(t) == 0`` and
    ``variable_i == coordinate_numerators[i](t) / coordinate_denominator(t)``.
    Construction supports exact characteristic-zero coefficient fields, including
    rational parameter fields. The latter describe generic fibers; specialize
    original equations before using root enumeration or making real-fiber claims.
    """

    variables: tuple[sp.Symbol, ...]
    parameter: sp.Symbol
    defining_polynomial: sp.Poly
    coordinate_denominator: sp.Poly
    coordinate_numerators: tuple[sp.Poly, ...]
    separating_linear_form: sp.Expr
    standard_exponents: tuple[tuple[int, ...], ...]
    quotient_dimension: int | None = None
    geometric_solution_count: int | None = None

    @property
    def dimension(self) -> int:
        """The vector-space dimension, including nonreduced multiplicities."""
        return (
            self.quotient_dimension
            if self.quotient_dimension is not None
            else len(self.standard_exponents)
        )

    @property
    def solution_count(self) -> int:
        """The number of distinct geometric points represented by the squarefree polynomial."""
        return (
            self.geometric_solution_count
            if self.geometric_solution_count is not None
            else self.defining_polynomial.degree()
        )

    @property
    def is_empty(self) -> bool:
        """Whether the representation describes an empty solution set."""
        return self.defining_polynomial.degree() <= 0

    def coordinate_expressions(self) -> tuple[sp.Expr, ...]:
        """Return the original rational coordinate maps before modular denominator inversion."""
        denominator = self.coordinate_denominator.as_expr()
        return tuple(
            sp.cancel(numer.as_expr() / denominator) for numer in self.coordinate_numerators
        )

    @cached_property
    def _normalized_coordinate_polynomials(self) -> tuple[sp.Poly, ...]:
        if self.is_empty:
            return tuple(
                sp.Poly(0, self.parameter, domain=self.defining_polynomial.domain)
                for _ in self.variables
            )
        domain = self.defining_polynomial.domain
        arithmetic = polynomial_domain(domain)
        defining = self.defining_polynomial.set_domain(arithmetic)
        denominator = self.coordinate_denominator.set_domain(arithmetic)
        try:
            inverse_denominator = denominator.invert(defining)
        except (sp.polys.polyerrors.NotInvertible, sp.PolynomialError, ValueError) as exc:
            raise RationalUnivariateError(
                "coordinate denominator is not invertible modulo the RUR polynomial"
            ) from exc
        return tuple(
            (inverse_denominator * numerator.set_domain(arithmetic))
            .rem(defining)
            .set_domain(domain)
            for numerator in self.coordinate_numerators
        )

    def normalized_coordinate_polynomials(self) -> tuple[sp.Poly, ...]:
        """Return cached denominator-free coordinate polynomials modulo the RUR polynomial."""
        return self._normalized_coordinate_polynomials


@dataclass(frozen=True)
class RationalUnivariatePoint:
    """A point represented by a RUR parameter root."""

    representation: RationalUnivariateRepresentation
    root: sp.Expr

    @property
    def variables(self) -> tuple[sp.Symbol, ...]:
        """Coordinate symbols in their declared order."""
        return self.representation.variables

    @property
    def coordinates(self) -> tuple[sp.Expr, ...]:
        """Evaluate the denominator-free coordinate maps at the represented parameter root."""
        t = self.representation.parameter
        return tuple(
            sp.cancel(poly.as_expr().subs(t, self.root))
            for poly in self.representation.normalized_coordinate_polynomials()
        )

    @property
    def assignment(self) -> Mapping[sp.Symbol, sp.Expr]:
        """Map each coordinate symbol to its exact represented value."""
        return dict(zip(self.variables, self.coordinates, strict=True))
