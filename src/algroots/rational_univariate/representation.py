from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property

import sympy as sp

from ..errors import PolynomialSystemError


class RationalUnivariateError(PolynomialSystemError):
    """Raised when a rational univariate representation cannot be computed."""


@dataclass(frozen=True)
class RationalUnivariateRepresentation:
    """Rational univariate representation of a zero-dimensional system.

    If ``parameter`` is named ``t``, each solution is represented by
    ``defining_polynomial(t) == 0`` and
    ``variable_i == coordinate_numerators[i](t) / coordinate_denominator(t)``.
    The implementation targets exact rational- or algebraic-coefficient,
    zero-dimensional polynomial systems and returns distinct algebraic solution branches.
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
        return (
            self.quotient_dimension
            if self.quotient_dimension is not None
            else len(self.standard_exponents)
        )

    @property
    def solution_count(self) -> int:
        return (
            self.geometric_solution_count
            if self.geometric_solution_count is not None
            else self.defining_polynomial.degree()
        )

    @property
    def is_empty(self) -> bool:
        return self.defining_polynomial.degree() <= 0

    def coordinate_expressions(self) -> tuple[sp.Expr, ...]:
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
        try:
            inverse_denominator = sp.invert(self.coordinate_denominator, self.defining_polynomial)
        except (sp.polys.polyerrors.NotInvertible, sp.PolynomialError, ValueError) as exc:
            raise RationalUnivariateError(
                "coordinate denominator is not invertible modulo the RUR polynomial"
            ) from exc
        return tuple(
            sp.Poly(
                (inverse_denominator * numerator).rem(self.defining_polynomial).as_expr(),
                self.parameter,
                domain=self.defining_polynomial.domain,
            )
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
        return self.representation.variables

    @property
    def coordinates(self) -> tuple[sp.Expr, ...]:
        t = self.representation.parameter
        return tuple(
            sp.cancel(poly.as_expr().subs(t, self.root))
            for poly in self.representation.normalized_coordinate_polynomials()
        )

    @property
    def assignment(self) -> Mapping[sp.Symbol, sp.Expr]:
        return dict(zip(self.variables, self.coordinates, strict=True))
