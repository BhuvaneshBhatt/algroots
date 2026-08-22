"""Algebraization of exact algebraic equations into polynomial systems."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import sympy as sp

from ._validation import normalize_equation, validate_variables
from .errors import PolynomialSystemInputError


@dataclass(frozen=True)
class AlgebraizedSystem:
    """Polynomial representation of an algebraic equation system."""

    original_equations: tuple[Any, ...]
    polynomial_equations: tuple[Any, ...]
    original_variables: tuple[Any, ...]
    augmented_variables: tuple[Any, ...]
    auxiliary_variables: tuple[Any, ...]
    nonzero_constraints: tuple[Any, ...]
    auxiliary_lift_expressions: tuple[Any, ...] = ()


@dataclass
class _AlgebraizeState:
    original_variables: tuple[Any, ...]
    used_names: set[str]
    max_auxiliary_variables: int
    auxiliary_variables: list[Any] = field(default_factory=list)
    auxiliary_lift_expressions: list[Any] = field(default_factory=list)
    relations: list[Any] = field(default_factory=list)
    nonzero_constraints: list[Any] = field(default_factory=list)

    def new_auxiliary(self, lift_expression: Any) -> Any:
        if len(self.auxiliary_variables) >= self.max_auxiliary_variables:
            raise PolynomialSystemInputError(
                f"algebraization exceeds max_auxiliary_variables={self.max_auxiliary_variables}"
            )
        index = len(self.auxiliary_variables)
        while True:
            name = f"_algroots_aux{index}"
            index += 1
            if name not in self.used_names:
                break
        self.used_names.add(name)
        variable = sp.Symbol(name)
        self.auxiliary_variables.append(variable)
        self.auxiliary_lift_expressions.append(sp.sympify(lift_expression))
        return variable


def _validate_constant(expression: Any) -> None:
    if expression.is_number and expression.is_algebraic is not True:
        raise PolynomialSystemInputError(
            f"constants must be exact algebraic numbers; unsupported constant: {expression!r}"
        )


def _record_nonzero(expression: Any, state: _AlgebraizeState) -> None:
    numerator, denominator = sp.fraction(sp.together(expression))
    numerator = sp.expand(numerator)
    denominator = sp.expand(denominator)
    if numerator == 0:
        raise PolynomialSystemInputError("a denominator/base is identically zero")
    if numerator.free_symbols:
        state.nonzero_constraints.append(numerator)
    if denominator != 1 and denominator.free_symbols:
        state.nonzero_constraints.append(denominator)


def _clear_denominator(expression: Any, state: _AlgebraizeState) -> Any:
    numerator, denominator = sp.fraction(sp.together(expression))
    numerator = sp.expand(numerator)
    denominator = sp.expand(denominator)
    if denominator == 0:
        raise PolynomialSystemInputError("equation contains an identically zero denominator")
    if denominator != 1 and denominator.free_symbols:
        state.nonzero_constraints.append(denominator)
    return numerator


def _add_relation(expression: Any, state: _AlgebraizeState) -> None:
    state.relations.append(_clear_denominator(expression, state))


def _algebraize_expression(expression: Any, state: _AlgebraizeState) -> Any:
    expression = sp.sympify(expression)

    all_variables = set(state.original_variables) | set(state.auxiliary_variables)
    if expression.free_symbols.isdisjoint(all_variables):
        if expression.free_symbols:
            unknown = sorted(str(symbol) for symbol in expression.free_symbols)
            raise PolynomialSystemInputError(
                "equation contains undeclared symbols: " + ", ".join(unknown)
            )
        _validate_constant(expression)
        return expression

    if isinstance(expression, sp.Symbol):
        if expression not in all_variables:
            raise PolynomialSystemInputError(f"equation contains undeclared symbol {expression!r}")
        return expression

    if expression.is_Add:
        return sp.Add(*(_algebraize_expression(arg, state) for arg in expression.args))

    if expression.is_Mul:
        return sp.Mul(*(_algebraize_expression(arg, state) for arg in expression.args))

    if expression.is_Pow:
        base, exponent = expression.as_base_exp()
        if exponent.is_Integer:
            base_alg = _algebraize_expression(base, state)
            integer_exponent = int(exponent)
            if integer_exponent < 0:
                _record_nonzero(base_alg, state)
            return base_alg**integer_exponent
        if exponent.is_Rational:
            base_alg = _algebraize_expression(base, state)
            numerator = int(exponent.p)
            denominator = int(exponent.q)
            if denominator <= 1:
                return base_alg**numerator

            lift_expression = base_alg ** sp.Rational(numerator, denominator)
            auxiliary = state.new_auxiliary(lift_expression)
            if numerator > 0:
                relation = auxiliary**denominator - base_alg**numerator
            elif numerator < 0:
                relation = auxiliary**denominator * base_alg ** (-numerator) - 1
            else:
                return sp.Integer(1)
            _add_relation(relation, state)
            return auxiliary

    _validate_constant(expression)
    raise PolynomialSystemInputError(
        f"unsupported non-algebraic expression in equation: {expression!r}"
    )


def _validate_poly_equations(
    equations: Iterable[Any],
    variables: tuple[Any, ...],
) -> tuple[Any, ...]:
    normalized: list[Any] = []
    for equation in equations:
        try:
            polynomial = sp.Poly(sp.expand(equation), *variables, extension=True)
        except (sp.PolynomialError, TypeError, ValueError) as exc:
            raise PolynomialSystemInputError(
                "algebraization did not produce a polynomial system"
            ) from exc
        for coefficient in polynomial.coeffs():
            if coefficient.is_algebraic is not True:
                raise PolynomialSystemInputError(
                    "algebraized polynomial coefficients must be exact algebraic numbers; "
                    f"unsupported coefficient: {coefficient!r}"
                )
        normalized.append(polynomial.as_expr())
    return tuple(normalized)


def algebraize_system(
    equations: Iterable[Any],
    variables: Sequence[Any],
    *,
    max_auxiliary_variables: int = 32,
) -> AlgebraizedSystem:
    """Convert exact algebraic equalities to an augmented polynomial system.

    Rational functions are cleared to polynomial numerators while denominator
    nonvanishing constraints are retained. Rational powers that depend on the
    unknowns are represented by auxiliary variables and exact power relations.
    The original equations are retained because they define principal-branch
    semantics and are used to filter algebraic branches after polynomial solving.
    """
    if not isinstance(max_auxiliary_variables, int) or max_auxiliary_variables < 0:
        raise ValueError("max_auxiliary_variables must be a nonnegative integer")

    original_variables = validate_variables(variables)
    equation_list = tuple(equations)
    if not equation_list:
        raise PolynomialSystemInputError("at least one equation is required")

    original_equations = tuple(normalize_equation(eq) for eq in equation_list)
    used_names = {
        symbol.name for expression in original_equations for symbol in expression.free_symbols
    }
    used_names.update(variable.name for variable in original_variables)
    state = _AlgebraizeState(
        original_variables=original_variables,
        used_names=used_names,
        max_auxiliary_variables=max_auxiliary_variables,
    )

    transformed: list[Any] = []
    for expression in original_equations:
        transformed_expression = _algebraize_expression(expression, state)
        transformed.append(_clear_denominator(transformed_expression, state))

    nonzero_constraints = tuple(sp.expand(constraint) for constraint in state.nonzero_constraints)
    if nonzero_constraints:
        denominator_product = sp.prod(nonzero_constraints)
        saturation_var = state.new_auxiliary(1 / denominator_product)
        state.relations.append(sp.expand(saturation_var * denominator_product - 1))

    augmented_variables = original_variables + tuple(state.auxiliary_variables)
    polynomial_equations = _validate_poly_equations(
        (*transformed, *state.relations),
        augmented_variables,
    )
    return AlgebraizedSystem(
        original_equations=original_equations,
        polynomial_equations=polynomial_equations,
        original_variables=original_variables,
        augmented_variables=augmented_variables,
        auxiliary_variables=tuple(state.auxiliary_variables),
        nonzero_constraints=nonzero_constraints,
        auxiliary_lift_expressions=tuple(state.auxiliary_lift_expressions),
    )
