"""Shared lightweight input and option validation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import sympy as sp

from .errors import PolynomialSystemInputError


def validate_variables(variables: Sequence[Any]) -> tuple[Any, ...]:
    """Validate and normalize a nonempty sequence of distinct SymPy symbols."""
    vars_tuple = tuple(variables)
    if not vars_tuple:
        raise PolynomialSystemInputError("at least one variable is required")
    if len(set(vars_tuple)) != len(vars_tuple):
        raise PolynomialSystemInputError("variables must be distinct")
    if any(not isinstance(var, sp.Symbol) for var in vars_tuple):
        raise PolynomialSystemInputError("variables must be SymPy Symbol objects")
    return vars_tuple


def normalize_equation(equation: Any, *, expand: bool = False) -> Any:
    """Convert an equality/expression to a zero-form expression."""
    equation = sp.sympify(equation)
    if equation is sp.true:
        expression = sp.Integer(0)
    elif equation is sp.false:
        expression = sp.Integer(1)
    elif isinstance(equation, sp.Equality):
        expression = equation.lhs - equation.rhs
    elif equation.is_Relational:
        raise PolynomialSystemInputError(
            "only equalities are supported; inequalities are outside algroots' scope"
        )
    else:
        expression = equation
    return sp.expand(expression) if expand else expression


def validate_recognition_options(recognize: bool, max_degree: int) -> None:
    """Validate common automatic-recognition options."""
    if not isinstance(recognize, bool):
        raise TypeError("recognize must be bool")
    if not isinstance(max_degree, int) or max_degree <= 0:
        raise ValueError("recognition_max_degree must be a positive integer")
