import pytest
import sympy as sp

from algroots import NotZeroDimensionalError, PolynomialSystemInputError, polysolve


def test_branch_dependent_triangular_system() -> None:
    x, y = sp.symbols("x y")
    equations = (x * y, x**2 - x, y**2 - y)

    result = polysolve(equations, (x, y), digits=40)

    assert result.method == "action_matrix"
    roots = {(round(float(sp.re(a)), 8), round(float(sp.re(b)), 8)) for a, b in result.roots}
    assert roots == {(0.0, 0.0), (0.0, 1.0), (1.0, 0.0)}


def test_inconsistent_system_returns_empty_result() -> None:
    x = sp.symbols("x")
    result = polysolve((x, x - 1), (x,))
    assert result.roots == ()


def test_positive_dimensional_system_is_rejected() -> None:
    x, y = sp.symbols("x y")
    with pytest.raises(NotZeroDimensionalError):
        polysolve((x * y,), (x, y))


def test_inexact_coefficient_is_rejected() -> None:
    x = sp.symbols("x")
    with pytest.raises(PolynomialSystemInputError):
        polysolve((x - 1.25,), (x,))


def test_transcendental_coefficient_is_rejected() -> None:
    x = sp.symbols("x")
    with pytest.raises(PolynomialSystemInputError):
        polysolve((x - sp.pi,), (x,))


def test_eq_input_is_supported() -> None:
    x = sp.symbols("x")
    result = polysolve((sp.Eq(x**2, 2),), (x,), digits=40)
    assert len(result.roots) == 2
