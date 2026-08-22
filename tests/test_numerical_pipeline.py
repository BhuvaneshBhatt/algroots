import mpmath as mp
import pytest
import sympy as sp

from algroots import polysolve
from algroots.numerical import (
    NumericExpression,
    compile_polynomials,
    compiled_jacobian,
    compiled_residual,
    newton_refine,
    sympy_to_mpc,
)


def test_compiled_residual_preserves_high_precision() -> None:
    x = sp.symbols("x")
    polynomials = compile_polynomials((x**2 - 2,), (x,))
    root = (sympy_to_mpc(sp.N(sp.sqrt(2), 80), 80),)

    with mp.workdps(80):
        assert compiled_residual(polynomials, root, 75) < mp.mpf("1e-70")


def test_compiled_newton_refines_without_sympy_nsolve() -> None:
    x = sp.symbols("x")
    expressions = (x**2 - 2,)
    polynomials = compile_polynomials(expressions, (x,))
    jacobian = compiled_jacobian(expressions, (x,))

    refined, succeeded = newton_refine(
        polynomials,
        jacobian,
        (mp.mpc("1.4"),),
        digits=60,
        maxsteps=50,
    )

    assert succeeded
    with mp.workdps(60):
        assert abs(refined[0] - mp.sqrt(2)) < mp.mpf("1e-50")


def test_compiled_rational_power_uses_principal_branch() -> None:
    x = sp.symbols("x")
    expression = sp.Pow(x, sp.Rational(2, 3), evaluate=False) - 4
    evaluator = NumericExpression(expression, (x,))

    with mp.workdps(60):
        assert abs(evaluator.evaluate((mp.mpc(8),), 55)) < mp.mpf("1e-50")
        assert abs(evaluator.evaluate((mp.mpc(-8),), 55)) > 1


def test_max_precision_option_is_validated() -> None:
    x = sp.symbols("x")
    with pytest.raises(ValueError, match="max_precision_digits"):
        polysolve((x**2 - 2,), (x,), digits=50, max_precision_digits=40)


def test_flint_action_and_shape_backends() -> None:
    pytest.importorskip("flint")
    x, y = sp.symbols("x y")

    action = polysolve(
        (x**2 - 2, y**2 - 3),
        (x, y),
        digits=45,
        method="action",
    )
    assert action.method == "action_matrix"
    assert len(action.roots) == 4

    shape = polysolve(
        (x - y**2, y**3 - 2),
        (x, y),
        digits=45,
        method="shape",
    )
    assert shape.method == "shape_position"
    assert len(shape.roots) == 3
