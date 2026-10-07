import pytest
import sympy as sp

from algroots import polysolve
from algroots.errors import ShapePositionError


def _residuals(equations, variables, root, digits=40):
    assignment = dict(zip(variables, root, strict=True))
    return [abs(sp.N(eq.subs(assignment), digits)) for eq in equations]


def test_shape_position_cubic_system() -> None:
    x, y = sp.symbols("x y")
    equations = (x - y**2, y**3 - 2)

    result = polysolve(equations, (x, y), digits=50, method="shape")

    assert result.method == "shape_position"
    assert result.parameter_variable == y
    assert sp.Poly(result.eliminant, y).degree() == 3
    assert len(result.roots) == 3
    for root in result.roots:
        assert max(_residuals(equations, (x, y), root)) < sp.Float("1e-35")


def test_shape_position_linear_relation_with_nonunit_coefficient() -> None:
    x, y = sp.symbols("x y")
    equations = (2 * x - y, y**2 - 2)

    result = polysolve(equations, (x, y), digits=45, method="shape")

    assert result.method == "shape_position"
    assert len(result.roots) == 2
    for x_value, y_value in result.roots:
        assert abs(sp.N(2 * x_value - y_value, 35)) < sp.Float("1e-30")


def test_shape_position_exact_algebraic_coefficient() -> None:
    x, y = sp.symbols("x y")
    equations = (x - sp.sqrt(2) * y, y**2 - 3)

    result = polysolve(equations, (x, y), digits=45)

    assert result.method == "shape_position"
    assert len(result.roots) == 2
    for root in result.roots:
        assert max(_residuals(equations, (x, y), root)) < sp.Float("1e-30")


def test_shape_position_three_variables() -> None:
    x, y, z = sp.symbols("x y z")
    equations = (x - z**2, y - z**3, z**4 - 2)

    result = polysolve(equations, (x, y, z), digits=50, method="shape")

    assert result.method == "shape_position"
    assert result.parameter_variable == z
    assert len(result.roots) == 4
    for root in result.roots:
        assert max(_residuals(equations, (x, y, z), root)) < sp.Float("1e-35")


def test_shape_method_rejects_nonshape_basis() -> None:
    x, y = sp.symbols("x y")
    with pytest.raises(ShapePositionError):
        polysolve(
            (x**2 - 1, y**2 - 1),
            (x, y),
            method="shape",
        )


def test_auto_uses_action_matrix_after_shape_position() -> None:
    x, y = sp.symbols("x y")
    equations = (x**2 - 1, y**2 - 1)

    result = polysolve(equations, (x, y), digits=40)

    assert result.method == "action_matrix"
    assert len(result.roots) == 4


def test_shape_solution_count_matches_squarefree_eliminant() -> None:
    x, y = sp.symbols("x y")
    equations = (x - y, (y - 1) ** 2 * (y + 2))

    result = polysolve(equations, (x, y), digits=40)

    assert result.method == "shape_position"
    assert len(result.roots) == 2
    approx = {(round(float(sp.re(a)), 8), round(float(sp.re(b)), 8)) for a, b in result.roots}
    assert approx == {(1.0, 1.0), (-2.0, -2.0)}
