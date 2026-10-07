import pytest
import sympy as sp

from algroots import polysolve
from algroots.errors import ActionMatrixError


def _max_residual(equations, variables, root, digits=35):
    assignment = dict(zip(variables, root, strict=True))
    return max(abs(sp.N(eq.subs(assignment), digits)) for eq in equations)


def test_action_matrix_independent_quadratics() -> None:
    x, y = sp.symbols("x y")
    equations = (x**2 - 2, y**2 - 3)

    result = polysolve(equations, (x, y), digits=50, method="action")

    assert result.method == "action_matrix"
    assert result.quotient_dimension == 4
    assert result.standard_monomials == ((0, 0), (0, 1), (1, 0), (1, 1))
    assert result.separator_coeffs is not None
    assert len(result.roots) == 4
    for root in result.roots:
        assert _max_residual(equations, (x, y), root) < sp.Float("1e-35")


def test_auto_uses_action_matrix_when_shape_is_unavailable() -> None:
    x, y = sp.symbols("x y")
    result = polysolve((x**2 - 1, y**2 - 1), (x, y), digits=40)

    assert result.method == "action_matrix"
    assert result.quotient_dimension == 4
    roots = {(round(float(sp.re(a))), round(float(sp.re(b)))) for a, b in result.roots}
    assert roots == {(-1, -1), (-1, 1), (1, -1), (1, 1)}


def test_action_matrix_branch_dependent_radical_ideal() -> None:
    x, y = sp.symbols("x y")
    equations = (x * y, x**2 - x, y**2 - y)

    result = polysolve(equations, (x, y), digits=40, method="action")

    assert result.quotient_dimension == 3
    roots = {(round(float(sp.re(a))), round(float(sp.re(b)))) for a, b in result.roots}
    assert roots == {(0, 0), (0, 1), (1, 0)}


def test_action_matrix_exact_algebraic_coefficients() -> None:
    x, y = sp.symbols("x y")
    equations = (x**2 - 2, y**2 - sp.sqrt(3))

    result = polysolve(equations, (x, y), digits=45, method="action")

    assert len(result.roots) == 4
    assert result.quotient_dimension == 4
    for root in result.roots:
        assert _max_residual(equations, (x, y), root, 32) < sp.Float("1e-28")


def test_action_matrix_three_variables() -> None:
    x, y, z = sp.symbols("x y z")
    equations = (x**2 - 2, y**2 - 3, z - x * y)

    result = polysolve(equations, (x, y, z), digits=45, method="action")

    assert result.quotient_dimension == 4
    assert len(result.roots) == 4
    for root in result.roots:
        assert _max_residual(equations, (x, y, z), root, 32) < sp.Float("1e-28")


def test_nonradical_action_backend_declines() -> None:
    x, y = sp.symbols("x y")

    with pytest.raises(ActionMatrixError):
        polysolve((x**2, y), (x, y), digits=40, method="action")


def test_auto_falls_back_for_nonradical_ideal() -> None:
    x, y = sp.symbols("x y")

    result = polysolve((x**2, y), (x, y), digits=40)

    assert result.method == "shape_position"
    assert len(result.roots) == 1
    assert all(abs(sp.N(value, 30)) < sp.Float("1e-25") for value in result.roots[0])


def test_explicit_triangular_backend_remains_available() -> None:
    x, y = sp.symbols("x y")

    result = polysolve(
        (x**2 - 1, y**2 - 1),
        (x, y),
        digits=40,
        method="triangular",
    )

    assert result.method == "triangular_groebner"
    assert len(result.roots) == 4


def test_action_dimension_budget_is_enforced() -> None:
    x, y = sp.symbols("x y")
    equations = (x**2 - 1, y**2 - 1)

    with pytest.raises(ActionMatrixError):
        polysolve(
            equations,
            (x, y),
            method="action",
            max_action_dimension=3,
        )

    fallback = polysolve(
        equations,
        (x, y),
        method="auto",
        max_action_dimension=3,
    )
    assert fallback.method == "rational_univariate"
    assert len(fallback.roots) == 4


def test_action_matrix_complex_roots() -> None:
    x, y = sp.symbols("x y")
    equations = (x**2 + 1, y**2 + 2)

    result = polysolve(equations, (x, y), digits=45, method="action")

    assert len(result.roots) == 4
    assert result.quotient_dimension == 4
    assert any(abs(sp.im(root[0])) > 0 for root in result.roots)
    assert any(abs(sp.im(root[1])) > 0 for root in result.roots)
    for root in result.roots:
        assert _max_residual(equations, (x, y), root, 32) < sp.Float("1e-28")
