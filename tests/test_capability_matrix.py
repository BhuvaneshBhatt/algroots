import pytest
import sympy as sp

from algroots import algsolve, polysolve

x, y = sp.symbols("x y")


@pytest.mark.parametrize(
    ("name", "equations", "variables", "solver", "methods", "expected_count"),
    [
        ("linear", [x - 1, y + 2], (x, y), polysolve, ("auto", "shape", "action", "triangular"), 1),
        (
            "coupled_quadratic",
            [x - y**2, y**2 - 2],
            (x, y),
            polysolve,
            ("auto", "shape", "action", "triangular"),
            2,
        ),
        (
            "complex_only",
            [x**2 + 1, y - x],
            (x, y),
            polysolve,
            ("auto", "shape", "action", "triangular"),
            2,
        ),
        (
            "rational",
            [(x**2 - 1) / (x - 1), y - x],
            (x, y),
            algsolve,
            ("auto", "shape", "action", "triangular"),
            1,
        ),
        (
            "radical",
            [sp.sqrt(x) - (x - 2), y - x],
            (x, y),
            algsolve,
            ("auto", "shape", "action", "triangular"),
            1,
        ),
    ],
)
def test_solver_capability_matrix(name, equations, variables, solver, methods, expected_count):
    for method in methods:
        result = solver(equations, variables, method=method, digits=30, verification_digits=18)
        assert len(result.roots) == expected_count, (name, method, result.roots)
        assert result.max_relative_residual < sp.Float("1e-18")
