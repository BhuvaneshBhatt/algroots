import sympy as sp

from algroots import RootDiagnostics, polysolve
from algroots.solver import _validate_roots


def test_validator_refines_inexact_candidate() -> None:
    x = sp.symbols("x")
    roots, diagnostics, max_residual = _validate_roots(
        [(sp.Float("1.4", 20),)],
        (x**2 - 2,),
        (x,),
        digits=40,
        work_digits=60,
        verification_digits=30,
        maxsteps=100,
        expected_count=1,
    )

    assert len(roots) == 1
    assert abs(sp.N(roots[0][0] - sp.sqrt(2), 35)) < sp.Float("1e-35")
    assert diagnostics[0].refinement_attempted
    assert diagnostics[0].refinement_succeeded
    assert diagnostics[0].refinement_improved
    assert diagnostics[0].final_relative_residual < diagnostics[0].initial_relative_residual
    assert max_residual <= sp.Float("1e-30")


def test_shape_backend_uses_common_validation() -> None:
    x, y = sp.symbols("x y")
    result = polysolve(
        [x - y**2, y**3 - 2],
        (x, y),
        digits=40,
        guard_digits=15,
        verification_digits=25,
        method="shape",
    )

    assert result.working_digits == 55
    assert len(result.diagnostics) == len(result.roots) == 3
    assert all(isinstance(item, RootDiagnostics) for item in result.diagnostics)
    assert all(item.final_relative_residual <= sp.Float("1e-25") for item in result.diagnostics)


def test_action_backend_uses_common_validation() -> None:
    x, y = sp.symbols("x y")
    result = polysolve(
        [x**2 - 2, y**2 - 3],
        (x, y),
        digits=35,
        guard_digits=15,
        verification_digits=20,
        method="action",
    )

    assert result.working_digits >= 50
    assert len(result.diagnostics) == len(result.roots) == 4
    assert all(item.final_relative_residual <= sp.Float("1e-20") for item in result.diagnostics)


def test_triangular_backend_uses_common_validation() -> None:
    x, y = sp.symbols("x y")
    result = polysolve(
        [x * y, x**2 - x, y**2 - y],
        (x, y),
        digits=35,
        guard_digits=15,
        verification_digits=20,
        method="triangular",
    )

    assert len(result.diagnostics) == len(result.roots) == 3
    assert all(item.final_relative_residual <= sp.Float("1e-20") for item in result.diagnostics)


def test_overdetermined_system_refines_and_verifies() -> None:
    x, y = sp.symbols("x y")
    result = polysolve(
        [x + y - 1, x - y, 2 * x - 1],
        (x, y),
        digits=40,
        guard_digits=20,
        verification_digits=25,
    )

    assert len(result.roots) == 1
    root = result.roots[0]
    assert abs(root[0] - sp.Rational(1, 2)) < sp.Float("1e-35")
    assert abs(root[1] - sp.Rational(1, 2)) < sp.Float("1e-35")
    assert result.max_relative_residual <= sp.Float("1e-25")


def test_validator_skips_unneeded_refinement() -> None:
    x = sp.symbols("x")
    candidate = (sp.N(sp.sqrt(2), 60),)
    roots, diagnostics, _ = _validate_roots(
        [candidate],
        (x**2 - 2,),
        (x,),
        digits=40,
        work_digits=60,
        verification_digits=25,
        maxsteps=100,
        expected_count=1,
    )

    assert len(roots) == 1
    assert not diagnostics[0].refinement_attempted
    assert not diagnostics[0].refinement_succeeded
    assert diagnostics[0].final_relative_residual <= sp.Float("1e-25")
