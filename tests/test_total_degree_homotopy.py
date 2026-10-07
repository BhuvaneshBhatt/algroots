import pytest
import sympy as sp

from algroots import algsolve, polysolve
from algroots.errors import HomotopySolveError, SystemSolveLimitError

x, y = sp.symbols("x y")


def _complex_roots(roots):
    return sorted(
        (tuple(complex(sp.N(value, 20)) for value in root) for root in roots),
        key=lambda root: tuple((round(value.real, 14), round(value.imag, 14)) for value in root),
    )


def test_total_degree_homotopy_matches_action_on_regular_square_system():
    equations = (x**2 - 1, y**2 - 1)
    homotopy = polysolve(equations, (x, y), method="homotopy", digits=35, recognize=False)
    action = polysolve(equations, (x, y), method="action", digits=35, recognize=False)
    assert homotopy.method == "total_degree_homotopy"
    assert homotopy.homotopy_paths_total == 4
    assert homotopy.homotopy_paths_succeeded == 4
    assert homotopy.homotopy_paths_failed == 0
    assert homotopy.homotopy_gamma is not None
    left = _complex_roots(homotopy.roots)
    right = _complex_roots(action.roots)
    assert len(left) == len(right) == 4
    for a, b in zip(left, right, strict=True):
        assert max(abs(u - v) for u, v in zip(a, b, strict=True)) < 1e-12


def test_total_degree_homotopy_handles_exact_algebraic_coefficients():
    result = polysolve(
        (x**2 - sp.sqrt(2),),
        (x,),
        method="homotopy",
        digits=35,
        recognize=False,
    )
    assert len(result.roots) == 2
    assert result.homotopy_paths_total == 2
    assert max(abs(complex(sp.N(root[0] ** 2 - sp.sqrt(2), 30))) for root in result.roots) < 1e-20


def test_total_degree_homotopy_is_not_selected_by_auto():
    result = polysolve((x**2 - 1,), (x,), method="auto", recognize=False)
    assert result.method != "total_degree_homotopy"
    assert result.homotopy_paths_total is None


def test_total_degree_homotopy_enforces_square_system():
    with pytest.raises(HomotopySolveError, match="square"):
        polysolve(
            (x**2 - 1, x - 1),
            (x,),
            method="homotopy",
            recognize=False,
        )


def test_total_degree_homotopy_rejects_singular_coalescing_endpoint():
    with pytest.raises(HomotopySolveError, match="singular"):
        polysolve(
            (x**2,),
            (x,),
            method="homotopy",
            digits=30,
            recognize=False,
        )


def test_total_degree_homotopy_path_limit_is_separate_from_solution_limit():
    with pytest.raises(SystemSolveLimitError, match="max_homotopy_paths"):
        polysolve(
            (x**4 - 1, y**4 - 1),
            (x, y),
            method="homotopy",
            max_homotopy_paths=8,
            max_solutions=100,
            recognize=False,
        )


def test_total_degree_homotopy_inconsistent_constant_equation_has_no_paths():
    result = polysolve(
        (sp.Integer(1),),
        (x,),
        method="homotopy",
        recognize=False,
    )
    assert result.roots == ()
    assert result.homotopy_paths_total == 0
    assert result.homotopy_paths_succeeded == 0
    assert result.homotopy_paths_failed == 0


def test_algebraic_solver_reports_current_homotopy_scope_for_radicals():
    with pytest.raises(HomotopySolveError, match="projective"):
        algsolve(
            (sp.sqrt(x) - 2,),
            (x,),
            method="homotopy",
            digits=30,
            recognize=False,
        )


def test_total_degree_homotopy_parallel_tracking_matches_serial():
    serial = polysolve(
        (x**2 - 1,),
        (x,),
        method="homotopy",
        digits=25,
        recognize=False,
    )
    parallel = polysolve(
        (x**2 - 1,),
        (x,),
        method="homotopy",
        digits=25,
        recognize=False,
        homotopy_parallel=True,
        homotopy_max_workers=2,
    )
    assert _complex_roots(serial.roots) == _complex_roots(parallel.roots)


def test_total_degree_homotopy_retries_gamma_after_path_failure(monkeypatch):
    import algroots.solver as solver_module
    from algroots.continuation import PathResult

    original = solver_module._track_total_degree_problem
    calls = 0

    def flaky(problem, gamma, options):
        nonlocal calls
        calls += 1
        if calls == 1:
            return tuple(
                PathResult(
                    start=tuple(start),
                    endpoint=tuple(start),
                    success=False,
                    final_t=0.5,
                    final_digits=35,
                    accepted_steps=1,
                    rejected_steps=1,
                    precision_increases=0,
                    max_relative_residual=sp.oo,
                    message="synthetic gamma failure",
                    classification="failed",
                )
                for start in problem.base.iter_start_points()
            )
        return original(problem, gamma, options)

    monkeypatch.setattr(solver_module, "_track_total_degree_problem", flaky)
    result = polysolve(
        (x**2 - 1,),
        (x,),
        method="homotopy",
        digits=30,
        homotopy_gamma_attempts=3,
        recognize=False,
    )
    assert calls == 2
    assert result.homotopy_gamma_attempts == 2
    assert result.homotopy_paths_failed == 0


def test_total_degree_homotopy_accepts_tiny_but_regular_jacobian():
    tiny = sp.Rational(1, 10**60)
    result = polysolve(
        (x**2 - tiny,),
        (x,),
        method="homotopy",
        digits=40,
        verification_digits=20,
        max_precision_digits=180,
        recognize=False,
    )
    assert len(result.roots) == 2
    assert result.homotopy_endpoint_smallest_singular_values is not None
    assert all(value != 0 for value in result.homotopy_endpoint_smallest_singular_values)


def test_total_degree_start_points_can_be_iterated_lazily():
    from algroots.total_degree_homotopy import build_total_degree_homotopy

    spec = build_total_degree_homotopy((x**3 - 1, y**2 - 1), (x, y))
    iterator = spec.iter_start_points()
    assert iter(iterator) is iterator
    first = next(iterator)
    assert len(first) == 2
    assert len(tuple(iterator)) == spec.path_count - 1


def test_algebraic_homotopy_exposes_tracking_controls_and_projection_counts():
    result = algsolve(
        (sp.sqrt(x) - (x - 2),),
        (x,),
        method="homotopy",
        digits=30,
        max_homotopy_paths=8,
        homotopy_seed=3,
        homotopy_gamma_attempts=3,
        recognize=False,
    )
    assert result.homotopy_paths_total is not None
    assert result.homotopy_projected_candidates is not None
    assert result.homotopy_projected_roots_rejected is not None
    assert result.homotopy_projected_roots_rejected >= 1
    assert len(result.roots) == 1


def test_total_degree_homotopy_gamma_retry_exhaustion_reports_attempts(monkeypatch):
    import algroots.solver as solver_module
    from algroots.continuation import PathResult

    def always_fail(problem, gamma, options):
        return tuple(
            PathResult(
                start=tuple(start),
                endpoint=tuple(start),
                success=False,
                final_t=0.5,
                final_digits=30,
                accepted_steps=1,
                rejected_steps=1,
                precision_increases=0,
                max_relative_residual=sp.oo,
                message="synthetic path failure",
                classification="failed",
            )
            for start in problem.base.iter_start_points()
        )

    monkeypatch.setattr(solver_module, "_track_total_degree_problem", always_fail)
    with pytest.raises(HomotopySolveError) as exc_info:
        polysolve(
            (x**2 - 1,),
            (x,),
            method="homotopy",
            digits=25,
            homotopy_gamma_attempts=3,
            recognize=False,
        )
    message = str(exc_info.value)
    assert "gamma attempt 1" in message
    assert "gamma attempt 2" in message
    assert "gamma attempt 3" in message
    assert "every deterministic gamma candidate" in message


def test_total_degree_homotopy_refuses_system_with_affine_path_at_infinity():
    # Bezout count is 2, but x*y - 1 == 0 and y - 1 == 0 have only the
    # affine root (1, 1); the missing projective path must not be ignored.
    with pytest.raises(HomotopySolveError, match="expected number of distinct roots|path"):
        polysolve(
            (x * y - 1, y - 1),
            (x, y),
            method="homotopy",
            digits=25,
            homotopy_gamma_attempts=1,
            recognize=False,
        )


def test_total_degree_homotopy_parallel_metadata_matches_serial_on_eight_paths():
    equations = (x**4 - 1, y**2 - 1)
    serial = polysolve(
        equations,
        (x, y),
        method="homotopy",
        digits=25,
        recognize=False,
        homotopy_gamma_attempts=2,
    )
    parallel = polysolve(
        equations,
        (x, y),
        method="homotopy",
        digits=25,
        recognize=False,
        homotopy_gamma_attempts=2,
        homotopy_parallel=True,
        homotopy_max_workers=2,
    )
    assert _complex_roots(serial.roots) == _complex_roots(parallel.roots)
    assert serial.homotopy_paths_total == parallel.homotopy_paths_total == 8
    assert serial.homotopy_paths_succeeded == parallel.homotopy_paths_succeeded == 8
    assert serial.homotopy_paths_failed == parallel.homotopy_paths_failed == 0
    assert serial.homotopy_paths_divergent == parallel.homotopy_paths_divergent == 0
    assert serial.homotopy_gamma == parallel.homotopy_gamma
    assert serial.homotopy_gamma_attempts == parallel.homotopy_gamma_attempts
