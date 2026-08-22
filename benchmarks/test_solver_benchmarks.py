"""Small, opt-in performance baselines for the main numerical routes.

Run with: python -m pytest benchmarks --benchmark-only
These benchmarks intentionally have no wall-clock assertions; comparison against a saved
baseline is more portable than absolute timing thresholds.
"""

import sympy as sp

from algroots import PathTrackerOptions, closed_additive_loop, polysolve, track_loop

x, y = sp.symbols("x y")


def test_benchmark_shape_solver(benchmark):
    equations = (x - y**2, y**3 - 2)
    result = benchmark(
        polysolve,
        equations,
        (x, y),
        method="shape",
        digits=35,
    )
    assert len(result.roots) == 3


def test_benchmark_action_solver(benchmark):
    equations = (x**2 - 1, y**2 - 1)
    result = benchmark(
        polysolve,
        equations,
        (x, y),
        method="action",
        digits=35,
    )
    assert len(result.roots) == 4


def test_benchmark_path_tracking(benchmark):
    loop = closed_additive_loop((x**2 - 1,), (x,), (2,))
    options = PathTrackerOptions(
        initial_step=0.03,
        max_step=0.07,
        initial_digits=40,
        residual_digits=24,
    )
    paths = benchmark(track_loop, loop, ((1,), (-1,)), options=options)
    assert all(path.success for path in paths)


def test_benchmark_rur_algebraic_degree_five(benchmark):
    result = benchmark(
        polysolve,
        (x**5 - sp.sqrt(2),),
        (x,),
        method="rur",
        digits=35,
        recognize=False,
    )
    assert len(result.roots) == 5


def test_benchmark_total_degree_homotopy(benchmark):
    result = benchmark(
        polysolve,
        (x**2 - 1, y**2 - 1),
        (x, y),
        method="homotopy",
        digits=30,
        recognize=False,
        homotopy_gamma_attempts=2,
    )
    assert len(result.roots) == 4
    assert result.homotopy_paths_total == 4


def test_benchmark_exact_border_basis(benchmark):
    from algroots import compute_border_basis

    result = benchmark(compute_border_basis, (x**2 - 1, y - x), (x, y))
    assert result.dimension == 2
    assert result.has_commuting_multiplication_matrices()


def test_benchmark_total_degree_homotopy_parallel_eight_paths(benchmark):
    result = benchmark(
        polysolve,
        (x**4 - 1, y**2 - 1),
        (x, y),
        method="homotopy",
        digits=25,
        recognize=False,
        homotopy_gamma_attempts=2,
        homotopy_parallel=True,
        homotopy_max_workers=2,
    )
    assert len(result.roots) == 8
    assert result.homotopy_paths_total == 8
