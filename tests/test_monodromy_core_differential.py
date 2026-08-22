"""Differential checks between monodromy tracking and the all-roots solver."""

import sympy as sp

from algroots import PathTrackerOptions, closed_additive_loop, polysolve, track_loop
from algroots.monodromy import deduplicate_roots, match_root

x, y = sp.symbols("x y")
OPTIONS = PathTrackerOptions(
    initial_step=0.02,
    max_step=0.05,
    initial_digits=45,
    residual_digits=28,
)


def _as_seed(root):
    return tuple(complex(sp.N(value, 35)) for value in root)


def test_multivariate_loop_endpoints_are_exactly_the_core_solver_root_set():
    equations = (x**2 - 1, y**2 - 4)
    solved = polysolve(equations, (x, y), digits=40)
    starts = tuple(_as_seed(root) for root in solved.roots)
    loop = closed_additive_loop(equations, (x, y), (2, 0))
    paths = track_loop(loop, starts, options=OPTIONS)

    assert all(path.success for path in paths)
    endpoints = deduplicate_roots(
        (path.endpoint for path in paths),
        tolerance=1e-8,
    )
    assert len(endpoints) == len(starts) == 4
    assert all(match_root(endpoint, starts, tolerance=1e-7) is not None for endpoint in endpoints)


def test_coupled_multivariate_loop_preserves_core_solver_root_set():
    equations = (x**2 + y - 2, y**2 - 1)
    solved = polysolve(equations, (x, y), digits=40)
    starts = tuple(_as_seed(root) for root in solved.roots)
    loop = closed_additive_loop(equations, (x, y), (0.35 + 0.2j, -0.25 + 0.15j))
    paths = track_loop(loop, starts, options=OPTIONS)

    assert all(path.success for path in paths)
    assert len(paths) == len(starts)
    assert all(match_root(path.endpoint, starts, tolerance=1e-7) is not None for path in paths)
