import cmath

import mpmath as mp
import pytest
import sympy as sp

from algroots import (
    PathTrackerOptions,
    closed_additive_loop,
    monodromy_permutation,
    track_loop,
)

x = sp.symbols("x")
OPTIONS = PathTrackerOptions(
    initial_step=0.025,
    max_step=0.06,
    initial_digits=45,
    residual_digits=28,
)


def test_closed_additive_loop_is_closed_at_both_endpoints():
    loop = closed_additive_loop([x**2 - 1], (x,), (2,))
    homotopy = loop.homotopy()
    with mp.workdps(50):
        for point in ((1,), (-1,)):
            assert max(abs(v) for v in homotopy.evaluate(point, 0)) < mp.mpf("1e-40")
            assert max(abs(v) for v in homotopy.evaluate(point, 1)) < mp.mpf("1e-40")


def test_quadratic_loop_swaps_the_two_roots():
    roots = ((1,), (-1,))
    loop = closed_additive_loop([x**2 - 1], (x,), (2,))
    result = monodromy_permutation(
        loop,
        roots,
        options=OPTIONS,
        match_tolerance=1e-8,
    )
    assert result.permutation == (1, 0)
    assert all(path.success for path in result.paths)


def test_cubic_loop_cycles_three_roots():
    omega = cmath.exp(2j * cmath.pi / 3)
    roots = ((1,), (omega,), (omega**2,))
    loop = closed_additive_loop([x**3 - 1], (x,), (2,))
    result = monodromy_permutation(
        loop,
        roots,
        options=OPTIONS,
        match_tolerance=1e-7,
    )
    assert result.permutation == (1, 2, 0)


def test_parallel_tracking_matches_serial_endpoints():
    roots = ((1,), (-1,))
    loop = closed_additive_loop([x**2 - 1], (x,), (2,))
    serial = track_loop(loop, roots, options=OPTIONS)
    parallel = track_loop(loop, roots, options=OPTIONS, parallel=True, max_workers=2)
    for left, right in zip(serial, parallel, strict=True):
        assert left.success and right.success
        assert abs(left.endpoint[0] - right.endpoint[0]) < 1e-12


def test_loop_rejects_nonsquare_or_nonpolynomial_system():
    y = sp.symbols("y")
    with pytest.raises(ValueError):
        closed_additive_loop([x**2 - 1], (x, y), (1,))
    with pytest.raises(ValueError):
        closed_additive_loop([sp.sin(x)], (x,), (1,))


def test_multivariate_loop_induces_expected_pairwise_permutation():
    y = sp.symbols("y")
    roots = ((1, 2), (1, -2), (-1, 2), (-1, -2))
    # Only the x-equation winds around its discriminant; y stays on its branch.
    loop = closed_additive_loop([x**2 - 1, y**2 - 4], (x, y), (2, 0))
    result = monodromy_permutation(
        loop,
        roots,
        options=OPTIONS,
        match_tolerance=1e-7,
    )
    assert all(path.success for path in result.paths)
    assert result.permutation == (2, 3, 0, 1)


def test_ambiguous_endpoint_matching_does_not_choose_arbitrarily():
    roots = ((1.0,), (1.0 + 1e-9,))
    loop = closed_additive_loop([x - 1], (x,), (0,))
    result = monodromy_permutation(
        loop,
        roots,
        options=OPTIONS,
        match_tolerance=1e-8,
    )
    assert result.permutation == (None, None)
