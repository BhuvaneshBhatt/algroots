import sympy as sp

from algroots.continuation import PathTrackerOptions
from algroots.monodromy import discover_monodromy_orbit
from algroots.monodromy_stopping import (
    capture_recapture_estimate,
    second_order_trace_test,
)

x = sp.symbols("x")
OPTIONS = PathTrackerOptions(
    initial_step=0.03,
    max_step=0.07,
    initial_digits=40,
    residual_digits=24,
)


def test_second_order_trace_distinguishes_complete_quadratic_fiber():
    complete = second_order_trace_test([x**2 - 1], (x,), [(1,), (-1,)], direction=(1,), digits=40)
    incomplete = second_order_trace_test([x**2 - 1], (x,), [(1,)], direction=(1,), digits=40)
    assert complete < 1e-25
    assert incomplete > 1e-3


def test_capture_recapture_estimate_for_full_recapture_is_exact():
    estimate = capture_recapture_estimate(2, 2, 2, confidence=0.95)
    assert estimate.estimate == 2
    assert estimate.standard_deviation == 0
    assert estimate.lower == estimate.upper == 2


def test_capture_recapture_with_no_overlap_predicts_larger_population():
    estimate = capture_recapture_estimate(2, 2, 0)
    assert estimate.estimate > 2
    assert estimate.upper > estimate.estimate


def test_exact_expected_root_count_stops_orbit_discovery():
    result = discover_monodromy_orbit(
        [x**2 - 1],
        (x,),
        [(1,)],
        max_loops=8,
        radius=2.0,
        random_seed=0,
        options=OPTIONS,
        expected_root_count=2,
    )
    assert len(result.roots) == 2
    assert result.stopping_reason == "expected_root_count"
    assert result.completeness_basis == "exact_root_count"
    assert result.loops_completed < 8


def test_trace_test_can_stop_closed_known_orbit():
    result = discover_monodromy_orbit(
        [x**2 - 1],
        (x,),
        [(1,), (-1,)],
        max_loops=5,
        radius=2.0,
        random_seed=0,
        options=OPTIONS,
        trace_test=True,
        trace_tolerance=1e-10,
        min_loops_before_stopping=1,
    )
    assert result.stopping_reason == "trace_test"
    assert result.completeness_basis == "none"
    assert result.trace_residual is not None
    assert result.trace_residual < 1e-10


def test_statistical_stop_records_statistical_basis_not_exact_completion():
    result = discover_monodromy_orbit(
        [x**2 - 1],
        (x,),
        [(1,), (-1,)],
        max_loops=5,
        radius=2.0,
        random_seed=0,
        options=OPTIONS,
        statistical_stop=True,
        confidence=0.95,
        min_loops_before_stopping=1,
    )
    assert result.stopping_reason == "statistical"
    assert result.completeness_basis == "statistical"
    assert result.population_estimate is not None
    assert result.population_estimate.upper <= len(result.roots) + 0.5
    assert not hasattr(result, "complete")


def test_complete_nonlinear_fiber_can_have_nonzero_curvature():
    y = sp.Symbol("y")
    residual = second_order_trace_test(
        (x - y**4, y**2 - 1), (x, y), ((1, 1), (1, -1)), direction=(0, 1)
    )
    assert residual == 2.0


def test_incomplete_fiber_can_have_zero_curvature():
    y = sp.Symbol("y")
    residual = second_order_trace_test(
        (x**2 - 1, y**2 - 1), (x, y), ((1, 1), (-1, -1)), direction=(1, 1)
    )
    assert residual < 1e-25
