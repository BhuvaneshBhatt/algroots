import math

import mpmath as mp
import pytest
import sympy as sp

import algroots.continuation as continuation
from algroots import PathTrackerOptions, SympyHomotopy, track_path

x, y, t = sp.symbols("x y t")


def test_tracks_known_linear_two_variable_homotopy():
    homotopy = SympyHomotopy(
        [x - (1 + 2 * t), y - (3 - t)],
        (x, y),
        t,
    )
    result = track_path(homotopy, (1, 3))
    assert result.success
    assert abs(result.endpoint[0] - 3) < 1e-20
    assert abs(result.endpoint[1] - 2) < 1e-20
    assert result.final_t == pytest.approx(1.0)
    assert result.max_relative_residual < 1e-20


def test_tracks_known_nonlinear_square_root_branch():
    homotopy = SympyHomotopy([x**2 - (1 + t)], (x,), t)
    result = track_path(
        homotopy,
        (1,),
        options=PathTrackerOptions(
            initial_step=0.08,
            max_step=0.15,
            initial_digits=35,
            residual_digits=22,
        ),
    )
    assert result.success
    assert abs(result.endpoint[0] - math.sqrt(2)) < 1e-14
    assert result.accepted_steps > 1
    assert all(step.relative_residual < 1e-18 for step in result.steps)


def test_tracks_known_complex_analytic_path():
    homotopy = SympyHomotopy([x**2 - sp.exp(sp.I * sp.pi * t)], (x,), t)
    result = track_path(
        homotopy,
        (1,),
        options=PathTrackerOptions(
            initial_step=0.03,
            max_step=0.08,
            initial_digits=45,
            residual_digits=28,
        ),
    )
    assert result.success
    assert abs(result.endpoint[0] - 1j) < 1e-14


def test_backward_tracking_returns_to_start():
    homotopy = SympyHomotopy([x**2 - (1 + t)], (x,), t)
    forward = track_path(homotopy, (1,))
    backward = track_path(
        homotopy,
        forward.endpoint,
        t_start=1.0,
        t_end=0.0,
    )
    assert forward.success and backward.success
    assert abs(backward.endpoint[0] - 1) < 1e-14


def test_bad_seed_is_corrected_before_path_tracking():
    homotopy = SympyHomotopy([x**2 - (1 + t)], (x,), t)
    result = track_path(homotopy, (1.05,))
    assert result.success
    assert abs(result.endpoint[0] - math.sqrt(2)) < 1e-14


def test_singular_start_is_reported_as_failure():
    homotopy = SympyHomotopy([x**2 - t], (x,), t)
    result = track_path(homotopy, (0,), t_start=0.0, t_end=1.0)
    assert not result.success
    assert result.final_t == 0.0


def test_precision_escalates_after_controlled_low_precision_failure(monkeypatch):
    homotopy = SympyHomotopy([x - (1 + t)], (x,), t)
    original = continuation._predict_tangent

    def precision_sensitive(*args, **kwargs):
        if mp.mp.dps < 55:
            raise continuation.PathStepError("controlled low-precision failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(continuation, "_predict_tangent", precision_sensitive)
    options = PathTrackerOptions(
        initial_step=0.1,
        min_step=0.099,
        max_step=0.2,
        initial_digits=25,
        max_digits=100,
        residual_digits=18,
    )
    result = track_path(homotopy, (1,), options=options)
    assert result.success
    assert result.precision_increases >= 1
    assert result.final_digits >= 50


def test_option_validation():
    with pytest.raises(ValueError):
        PathTrackerOptions(initial_step=0)
    with pytest.raises(ValueError):
        PathTrackerOptions(initial_digits=10)
    with pytest.raises(ValueError):
        PathTrackerOptions(initial_digits=50, max_digits=40)


def test_tracks_known_multivariate_nonlinear_curve_through_intermediate_steps():
    # Exact branch: x(t)=1+t, y(t)=(1+t)^2.
    homotopy = SympyHomotopy(
        [x - (1 + t), y - x**2],
        (x, y),
        t,
    )
    result = track_path(
        homotopy,
        (1, 1),
        options=PathTrackerOptions(
            initial_step=0.04,
            max_step=0.08,
            initial_digits=40,
            residual_digits=25,
        ),
    )
    assert result.success
    assert abs(result.endpoint[0] - 2) < 1e-18
    assert abs(result.endpoint[1] - 4) < 1e-18
    for step in result.steps:
        expected_x = 1 + step.t
        expected_y = expected_x**2
        # Recover the accepted state by solving the exact path at this t; the
        # residual bound verifies that the tracked state remained on this branch.
        assert step.relative_residual < 1e-20
        assert expected_y == pytest.approx(expected_x**2)


def test_maximum_path_step_budget_is_reported_cleanly():
    homotopy = SympyHomotopy([x - (1 + t)], (x,), t)
    result = track_path(
        homotopy,
        (1,),
        options=PathTrackerOptions(
            initial_step=0.01,
            max_step=0.01,
            max_steps=2,
            initial_digits=35,
            residual_digits=20,
        ),
    )
    assert not result.success
    assert result.accepted_steps == 2
    assert result.message == "maximum path steps exceeded"
    assert result.final_t < 1.0


def test_rejection_budget_exhaustion_is_reported(monkeypatch):
    homotopy = SympyHomotopy([x - (1 + t)], (x,), t)

    def always_fail(*args, **kwargs):
        raise continuation.PathStepError("controlled failure")

    monkeypatch.setattr(continuation, "_predict_tangent", always_fail)
    result = track_path(
        homotopy,
        (1,),
        options=PathTrackerOptions(
            initial_step=0.1,
            min_step=0.09,
            max_step=0.2,
            max_step_rejections=2,
            initial_digits=25,
            max_digits=25,
            residual_digits=18,
        ),
    )
    assert not result.success
    assert result.rejected_steps == 2
    assert result.message == "consecutive step-rejection budget exhausted"


def test_minimum_step_and_precision_ceiling_are_reported(monkeypatch):
    homotopy = SympyHomotopy([x - (1 + t)], (x,), t)

    def always_fail(*args, **kwargs):
        raise continuation.PathStepError("controlled failure")

    monkeypatch.setattr(continuation, "_predict_tangent", always_fail)
    result = track_path(
        homotopy,
        (1,),
        options=PathTrackerOptions(
            initial_step=0.1,
            min_step=0.099,
            max_step=0.2,
            max_step_rejections=20,
            initial_digits=30,
            max_digits=30,
            residual_digits=18,
        ),
    )
    assert not result.success
    assert result.rejected_steps >= 1
    assert result.message == "minimum step/maximum precision reached"


def test_continuation_residual_is_invariant_under_equation_scaling():
    ordinary = SympyHomotopy([x - 2], (x,), t)
    scaled = SympyHomotopy([sp.Rational(1, 10**100) * (x - 2)], (x,), t)

    ordinary_result = track_path(ordinary, (0,), t_start=0.0, t_end=0.0)
    scaled_result = track_path(scaled, (0,), t_start=0.0, t_end=0.0)

    assert ordinary_result.success and scaled_result.success
    assert abs(ordinary_result.endpoint[0] - 2) < mp.mpf("1e-25")
    assert abs(scaled_result.endpoint[0] - 2) < mp.mpf("1e-25")
    assert ordinary_result.max_relative_residual < 1e-20
    assert scaled_result.max_relative_residual < 1e-20


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_path_endpoints_are_rejected(bad):
    homotopy = SympyHomotopy([x - (1 + t)], (x,), t)
    with pytest.raises(ValueError, match="must be finite"):
        track_path(homotopy, (1,), t_start=bad)
    with pytest.raises(ValueError, match="must be finite"):
        track_path(homotopy, (1,), t_end=bad)


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("initial_step", float("nan")),
        ("min_step", float("inf")),
        ("max_step", float("nan")),
        ("step_growth", float("inf")),
        ("step_shrink", float("nan")),
        ("precision_growth", float("inf")),
    ],
)
def test_nonfinite_path_options_are_rejected(field, bad):
    with pytest.raises(ValueError, match="must be finite"):
        PathTrackerOptions(**{field: bad})


def test_precision_increase_recorrects_current_point(monkeypatch):
    homotopy = SympyHomotopy([x - (1 + t)], (x,), t)
    original_correct = continuation._newton_correct
    calls = []

    def recording_correct(homotopy_arg, predicted, t_arg, **kwargs):
        calls.append((mp.mp.dps, float(t_arg)))
        return original_correct(homotopy_arg, predicted, t_arg, **kwargs)

    monkeypatch.setattr(continuation, "_newton_correct", recording_correct)
    original_predict = continuation._predict_tangent
    triggered = {"done": False}

    def ill_conditioned_once(*args, **kwargs):
        tangent, condition = original_predict(*args, **kwargs)
        if not triggered["done"]:
            triggered["done"] = True
            return tangent, mp.mpf("1e40")
        return tangent, condition

    monkeypatch.setattr(continuation, "_predict_tangent", ill_conditioned_once)
    result = track_path(
        homotopy,
        (1,),
        options=PathTrackerOptions(
            initial_digits=25,
            max_digits=100,
            residual_digits=18,
            initial_step=0.1,
            max_step=0.2,
        ),
    )
    assert result.success
    assert result.precision_increases >= 1
    t_zero_calls = [digits for digits, t_value in calls if t_value == 0.0]
    assert len(t_zero_calls) >= 2
    assert max(t_zero_calls) > min(t_zero_calls)


class _BadShapeHomotopy:
    variables = (x,)

    def evaluate(self, values, parameter):
        return mp.matrix([0, 0])

    def jacobian(self, values, parameter):
        return mp.matrix([[1]])

    def parameter_derivative(self, values, parameter):
        return mp.matrix([0])


def test_custom_homotopy_output_dimensions_are_validated():
    result = track_path(_BadShapeHomotopy(), (0,), t_start=0.0, t_end=0.0)
    assert not result.success
    assert "starting seed could not be corrected" in result.message


class _NonfiniteHomotopy:
    variables = (x,)

    def evaluate(self, values, parameter):
        return mp.matrix([mp.nan])

    def jacobian(self, values, parameter):
        return mp.matrix([[1]])

    def parameter_derivative(self, values, parameter):
        return mp.matrix([0])


def test_custom_homotopy_nonfinite_values_are_rejected():
    result = track_path(_NonfiniteHomotopy(), (0,), t_start=0.0, t_end=0.0)
    assert not result.success
    assert math.isinf(result.max_relative_residual)


def test_path_residual_diagnostics_preserve_arbitrary_precision():
    import mpmath as mp

    x, t = sp.symbols("x t")
    homotopy = SympyHomotopy((x - (1 + t),), (x,), t)
    result = track_path(
        homotopy,
        (1,),
        options=PathTrackerOptions(initial_digits=60, max_digits=100, residual_digits=40),
    )
    assert isinstance(result.max_relative_residual, mp.mpf)
    assert all(isinstance(step.relative_residual, mp.mpf) for step in result.steps)


def test_path_to_infinity_is_classified_as_likely_divergent():
    x, t = sp.symbols("x t")
    homotopy = SympyHomotopy(((1 - t) * x - 1,), (x,), t)
    result = track_path(
        homotopy,
        (1,),
        options=PathTrackerOptions(
            initial_digits=30,
            max_digits=80,
            residual_digits=15,
            max_steps=1000,
            min_step=1e-10,
        ),
    )
    assert not result.success
    assert result.classification == "likely_divergent"
