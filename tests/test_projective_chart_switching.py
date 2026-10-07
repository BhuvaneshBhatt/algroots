import mpmath as mp
import pytest
import sympy as sp

from algroots.continuation import PathTrackerOptions, SympyHomotopy
from algroots.endgames import cauchy_endgame, projective_homotopy
from algroots.errors import RootCertificationError
from algroots.projective_tracking import track_projective_path

x, t = sp.symbols("x t")


def test_automatic_chart_switch_resolves_affine_infinity():
    projective = projective_homotopy(((1 - t) * x - 1,), (x,), t, patch=(0, 1))
    result = track_projective_path(projective, projective.lift((1,)), segment_size=0.1)
    assert result.success
    assert result.classification == "numerically_near_infinity"
    assert len(result.chart_switches) >= 1
    assert abs(result.endpoint[0] - 1) < mp.mpf("1e-20")
    assert abs(result.endpoint[-1]) < mp.mpf("1e-20")
    assert result.status == "numerical"
    with pytest.raises(ValueError):
        result.affine_endpoint()


@pytest.mark.parametrize("segment_size", [0.03, 0.1, 0.2])
def test_multiple_chart_crossings_preserve_projective_branch(segment_size):
    expression = (t - sp.Rational(3, 10)) * (t - sp.Rational(7, 10)) * x - (t - sp.Rational(1, 2))
    projective = projective_homotopy((expression,), (x,), t, patch=(0, 1))
    result = track_projective_path(
        projective, projective.lift((-sp.Rational(50, 21),)), segment_size=segment_size
    )
    assert result.success
    assert len(result.chart_switches) >= 3
    assert result.classification == "finite"
    assert abs(result.affine_endpoint()[0] - mp.mpf(50) / 21) < mp.mpf("1e-14")
    for switch in result.chart_switches:
        assert abs(
            sum(c * v for c, v in zip(switch.next_patch, switch.point, strict=True)) - 1
        ) < mp.mpf("1e-20")


def test_backward_chart_tracking_and_homogeneous_scale_invariance():
    projective = projective_homotopy((x - t,), (x,), t, patch=(1, 0))
    start = projective.lift((1,))
    result = track_projective_path(projective, tuple(13 * v for v in start), t_start=1, t_end=0)
    assert result.success
    assert result.classification == "finite"
    assert abs(result.affine_endpoint()[0]) < mp.mpf("1e-20")


def test_segment_and_chart_budgets_do_not_claim_success():
    projective = projective_homotopy(((1 - t) * x - 1,), (x,), t, patch=(0, 1))
    result = track_projective_path(projective, projective.lift((1,)), max_segments=1)
    assert not result.success
    assert result.classification == "unresolved"
    expression = (t - sp.Rational(3, 10)) * (t - sp.Rational(7, 10)) * x - (t - sp.Rational(1, 2))
    projective = projective_homotopy((expression,), (x,), t, patch=(0, 1))
    result = track_projective_path(
        projective,
        projective.lift((-sp.Rational(50, 21),)),
        max_chart_switches=1,
        options=PathTrackerOptions(
            initial_digits=25, max_digits=40, residual_digits=16, max_steps=200
        ),
    )
    assert not result.success
    assert "budget" in result.message


@pytest.mark.parametrize("start", [(0, 0), (mp.inf, 1), (1,)])
def test_invalid_homogeneous_seeds_are_rejected(start):
    projective = projective_homotopy((x - t,), (x,), t, patch=(0, 1))
    with pytest.raises(ValueError):
        track_projective_path(projective, start)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"segment_size": 0},
        {"switch_ratio": 1},
        {"max_segments": True},
        {"max_chart_switches": 0},
        {"t_end": float("nan")},
    ],
)
def test_projective_controls_are_validated(kwargs):
    projective = projective_homotopy((x - t,), (x,), t, patch=(0, 1))
    with pytest.raises(ValueError):
        track_projective_path(projective, (0, 1), **kwargs)


@pytest.mark.parametrize("multiplicity", [2, 3])
def test_cauchy_endgame_certifies_and_deflates_singular_endpoint(multiplicity):
    system = SympyHomotopy((x**multiplicity - (1 - t),), (x,), t)
    with mp.workdps(40):
        start = (mp.mpf(".1") ** (mp.mpf(1) / multiplicity),)
        result = cauchy_endgame(
            system,
            start,
            samples=16,
            certify=True,
            deflate=True,
            options=PathTrackerOptions(initial_digits=35, max_digits=70, residual_digits=20),
        )
    assert result.status == "certified_endpoint"
    assert result.certificate.multiplicity == multiplicity
    assert result.certificate.is_singular
    assert result.certificate.verify()
    assert result.deflation.regular
    assert result.deflation.verify()
    assert result.limitations  # No numerical path completeness claim.


def test_unrelated_box_and_invalid_certification_controls_are_rejected():
    system = SympyHomotopy((x - (1 - t),), (x,), t)
    with pytest.raises(ValueError, match="requires certify"):
        cauchy_endgame(system, (mp.mpf(".1"),), deflate=True)
    with pytest.raises(ValueError, match="exact algebraic target"):
        cauchy_endgame(system, (mp.mpf(".1"),), certify=True, target=1.0)
    with pytest.raises(RootCertificationError, match="not inside"):
        cauchy_endgame(
            system, (mp.mpf(".1"),), samples=8, certify=True, certification_box=((1, 2, -1, 1),)
        )


def test_decimal_endpoint_at_high_ambient_precision_does_not_stall():
    from algroots.continuation import track_path

    system = SympyHomotopy((x - t,), (x,), t)
    with mp.workdps(240):
        result = track_path(
            system,
            (mp.mpf(".1"),),
            t_start=0.1,
            t_end=0.15,
            options=PathTrackerOptions(
                initial_digits=25, max_digits=40, residual_digits=16, max_steps=10
            ),
        )
        assert result.success
        assert result.final_t == 0.15
        assert result.accepted_steps <= 2
        assert abs(result.endpoint[0] - mp.mpf(".15")) < mp.mpf("1e-20")


def test_zero_length_projective_request_validates_and_corrects_seed():
    projective = projective_homotopy((x - t,), (x,), t, patch=(0, 1))
    result = track_projective_path(projective, (sp.Rational(1, 100), 1), t_start=0, t_end=0)
    assert result.success
    assert abs(result.affine_endpoint()[0]) < mp.mpf("1e-20")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"certification_box": ((-1, 1, -1, 1),)},
        {"certify": True, "max_quotient_dimension": 0},
        {"samples": 8.5},
        {"max_cycle": True},
    ],
)
def test_endgame_proof_options_are_validated_before_tracking(kwargs):
    system = SympyHomotopy((x - (1 - t),), (x,), t)
    with pytest.raises(ValueError):
        cauchy_endgame(system, (mp.mpf(".1"),), **kwargs)


def test_homotopy_retries_smaller_steps_after_duplicate_regular_endpoints():
    from algroots import polysolve

    y = sp.Symbol("y")
    equations = ((x + 2 * y + 1) ** 2 - 1, (-2 * x + y + 1) ** 2 - 4)
    result = polysolve(
        equations,
        (x, y),
        method="homotopy",
        digits=28,
        recognize=False,
        homotopy_gamma_attempts=2,
        max_homotopy_paths=8,
    )
    assert len(result.roots) == 4
    inverse = sp.Matrix([[1, 2], [-2, 1]]).inv()
    expected = {tuple(inverse * sp.Matrix([u - 1, v - 1])) for u in (-1, 1) for v in (-2, 2)}
    actual = {tuple(round(float(sp.re(v)), 10) for v in root) for root in result.roots}
    assert actual == {tuple(float(v) for v in root) for root in expected}


def test_endgame_result_preserves_original_positional_status_fields():
    from algroots.endgames import EndgameResult

    result = EndgameResult(
        (0,), True, 1, (sp.Rational(1, 10),), ((0,),), 0, 0, "numerical", ("original limitation",)
    )
    assert result.status == "numerical"
    assert result.limitations == ("original limitation",)
    assert result.certificate is None and result.deflation is None
