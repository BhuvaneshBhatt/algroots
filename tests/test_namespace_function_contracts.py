"""Boundary and independent-oracle tests for specialized public entry points."""

import math
from statistics import NormalDist

import pytest
import sympy as sp

from algroots.border_basis import compute_border_basis_linear
from algroots.errors import BorderBasisError
from algroots.monodromy_stopping import capture_recapture_estimate, second_order_trace_test
from algroots.rational_univariate import (
    compute_rational_univariate_representation,
    solve_rur_points,
    solve_rur_representation,
)

x, y = sp.symbols("x y")


@pytest.mark.parametrize(
    "equations,variables",
    [
        ((x**2 + 1,), (x,)),
        ((1,), (x,)),
        ((x**3,), (x,)),
        ((x**2 - 2, y - x), (x, y)),
    ],
)
@pytest.mark.parametrize("real", [True, False])
def test_rur_point_extraction_matches_exact_coordinates(equations, variables, real):
    rep = compute_rational_univariate_representation(equations, variables)
    points = solve_rur_points(rep, real=real)
    assert tuple(p.coordinates for p in points) == solve_rur_representation(rep, real=real)
    for point in points:
        assert tuple(point.assignment[v] for v in variables) == point.coordinates
        assert all(sp.simplify(eq.subs(point.assignment)) == 0 for eq in equations)
    if equations == (x**3,):
        assert len(points) == 1
    if equations == (x**2 + 1,):
        assert len(points) == (0 if real else 2)


@pytest.mark.parametrize("marked,captured,overlap", [(7, 11, 3), (5, 8, 1), (3, 3, 3)])
def test_capture_recapture_independent_rational_oracle(marked, captured, overlap):
    result = capture_recapture_estimate(marked, captured, overlap, confidence=0.9)
    mean = sp.Rational((marked + 1) * (captured + 1), overlap + 1) - 1
    variance = sp.Rational(
        (marked + 1) * (captured + 1) * (marked - overlap) * (captured - overlap),
        (overlap + 1) ** 2 * (overlap + 2),
    )
    assert result.estimate == pytest.approx(float(mean))
    assert result.standard_deviation == pytest.approx(float(sp.sqrt(variance)))
    z = NormalDist().inv_cdf(0.95)
    assert result.lower == pytest.approx(
        max(marked, captured, float(mean) - z * float(sp.sqrt(variance)))
    )
    assert result.upper == pytest.approx(float(mean) + z * float(sp.sqrt(variance)))
    assert result.lower <= result.estimate <= result.upper


@pytest.mark.parametrize(
    "counts,confidence",
    [
        ((0, 1, 0), 0.95),
        ((1, -1, 0), 0.95),
        ((2, 3, -1), 0.95),
        ((2, 3, 3), 0.95),
        ((2, 3, 1), 0),
        ((2, 3, 1), 1),
    ],
)
def test_capture_recapture_rejects_invalid_inputs(counts, confidence):
    with pytest.raises(ValueError):
        capture_recapture_estimate(*counts, confidence=confidence)


def test_trace_empty_and_singular_fibers_are_unavailable():
    assert math.isinf(second_order_trace_test((x**2,), (x,), []))
    assert math.isinf(second_order_trace_test((x**2,), (x,), [(0,)]))


@pytest.mark.parametrize(
    "equations,variables,roots,direction",
    [
        ((x,), (x, y), [(0, 0)], (1,)),
        ((x,), (x,), [(0, 0)], (1,)),
        ((x,), (x,), [(0,)], (1, 2)),
    ],
)
def test_trace_dimension_validation(equations, variables, roots, direction):
    with pytest.raises(ValueError):
        second_order_trace_test(equations, variables, roots, direction=direction)


@pytest.mark.parametrize("scale,direction", [(1, (1, 2)), (7, (3, -2))])
def test_trace_multivariate_complete_and_incomplete_fibers(scale, direction):
    equations = (scale * (x**2 - 1), y - x)
    assert (
        second_order_trace_test(equations, (x, y), [(1, 1), (-1, -1)], direction=direction) < 1e-30
    )
    assert second_order_trace_test(equations, (x, y), [(1, 1)], direction=direction) > 1e-4


def test_small_trace_does_not_prove_completeness():
    # Only two of four roots: paired accelerations cancel in the chosen direction.
    assert second_order_trace_test((x**2 - 1, y**2 - 1), (x, y), [(1, 1), (-1, -1)]) < 1e-30


def test_linear_border_unit_and_failure_contracts():
    assert compute_border_basis_linear((1,), (x,)).dimension == 0
    with pytest.raises(BorderBasisError, match="zero-dimensional"):
        compute_border_basis_linear((x * y,), (x, y))
    assert not compute_border_basis_linear((x * y,), (x, y), strict=False).diagnostics.success
    with pytest.raises(BorderBasisError):
        compute_border_basis_linear((x**2 - 1, y - x), (x, y), max_degree=1)
    assert not compute_border_basis_linear(
        (x**2 - 1, y - x), (x, y), max_degree=1, strict=False
    ).diagnostics.success
    with pytest.raises(BorderBasisError):
        compute_border_basis_linear((x - sp.sqrt(2),), (x,))


@pytest.mark.parametrize("multiplicity", [2, 3])
def test_coupled_singular_endgame_proves_only_endpoint(multiplicity):
    import mpmath as mp

    from algroots.continuation import PathTrackerOptions, SympyHomotopy
    from algroots.endgames import cauchy_endgame

    t = sp.Symbol("t")
    system = SympyHomotopy((x**multiplicity - (1 - t), y - 2 * x), (x, y), t)
    with mp.workdps(40):
        a = mp.mpf(".1") ** (mp.mpf(1) / multiplicity)
        result = cauchy_endgame(
            system,
            (a, 2 * a),
            samples=16,
            certify=True,
            deflate=True,
            options=PathTrackerOptions(initial_digits=35, max_digits=70, residual_digits=20),
        )
    assert result.status == "certified_endpoint"
    assert result.certificate.multiplicity == multiplicity
    assert result.certificate.verify()
    assert result.deflation.regular and result.deflation.verify()
    assert result.limitations
