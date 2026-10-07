from dataclasses import replace

import pytest
import sympy as sp

from algroots import polysolve
from algroots.certification import certify_numerical_roots
from algroots.quotient import QuotientAlgebra

x, y = sp.symbols("x y")


@pytest.mark.parametrize("power", [15, 25, 35])
def test_adaptive_cluster_boxes(power):
    delta = sp.Rational(1, 10**power)
    result = polysolve(
        (x * (x - delta),),
        (x,),
        method="rur",
        recognize=False,
        digits=70,
        certify="required",
        certification_max_refinements=256,
    )
    assert len(result.root_certifications) == 2
    for attempt in result.root_certifications:
        assert attempt.certificate.verify(max_refinements=256)
        assert 1 <= attempt.box_attempts <= 16


def test_adaptive_separator_cancellation():
    # Nearby projected separator values despite coordinates of order one.
    delta = sp.Rational(1, 10**14)
    result = polysolve((x * (x - 1), y + x - delta * x), (x, y), recognize=False, method="rur")
    attempts = certify_numerical_roots(result, max_box_attempts=16)
    assert all(a.status == "certified" for a in attempts)
    assert any(a.box_attempts > 1 for a in attempts)


def test_exact_wrong_candidates_never_certified():
    result = polysolve((x - 1,), (x,), recognize=False)
    attempts = certify_numerical_roots(replace(result, roots=((sp.Float(8),),)), max_box_attempts=3)
    assert attempts[0].status == "failed" and attempts[0].box_attempts == 3


def test_duplicate_singular_path_proposals():
    result = polysolve((x**2,), (x,), recognize=False)
    candidates = ((sp.Float("1e-50"),), (sp.Float("1.00000001e-50"),))
    attempts = certify_numerical_roots(
        replace(
            result,
            roots=candidates,
            multiplicities=None,
            multiplicity_evidence=(),
            multiplicity_certifications=None,
        )
    )
    assert all(a.status == "certified" and a.certificate.multiplicity == 2 for a in attempts)


@pytest.mark.parametrize("equations", [(x**3 - 2,), (x**3,), (x**2 - sp.sqrt(2),)])
def test_cache_exactness_and_hits(equations):
    cached = QuotientAlgebra.from_polynomials(equations, (x,), normal_form_cache_size=2)
    uncached = QuotientAlgebra.from_polynomials(equations, (x,), normal_form_cache_size=0)
    for power in range(3, 8):
        expression = x**power + 1
        assert cached.normal_form(expression) == uncached.normal_form(expression)
        assert cached.normal_form(expression) == uncached.normal_form(expression)
    info = cached.normal_form_cache_info
    assert info["hits"] == 5 and info["evictions"] == 3 and info["entries"] == 2
    assert info["reduction_seconds"] >= 0
    assert uncached.normal_form_cache_info["entries"] == 0
    cached.clear_normal_form_cache()
    assert cached.normal_form_cache_info["hits"] == cached.normal_form_cache_info["entries"] == 0


def test_cache_not_shared_across_ideals():
    left = QuotientAlgebra.from_polynomials((x**2 - 1,), (x,))
    right = QuotientAlgebra.from_polynomials((x**2 - 2,), (x,))
    assert left.normal_form(x**2) == 1 and right.normal_form(x**2) == 2


def test_large_expressions_not_admitted():
    quotient = QuotientAlgebra.from_polynomials((x**2 - 1, y**2 - 1), (x, y))
    expression = sum(x**i * y ** (i + 1) for i in range(100))
    assert quotient.normal_form(expression) == 50 * (x + y)
    assert quotient.normal_form_cache_info["entries"] == 0


@pytest.mark.parametrize("capacity", [False, -1, 1.5])
def test_bad_cache_controls(capacity):
    with pytest.raises(ValueError):
        QuotientAlgebra.from_polynomials((x**2 - 1,), (x,), normal_form_cache_size=capacity)


def test_cost_diagnostics_are_observations():
    result = polysolve((x**2 - 2, y**2 - 3), (x, y), presolve=False, recognize=False)
    phases = dict(result.cost_diagnostics.phase_seconds)
    assert all(value >= 0 for value in phases.values())
    assert phases["total"] >= phases["solve"] >= phases["solve.groebner"]
    assert dict(result.cost_diagnostics.structural_estimates)["selected_backend"] == result.method
    assert dict(result.cost_diagnostics.normal_form_statistics)["misses"] > 0
    assert result.completeness.status == "conditional"


def test_large_coefficient_not_retained():
    quotient = QuotientAlgebra.from_polynomials((x**2 - 1,), (x,))
    coefficient = sp.Integer(2) ** 10000
    assert quotient.normal_form(coefficient * x**2) == coefficient
    assert quotient.normal_form_cache_info["entries"] == 0
