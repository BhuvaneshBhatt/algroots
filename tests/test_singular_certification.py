from dataclasses import replace

import pytest
import sympy as sp

from algroots.certification import RationalComplexBox, certify_isolated_root, certify_root_box
from algroots.deflation import deflate_isolated_root
from algroots.errors import QuotientAlgebraError, RootCertificationError

x, y = sp.symbols("x y")


@pytest.mark.parametrize(
    "equations,variables,point,multiplicity,rank",
    [
        ((x**2,), (x,), (0,), 2, 0),
        (((x - 1) ** 2 * (x + 2) ** 3,), (x,), (1,), 2, 0),
        (((x - 1) ** 2 * (x + 2) ** 3,), (x,), (-2,), 3, 0),
        ((x**2, y**3), (x, y), (0, 0), 6, 0),
        ((x**2 - y, y**2), (x, y), (0, 0), 4, 1),
        (((x**2 - 2) ** 2,), (x,), (sp.sqrt(2),), 2, 0),
        (((x - sp.sqrt(2)) ** 3,), (x,), (sp.sqrt(2),), 3, 0),
        (((x - sp.I) ** 2,), (x,), (sp.I,), 2, 0),
        ((x**2, x), (x,), (0,), 1, 1),
    ],
)
def test_exact_certificates_and_local_multiplicities(
    equations, variables, point, multiplicity, rank
):
    certificate = certify_isolated_root(equations, variables, point)
    assert certificate.status == "certified"
    assert certificate.multiplicity == multiplicity
    assert certificate.jacobian_rank == rank
    assert certificate.is_singular == (rank < len(variables))
    assert certificate.verify()


@pytest.mark.parametrize("point", [(sp.Float(0),), (sp.pi,), (sp.Symbol("a"),), (1,), (0, 0)])
def test_invalid_exact_candidates_are_not_certified(point):
    with pytest.raises(RootCertificationError):
        certify_isolated_root((x**2,), (x,), point)


def test_positive_dimensional_and_unit_ideals_are_not_certified():
    with pytest.raises(QuotientAlgebraError):
        certify_isolated_root((x**2,), (x, y), (0, 0))
    with pytest.raises(RootCertificationError):
        certify_isolated_root((1,), (x,), (0,))


def test_dimension_guard_precedes_certificate_matrix_allocation():
    with pytest.raises(QuotientAlgebraError, match="max_dimension"):
        certify_isolated_root((x**100,), (x,), (0,), max_quotient_dimension=3)


@pytest.mark.parametrize(
    "field,value",
    [
        ("multiplicity", 1),
        ("jacobian_rank", 1),
        ("status", "numerical"),
        ("geometric_solution_count", 2),
    ],
)
def test_exact_certificate_replay_detects_tampering(field, value):
    certificate = certify_isolated_root((x**3,), (x,), (0,))
    assert not replace(certificate, **{field: value}).verify()


@pytest.mark.parametrize(
    "equations,point",
    [
        (((x**2 - 2) ** 2,), (sp.CRootOf(x**2 - 2, 1),)),
        (((x**2 + 1) ** 3,), (sp.CRootOf(x**2 + 1, 1),)),
    ],
)
def test_rootof_candidates_are_exact(equations, point):
    certificate = certify_isolated_root(equations, (x,), point)
    assert certificate.is_singular
    assert certificate.verify()


@pytest.mark.parametrize(
    "equations,variables,box,expected",
    [
        (
            (x**3,),
            (x,),
            ((-sp.Rational(1, 10), sp.Rational(1, 10), -sp.Rational(1, 10), sp.Rational(1, 10)),),
            3,
        ),
        (((x**2 - 2) ** 2,), (x,), ((1, 2, -sp.Rational(1, 10), sp.Rational(1, 10)),), 2),
        (
            ((x**2 + 1) ** 3,),
            (x,),
            ((-sp.Rational(1, 10), sp.Rational(1, 10), sp.Rational(9, 10), sp.Rational(11, 10)),),
            3,
        ),
        (
            (x - y**2, (y - 1) ** 2),
            (x, y),
            ((sp.Rational(9, 10), sp.Rational(11, 10), -sp.Rational(1, 10), sp.Rational(1, 10)),)
            * 2,
            2,
        ),
    ],
)
def test_rational_box_proves_unique_singular_root(equations, variables, box, expected):
    certificate = certify_root_box(equations, variables, box)
    assert certificate.multiplicity == expected
    assert certificate.box == RationalComplexBox(box)
    assert certificate.is_singular
    assert certificate.verify()


def test_coarse_empty_boundary_and_insufficiently_refined_boxes_fail():
    with pytest.raises(RootCertificationError, match="2 distinct roots"):
        certify_root_box((x**2 - 1,), (x,), ((-2, 2, -1, 1),))
    with pytest.raises(RootCertificationError):
        certify_root_box((x**2 - 1,), (x,), ((2, 3, -1, 1),))
    with pytest.raises(RootCertificationError):
        certify_root_box((x**2,), (x,), ((0, 1, -1, 1),))
    with pytest.raises(RootCertificationError):
        certify_root_box(
            ((x**2 - 2) ** 2,),
            (x,),
            ((sp.Rational(14142135, 10**7), sp.Rational(14142136, 10**7), -1, 1),),
            max_refinements=1,
        )


def test_box_does_not_accept_nonrational_coefficients_or_wrong_dimensions():
    with pytest.raises(RootCertificationError, match="rational coefficients"):
        certify_root_box(((x - sp.sqrt(2)) ** 2,), (x,), ((1, 2, -1, 1),))
    with pytest.raises(RootCertificationError, match="dimension"):
        certify_root_box((x**2,), (x,), ((-1, 1, -1, 1), (-1, 1, -1, 1)))


@pytest.mark.parametrize(
    "bounds", [(), ((0, 0, -1, 1),), ((0, 1, 1, -1),), ((0.0, 1, -1, 1),), ((0, 1, 2),)]
)
def test_box_validation_is_exact(bounds):
    with pytest.raises(ValueError):
        RationalComplexBox(bounds)


@pytest.mark.parametrize(
    "equations,variables,point,stage_count",
    [
        ((x**2,), (x,), (0,), 1),
        ((x**5,), (x,), (0,), 4),
        ((x**2, y**3), (x, y), (0, 0), 2),
        ((x**2 + y, y**2), (x, y), (0, 0), 3),
        (((x**2 - 2) ** 2,), (x,), (sp.sqrt(2),), 1),
        ((x - 1,), (x,), (1,), 0),
    ],
)
def test_deflation_proves_full_rank_and_preserves_original_root(
    equations, variables, point, stage_count
):
    result = deflate_isolated_root(equations, variables, point)
    assert result.regular
    assert len(result.stages) == stage_count
    assert result.final_rank == len(variables)
    assert result.stopping_reason == "regular"
    assert all(e in result.equations for e in result.certificate.equations)
    assert all(
        sp.simplify(e.subs(dict(zip(variables, point, strict=True)))) == 0 for e in result.equations
    )
    assert result.verify()


def test_local_deflation_does_not_claim_to_preserve_other_roots():
    result = deflate_isolated_root((x**2 * (x - 1),), (x,), (0,))
    assert result.regular
    assert any(e.subs(x, 1) != 0 for e in result.equations)


def test_deflation_limits_and_replay_are_conservative():
    limited = deflate_isolated_root((x**5,), (x,), (0,), max_stages=2)
    assert not limited.regular
    assert limited.stopping_reason == "stage_limit"
    assert limited.verify()
    limited = deflate_isolated_root((x**2, y**2), (x, y), (0, 0), max_added_equations=1)
    assert not limited.regular
    assert limited.stopping_reason == "equation_limit"
    assert limited.verify()


def test_deflation_replay_rejects_false_regular_rank():
    result = deflate_isolated_root((x**3,), (x,), (0,))
    assert not replace(result, final_rank=0).verify()
    false_stage = replace(result.stages[0], pivot_value=0)
    assert not replace(result, stages=(false_stage, *result.stages[1:])).verify()


def test_deflation_refinement_resolves_repeated_root():
    result = deflate_isolated_root((x**5, y**3), (x, y), (0, 0))
    refined = result.refine((sp.Rational(1, 100), -sp.Rational(1, 100)), digits=35)
    assert refined.converged
    assert max(abs(v) for v in refined.root) < sp.Rational(1, 10**20)
    assert refined.original_relative_residual < sp.Rational(1, 10**20)
    assert refined.status == "numerical"


@pytest.mark.parametrize("scale", [sp.Rational(1, 10**50), sp.Integer(10) ** 50, -7])
def test_scaled_recombined_equations_preserve_certificate_and_deflation(scale):
    equations = (scale * x**2, y**2 + x**2)
    certificate = certify_isolated_root(equations, (x, y), (0, 0))
    assert certificate.multiplicity == 4
    assert deflate_isolated_root(equations, (x, y), (0, 0)).regular


def test_deflation_uses_one_original_groebner_basis(monkeypatch):
    original = sp.groebner
    calls = []

    def record(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(sp, "groebner", record)
    assert deflate_isolated_root((x**4,), (x,), (0,)).regular
    assert len(calls) == 1
