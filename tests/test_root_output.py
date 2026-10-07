"""Multiplicity views, canonical ordering, and aligned result evidence."""

from dataclasses import replace
from itertools import islice

import pytest
import sympy as sp

from algroots import algsolve, polysolve
from algroots.errors import (
    PolynomialSystemInputError,
    RootMultiplicityError,
    RootOrderingError,
    SystemSolveLimitError,
)
from algroots.root_output import permute_root_records

x, y = sp.symbols("x y")


def test_mixed_multiplicities_have_lazy_and_materialized_views():
    result = polysolve(
        ((x - 1) ** 3 * (x + 2),),
        (x,),
        recognize=False,
        root_mode="with_multiplicity",
        root_order="required",
    )
    assert result.multiplicities == (1, 3)
    assert len(result.roots) == result.geometric_solution_count == 2
    assert len(result) == result.total_multiplicity == 4
    assert result.output_roots == result.roots_with_multiplicity == tuple(result)
    assert [sp.re(root[0]) < 0 for root in result] == [True, False, False, False]
    assert list(islice(result.iter_roots(with_multiplicity=True), 2)) == list(
        result.output_roots[:2]
    )
    assert len(result.diagnostics) == len(result.multiplicity_certifications) == 2
    assert all(attempt.certificate.verify() for attempt in result.multiplicity_certifications)


@pytest.mark.parametrize("variables", [(x, y), (y, x)])
@pytest.mark.parametrize("method", ["auto", "rur", "triangular"])
def test_nonreduced_multivariate_local_counts_and_metadata(variables, method):
    result = polysolve(
        (x * x, (y - 1) ** 3 * (y + 2)),
        variables,
        method=method,
        recognize=False,
        multiplicity="required",
        root_order="required",
    )
    assert sorted(result.multiplicities) == [2, 6]
    assert sum(result.multiplicities) == result.total_multiplicity == 8
    for root, multiplicity, attempt in zip(
        result.roots, result.multiplicities, result.multiplicity_certifications, strict=True
    ):
        assert attempt.certificate.multiplicity == multiplicity
        assert all(
            abs(sp.N(a - b, 30)) < sp.Rational(1, 10**25)
            for a, b in zip(root, attempt.certificate.point, strict=True)
        )
        assert attempt.certificate.verify()


def test_radical_multiplicities_need_no_additional_certificate(monkeypatch):
    import algroots.certification as certification

    monkeypatch.setattr(
        certification,
        "certify_numerical_roots",
        lambda *a, **k: pytest.fail("unnecessary local proof"),
    )
    result = polysolve((x * x - 2,), (x,), recognize=False)
    assert result.multiplicities == (1, 1)
    assert result.multiplicity_evidence == ("exact_radical_quotient",) * 2
    assert result.multiplicity_certifications is None
    assert result.ordering.status == "numerical"


def test_unknown_counts_are_not_inferred_from_global_total():
    result = polysolve(((x - 1) ** 3 * (x + 2),), (x,), recognize=False, multiplicity=False)
    assert result.total_multiplicity == 4
    assert result.multiplicities == (None, None)
    with pytest.raises(RootMultiplicityError):
        tuple(result.iter_roots(with_multiplicity=True))
    with pytest.raises(RootMultiplicityError):
        polysolve(
            (x * x,), (x,), recognize=False, root_mode="with_multiplicity", multiplicity=False
        )


def test_proof_budget_preserves_unknown_and_required_refuses():
    result = polysolve((x * x,), (x,), recognize=False, certification_max_dimension=1)
    assert result.multiplicities == (None,)
    assert result.multiplicity_certifications[0].status == "unavailable"
    with pytest.raises(RootMultiplicityError):
        polysolve(
            (x * x,), (x,), recognize=False, multiplicity="required", certification_max_dimension=1
        )


def test_expansion_limit_checked_before_materialization():
    with pytest.raises(SystemSolveLimitError, match="max_returned_roots"):
        polysolve(
            ((x - 1) ** 5,),
            (x,),
            recognize=False,
            root_mode="with_multiplicity",
            max_returned_roots=4,
        )
    result = polysolve(((x - 1) ** 5,), (x,), recognize=False, max_returned_roots=4)
    assert len(result.roots) == 1
    with pytest.raises(SystemSolveLimitError):
        _ = result.roots_with_multiplicity


def test_empty_repeated_view_has_trivial_certified_order():
    result = polysolve(
        (1,), (x,), root_mode="with_multiplicity", multiplicity="required", root_order="required"
    )
    assert result.roots == result.output_roots == result.multiplicities == ()
    assert len(result) == 0 and result.ordering.status == "certified"


@pytest.mark.parametrize("method", ["auto", "shape", "rur", "action", "triangular"])
@pytest.mark.parametrize("digits", [20, 40])
def test_certified_complex_order_is_backend_and_precision_independent(method, digits):
    result = polysolve(
        (x * x + 1,), (x,), digits=digits, method=method, recognize=False, root_order="required"
    )
    points = tuple(attempt.certificate.point for attempt in result.multiplicity_certifications)
    polynomial = sp.Poly(x * x + 1, x)
    assert polynomial.same_root(points[0][0], -sp.I)
    assert polynomial.same_root(points[1][0], sp.I)
    assert all(a.certificate.verify() for a in result.multiplicity_certifications)
    assert result.ordering.status == "certified"
    assert result.multiplicities == (1, 1)


def test_exact_algebraic_order_uses_original_variable_priority():
    a = polysolve(
        (x * x - 2, y + x), (x, y), recognize=False, root_order="required", variable_order="auto"
    )
    b = polysolve(
        (y + x, x * x - 2), (y, x), recognize=False, root_order="required", variable_order="input"
    )
    assert sp.re(a.roots[0][0]) < 0 < sp.re(a.roots[1][0])
    assert sp.re(b.roots[0][0]) < 0 < sp.re(b.roots[1][0])
    assert sp.re(a.roots[0][1]) > 0 and sp.re(b.roots[0][1]) > 0


def test_small_complex_cluster_is_ordered_by_exact_coordinates():
    result = polysolve(
        (x * x + sp.Rational(1, 10**26),), (x,), digits=15, recognize=False, root_order="required"
    )
    assert sp.im(result.roots[0][0]) < 0 < sp.im(result.roots[1][0])
    assert all(a.certificate.verify() for a in result.multiplicity_certifications)


def test_strict_ordering_budget_failure_is_explicit():
    with pytest.raises(RootOrderingError):
        polysolve(
            (x * x - 2,),
            (x,),
            recognize=False,
            root_order="required",
            certification_max_dimension=1,
        )


def test_all_per_point_arrays_follow_one_permutation():
    result = polysolve((x * x - 1,), (x,), recognize=False, certify="required")
    result = replace(
        result,
        root_deflations=("negative", "positive"),
        homotopy_endpoint_condition_estimates=(10, 20),
        homotopy_endpoint_smallest_singular_values=(1, 2),
    )
    reordered = permute_root_records(result, (1, 0))
    assert reordered.roots == result.roots[::-1]
    assert reordered.diagnostics == result.diagnostics[::-1]
    assert reordered.root_certifications == result.root_certifications[::-1]
    assert reordered.multiplicity_certifications == result.multiplicity_certifications[::-1]
    assert reordered.root_deflations == ("positive", "negative")
    assert reordered.homotopy_endpoint_condition_estimates == (20, 10)
    for root, attempt in zip(reordered.roots, reordered.root_certifications, strict=True):
        assert abs(root[0] - attempt.certificate.point[0]) < sp.Rational(1, 10**30)


def test_projected_branch_multiplicity_is_not_transferred():
    result = algsolve((sp.sqrt(x) - (x - 2),), (x,), recognize=False)
    assert result.multiplicities == (None,)
    assert result.multiplicity_evidence == ("projected_cover_not_transferred",)
    with pytest.raises(RootMultiplicityError):
        algsolve((sp.sqrt(x) - (x - 2),), (x,), recognize=False, root_mode="with_multiplicity")


def test_algsolve_polynomial_identity_supports_multiplicity_views():
    result = algsolve(
        ((x - 1) ** 3 * (x + 2),),
        (x,),
        recognize=False,
        root_mode="with_multiplicity",
        root_order="required",
    )
    assert result.multiplicities == (1, 3) and len(result) == 4


@pytest.mark.parametrize(
    "options",
    [
        {"root_mode": "paths"},
        {"root_order": False},
        {"multiplicity": 0},
        {"multiplicity": True},
        {"max_returned_roots": True},
        {"ordering_max_refinements": 0},
    ],
)
@pytest.mark.parametrize("solver", [polysolve, algsolve])
def test_invalid_output_options_rejected_before_solving(options, solver):
    with pytest.raises((PolynomialSystemInputError, ValueError)):
        solver((x * x - 1,), (x,), **options)


def test_existing_certificate_batch_is_reused_for_counts_and_order(monkeypatch):
    import algroots.certification as certification

    original = certification.certify_numerical_roots
    calls = []

    def record(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(certification, "certify_numerical_roots", record)
    result = polysolve(
        ((x - 1) ** 3 * (x + 2),),
        (x,),
        recognize=False,
        certify="required",
        multiplicity="required",
        root_order="required",
    )
    assert calls == [1]
    assert (
        result.root_certifications is result.multiplicity_certifications
        or result.root_certifications == result.multiplicity_certifications
    )
    assert result.multiplicities == (1, 3)


def test_separated_certificate_boxes_avoid_symbolic_sign_kernel(monkeypatch):
    import algroots.root_output as output

    monkeypatch.setattr(
        output, "_exact_sign", lambda *a: pytest.fail("unnecessary symbolic comparison")
    )
    result = polysolve((x * x - 2,), (x,), recognize=False, root_order="required")
    assert result.ordering.status == "certified"


@pytest.mark.recognition
def test_joint_recognition_reused_for_exact_order(monkeypatch):
    import algroots.certification as certification

    monkeypatch.setattr(
        certification,
        "certify_numerical_roots",
        lambda *a, **k: pytest.fail("unnecessary endpoint proof"),
    )
    result = polysolve((x * x - 2,), (x,), recognize=True, root_order="required")
    assert result.ordering.status == "certified"
    assert all(r.jointly_certified for r in result.recognized_roots)
    assert result.recognized_roots[0].exact_coordinates == (-sp.sqrt(2),)


def test_rur_coordinate_proofs_share_one_primitive_field():
    from algroots.certification import _point_field, certify_isolated_root
    from algroots.quotient import QuotientAlgebra

    root = sp.CRootOf(x**3 - 2, 1)
    quotient = QuotientAlgebra.from_polynomials((x**3 - 2, y - x * x), (x, y))
    point, field = _point_field(quotient, (root, root * root))
    assert field.ext.as_expr() == root
    assert all(field.to_sympy(field.from_sympy(c)) == c for c in point)
    certificate = certify_isolated_root((x**3 - 2, y - x * x), (x, y), point)
    assert certificate.multiplicity == 1 and certificate.verify()
