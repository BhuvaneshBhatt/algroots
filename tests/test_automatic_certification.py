import pytest
import sympy as sp

from algroots import polysolve
from algroots.errors import RootCertificationError
from algroots.quotient import QuotientAlgebra

x, y = sp.symbols("x y")


@pytest.mark.parametrize(
    "equations,variables",
    [
        ((x**2 - 2,), (x,)),
        ((x**2 + 1,), (x,)),
        ((x - y**2, (y - 1) ** 3), (x, y)),
        ((x - 2, y - 3), (x, y)),
    ],
)
def test_automatic_original_endpoint_proofs(equations, variables):
    result = polysolve(equations, variables, certify="required", recognize=False)
    assert len(result.root_certifications) == len(result.roots)
    for root, attempt in zip(result.roots, result.root_certifications, strict=True):
        assert attempt.status == "certified"
        certificate = attempt.certificate
        assert certificate.equations == tuple(sp.expand(e) for e in equations)
        assert certificate.variables == variables
        assert certificate.verify()
        for coordinate, bounds in zip(root, certificate.box.bounds, strict=True):
            re, im = coordinate.as_real_imag()
            assert bounds[0] < re < bounds[1] and bounds[2] < im < bounds[3]
    if equations == (x - y**2, (y - 1) ** 3):
        assert result.root_certifications[0].certificate.multiplicity == 3


def test_batch_certification_builds_one_original_quotient(monkeypatch):
    from algroots.certification import certify_numerical_roots

    result = polysolve((x**3 - x,), (x,), recognize=False)
    original = QuotientAlgebra.from_polynomials.__func__
    calls = []

    def counted(cls, *args, **kwargs):
        calls.append(args)
        return original(cls, *args, **kwargs)

    monkeypatch.setattr(QuotientAlgebra, "from_polynomials", classmethod(counted))
    assert all(a.status == "certified" for a in certify_numerical_roots(result))
    assert len(calls) == 1


@pytest.mark.parametrize(
    "options", [dict(certification_max_dimension=1), dict(certification_max_refinements=1)]
)
def test_explicit_budget_failure(options):
    # A one-step refinement budget must be tested with fresh isolating intervals,
    # independently of exact ordering/proof work performed by earlier tests.
    sp.CRootOf.clear_cache()
    result = polysolve((x**2 - 2,), (x,), certify="auto", recognize=False, **options)
    assert all(a.status != "certified" and a.error for a in result.root_certifications)
    sp.CRootOf.clear_cache()
    with pytest.raises(RootCertificationError):
        polysolve((x**2 - 2,), (x,), certify="required", recognize=False, **options)


def test_algebraic_box_limitation_is_explicit():
    result = polysolve((x - sp.sqrt(2),), (x,), certify=True, recognize=False)
    assert result.root_certifications[0].status == "unavailable"
    assert "rational coefficients" in result.root_certifications[0].error


def test_empty_and_opt_out():
    assert polysolve((1,), (x,), certify="required").root_certifications == ()
    assert polysolve((x - 1,), (x,)).root_certifications is None


@pytest.mark.parametrize(
    "options",
    [
        dict(certify="maybe"),
        dict(certify=1),
        dict(certification_max_dimension=0),
        dict(presolve_max_terms=False),
    ],
)
def test_invalid_proof_options(options):
    with pytest.raises(ValueError):
        polysolve((x - 1,), (x,), **options)


def test_clustered_adaptive_boxes_certify_unique_roots():
    epsilon = sp.Rational(1, 10**15)
    result = polysolve((x * (x - epsilon),), (x,), method="rur", certify="auto", recognize=False)
    assert len(result.roots) == 2
    assert all(
        a.status == "certified" and a.certificate is not None for a in result.root_certifications
    )
    assert len({a.certificate.point for a in result.root_certifications}) == 2


def test_wrong_numeric_candidate_is_rejected():
    from dataclasses import replace

    from algroots.certification import certify_numerical_roots

    result = polysolve((x - 1,), (x,), recognize=False)
    attempts = certify_numerical_roots(
        replace(
            result,
            roots=((sp.Float(9),),),
            multiplicities=None,
            multiplicity_evidence=(),
            multiplicity_certifications=None,
        )
    )
    assert attempts[0].status == "failed"
    assert attempts[0].certificate is None


@pytest.mark.parametrize("scale", [sp.Integer(10) ** 100, sp.Rational(1, 10**100)])
def test_scaled_automatic_boxes(scale):
    result = polysolve((x - scale,), (x,), certify="required", recognize=False)
    certificate = result.root_certifications[0].certificate
    assert certificate.point == (scale,)
    assert certificate.verify()
