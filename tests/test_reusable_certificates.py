"""Replay reusable ideal and RUR certificates, including hostile mutations."""

import pickle
from dataclasses import replace

import pytest
import sympy as sp

from algroots import (
    AlgebraicFunctionField,
    GroebnerVerificationError,
    QuotientAlgebra,
    certify_groebner_basis,
    certify_rational_univariate_representation,
    groebner_basis,
)

x, y, t, u, a, z = sp.symbols("x y t u a z")


@pytest.mark.parametrize("order", ["lex", "grlex", "grevlex"])
@pytest.mark.parametrize(
    "domain", [sp.QQ, sp.GF(5), sp.QQ.algebraic_field(sp.sqrt(2)), sp.QQ.frac_field(t)]
)
def test_basis_replay(domain, order):
    source = (x * y - 1, y * y - 1)
    proof = certify_groebner_basis(source, (x, y), domain=domain, order=order)
    restored = pickle.loads(pickle.dumps(proof))
    assert restored.verify()
    assert (
        groebner_basis(
            source, (x, y), domain=domain, order=order, certificate=restored, backend="sympy"
        )
        == proof.basis
    )
    assert QuotientAlgebra.from_certificate(restored).dimension == 2
    with pytest.raises(GroebnerVerificationError, match="does not match"):
        groebner_basis(
            [x, y], (x, y), domain=domain, order=order, certificate=proof, backend="sympy"
        )


def test_candidate_without_reference(monkeypatch):
    candidate = groebner_basis([x * y - 1, y * y - 1], (x, y), backend="sympy")

    def forbidden(*args, **kwargs):
        raise RuntimeError("reference recomputation")

    monkeypatch.setattr("sympy.polys.groebnertools.groebner", forbidden)
    proof = certify_groebner_basis([x * y - 1, y * y - 1], (x, y), candidate=candidate)
    assert proof.verify()
    assert groebner_basis(proof.source, (x, y), certificate=proof, backend="sympy") == candidate


def test_basis_mutations():
    proof = certify_groebner_basis([x * y - 1, y * y - 1], (x, y))
    for bad in (
        replace(proof, memberships=()),
        replace(proof, generator_reductions=()),
        replace(proof, pair_reductions=()),
        replace(proof, source=(x, y)),
    ):
        assert not bad.verify()
        with pytest.raises(GroebnerVerificationError):
            bad.verified_basis()
    with pytest.raises((ValueError, GroebnerVerificationError)):
        certify_groebner_basis(
            [x * y], (x, y), candidate=groebner_basis([1], (x, y), backend="sympy")
        )


@pytest.mark.parametrize("source", [(), (sp.S.Zero,)])
def test_zero_ideal(source):
    proof = certify_groebner_basis(source, (x,))
    assert proof.verify() and list(proof.basis) == []


@pytest.mark.parametrize(
    "source,counts",
    [
        ([x * x - t, y - x], (1, 2, 2)),
        ([(x - t) ** 2 * (x + t) ** 3, y - x], (1, 2, 2)),
        ([t * x - 1, y], (0, 1, 1)),
        ([t, x, y], (1, 0, 0)),
    ],
)
def test_rur_fibers(source, counts):
    proof = certify_rational_univariate_representation(source, (x, y), parameters=(t,))
    assert proof.verify()
    for value, count in zip((0, 1, 2), counts, strict=True):
        result = proof.specialize({t: value})
        assert result.solution_count == count
        coords = dict(
            zip(
                (x, y),
                (p.as_expr() for p in result.normalized_coordinate_polynomials()),
                strict=True,
            )
        )
        for p in source:
            assert (
                sp.rem(
                    sp.sympify(p).subs(t, value).subs(coords, simultaneous=True),
                    result.defining_polynomial.as_expr(),
                    result.parameter,
                )
                == 0
            )


def test_rur_guard_and_mutations():
    proof = certify_rational_univariate_representation([x * x - t], (x,), parameters=(t,))
    assert proof.guard == t
    r = proof.representation
    bad_rurs = [
        replace(
            r, coordinate_numerators=(sp.Poly(0, r.parameter, domain=r.defining_polynomial.domain),)
        ),
        replace(r, quotient_dimension=3),
        replace(r, geometric_solution_count=1),
        replace(r, separating_linear_form=2 * x),
    ]
    for bad in bad_rurs:
        assert not replace(proof, representation=bad).verify()
    assert not replace(proof, guard=sp.S.One).verify()
    assert not replace(proof, input_guard=t).verify()
    with pytest.raises(GroebnerVerificationError):
        replace(proof, guard=sp.S.One).specialize({t: 1})


def test_rur_poles_and_partial_values():
    proof = certify_rational_univariate_representation([x * x - 1 / t], (x,), parameters=(t,))
    for values in ({t: 0}, {}, {t: 1.0}):
        with pytest.raises(ValueError):
            proof.specialize(values)


def test_rur_rejects_missing_points():
    proof = certify_rational_univariate_representation([x * x - 1], (x,))
    r = proof.representation
    bad = replace(
        r,
        defining_polynomial=sp.Poly(r.parameter - 1, r.parameter, domain=sp.QQ),
        coordinate_denominator=sp.Poly(1, r.parameter, domain=sp.QQ),
        coordinate_numerators=(sp.Poly(1, r.parameter, domain=sp.QQ),),
        geometric_solution_count=1,
    )
    assert not replace(proof, representation=bad).verify()


def test_tower_specialization():
    K = AlgebraicFunctionField(sp.Poly(a * a - t, a, domain=sp.QQ.frac_field(t)))
    proof = certify_groebner_basis([x * x - a, y - x], (x, y), domain=K)
    embedding = K.specialize({t: 4}, roots={a: 2})
    specialized = proof.specialization_certificate(embedding)
    assert specialized.verify()
    assert specialized.basis == groebner_basis([x * x - 2, y - x], (x, y), backend="sympy")
    rur = certify_rational_univariate_representation([x * x - a], (x,), domain=K)
    assert rur.specialize_field(embedding).solution_count == 2
    assert rur.specialize_field(K.specialize({t: 0}, roots={a: 0})).solution_count == 1


def test_certificate_verification_route(monkeypatch):
    from algroots.native_f4 import compiled_available

    if not compiled_available():
        pytest.skip("native F4 is not installed")

    def forbidden(*args, **kwargs):
        raise RuntimeError("reference recomputation")

    monkeypatch.setattr("sympy.polys.groebnertools.groebner", forbidden)
    result = groebner_basis(
        [x * x - 2, y - x], (x, y), backend="native", verification="certificate"
    )
    assert set(result) == {y * y - 2, x - y}


def test_certificate_budget():
    from algroots import GroebnerLimitError, NativeF4Limits

    with pytest.raises(GroebnerLimitError, match="basis-count"):
        certify_groebner_basis([x, y], (x, y), proof_limits=NativeF4Limits(max_basis=1))
    with pytest.raises(ValueError, match="verification"):
        groebner_basis([x], (x,), verification="unchecked", backend="sympy")


def test_parametric_persistence():
    from algroots import parametric_groebner_basis

    for proof in (
        parametric_groebner_basis([t * x], (x,), (t,)),
        certify_rational_univariate_representation([x * x - t], (x,), parameters=(t,)),
    ):
        restored = pickle.loads(pickle.dumps(proof))
        assert restored.verify()
        assert restored.guard == t
        specialized = restored.specialize({t: 0})
        assert (
            list(specialized) == []
            if hasattr(specialized, "polys")
            else specialized.solution_count == 1
        )


def test_rur_ground_field_option():
    proof = certify_rational_univariate_representation(
        [x * x - t], (x,), parameters=(t,), domain=sp.QQ
    )
    assert proof.verify() and proof.guard == t


def test_rur_cache_is_not_evidence():
    proof = certify_rational_univariate_representation([x * x - 1], (x,))
    r = proof.representation
    original = r.normalized_coordinate_polynomials()
    bad = replace(r, coordinate_numerators=(sp.Poly(0, r.parameter, domain=sp.QQ),))
    object.__setattr__(bad, "_normalized_coordinate_polynomials", original)
    assert not replace(proof, representation=bad).verify()


def test_rur_rejects_corrupt_cache():
    proof = certify_rational_univariate_representation([x * x - 1], (x,))
    r = replace(proof.representation)
    object.__setattr__(
        r, "_normalized_coordinate_polynomials", (sp.Poly(0, r.parameter, domain=sp.QQ),)
    )
    assert not replace(proof, representation=r).verify()
