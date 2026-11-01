"""End-to-end identities, exhaustive rational search and candidate-only claims."""

from dataclasses import replace

import pytest
import sympy as sp

from algroots.functional_decomposition import (
    AlgebraicDecomposition,
    DecompositionLimitError,
    FunctionalDecompositionError,
    RationalDecomposition,
    algebraic_decompose,
    rational_decompose,
)
from algroots.quotient import QuotientAlgebra

x, y, z, a = sp.symbols("x y z a")


@pytest.mark.parametrize(
    "function,degrees",
    [(x**4, {(2, 2)}), (x**6, {(2, 3), (3, 2)}), ((x + 1 / x) ** 2, {(2, 2)})],
)
def test_exhaustive_rational_two_factor_search(function, degrees):
    result = rational_decompose(function, x, parameter=y)
    assert result.status == "decomposable" and result.complete and result.verify()
    assert {d.degrees for d in result.decompositions} == degrees
    for d in result.decompositions:
        assert d.verify() and sp.cancel(d.outer.subs(y, d.inner) - function) == 0
    # All three degree-two intermediate fields of (x+1/x)^2 are needed.
    if function == (x + 1 / x) ** 2:
        assert len(result.decompositions) == 3
    assert not replace(result, decompositions=()).verify()
    assert not replace(result, candidate_count=result.candidate_count + 1).verify()
    assert not replace(result, examined=0).verify()
    assert not replace(result, factors=result.factors[:-1]).verify()


def test_factor_products_not_only_individual_factors_are_searched():
    result = rational_decompose(x**6, x, parameter=y)
    assert any(d.degrees == (3, 2) for d in result.decompositions)
    assert any(d.degrees == (2, 3) for d in result.decompositions)
    assert result.candidate_count == 3


def test_mobius_changed_inner_and_outer_recover_the_same_subfields():
    h = (x * x + 2) / (x * x - 1)
    f = sp.cancel((y**3 - y + 2).subs(y, h))
    result = rational_decompose(f, x, parameter=y)
    assert result.complete and result.verify()
    assert any(d.degrees == (3, 2) for d in result.decompositions)
    assert len([d for d in result.decompositions if d.degrees == (3, 2)]) == 1


@pytest.mark.parametrize("function", [x**4 + x, (x**4 + x) / (x + 1), x**5 + x + 1])
def test_exhaustive_negative_decisions(function):
    result = rational_decompose(function, x)
    assert result.complete and result.status == "indecomposable" and result.verify()


@pytest.mark.parametrize("function", [0, 2, x, (x + 1) / (x - 1)])
def test_constants_and_degree_one_are_explicit(function):
    result = rational_decompose(function, x)
    assert result.complete and result.status == "degree_below_two" and result.verify()


def test_candidate_budget_never_becomes_an_indecomposability_claim():
    empty = rational_decompose(x**6, x, parameter=y, max_candidates=0)
    assert empty.status == "incomplete" and not empty.complete and empty.verify()
    partial = rational_decompose(x**6, x, parameter=y, max_candidates=1)
    assert partial.status == "decomposable" and not partial.complete and partial.verify()
    assert partial.examined == 1 and partial.candidate_count == 3
    assert not replace(partial, examined=3).verify()
    with pytest.raises(DecompositionLimitError):
        rational_decompose(x**12, x, max_degree=8)


def test_exact_number_field_constants():
    K = sp.QQ.algebraic_field(sp.sqrt(2))
    f = (x * x + sp.sqrt(2)) ** 2 + sp.sqrt(2)
    result = rational_decompose(f, x, parameter=y, constants=K)
    assert result.complete and result.verify() and result.decompositions
    assert all(d.constants == K for d in result.decompositions)


def test_original_visible_poles_survive_cancellation():
    original = sp.Mul(x**4 * (x - 1), sp.Pow(x - 1, -1, evaluate=False), evaluate=False)
    result = rational_decompose(original, x, parameter=y)
    d = result.decompositions[0]
    assert d.function == x**4 and d.input_guard == sp.Ne(x - 1, 0) and d.verify()
    assert not replace(d, inner=x**3).verify()
    assert not replace(d, outer=y**2 + 1).verify()
    assert not replace(d, parameter=x).verify()


def test_symbol_identity_and_assumptions_are_preserved():
    v = sp.Dummy("x", positive=True)
    t = sp.Symbol("x", real=True)
    result = rational_decompose(v**4, v, parameter=t)
    assert result.variable is v and result.parameter is t and result.verify()


@pytest.mark.parametrize("function", [x + z, sp.Float("1.5") * x, sp.sin(x), sp.oo, sp.zoo])
def test_invalid_rational_inputs(function):
    with pytest.raises((ValueError, TypeError)):
        rational_decompose(function, x)


@pytest.mark.parametrize(
    "options",
    [
        {"parameter": x},
        {"max_candidates": -1},
        {"max_candidates": True},
        {"max_degree": 0},
        {"constants": sp.RR},
    ],
)
def test_invalid_rational_options(options):
    with pytest.raises(ValueError):
        rational_decompose(x**4, x, **options)


def test_paper_example_without_adjoining_the_input_to_coefficient_field():
    F = x**4 - z * x * x * (x + 1) + z**3 * (x + 1) ** 2
    A = QuotientAlgebra.from_polynomials([a * a - z * a + z**3], (a,), domain=sp.QQ.frac_field(z))
    result = algebraic_decompose(F, (x, z), candidate=x * x - a * (x + 1), ambient=A, parameter=y)
    assert result.status == "decomposable" and not result.complete and result.verify()
    assert not result.coefficient_field.contains(z)
    D = result.decomposition
    assert D.verify() and D.exponent == 1
    assert D.outer == sp.Poly(x * x + y * (x + 1), x, y, domain=sp.QQ)
    assert D.inner == sp.Poly(y * y + z * y + z**3, y, z, domain=sp.QQ)
    assert not replace(D, inner=sp.Poly(y * y + z * y + z**3 + 1, y, z)).verify()
    assert not replace(D, unit=z).verify()
    assert not replace(result, candidate=x * x + a * (x + 1)).verify()
    assert not replace(result, reason="positive_genus").verify()
    assert not replace(result, decomposition=replace(D, exponent=2)).verify()


def test_overlapping_conjugate_factors_give_a_resultant_power():
    A = QuotientAlgebra.from_polynomials([a**3 - 3 * a - z], (a,), domain=sp.QQ.frac_field(z))
    result = algebraic_decompose(
        x**3 - 3 * x - z, (x, z), candidate=x * x + a * x + a * a - 3, ambient=A, parameter=y
    )
    assert result.verify() and result.decomposition.exponent == 2
    assert not replace(result.decomposition, exponent=1).verify()
    assert not replace(result.decomposition, unit=2).verify()


def test_full_factor_can_expose_a_nontrivial_rational_inner_component():
    A = QuotientAlgebra.from_polynomials([a - z], (a,), domain=sp.QQ.frac_field(z))
    F = x * x - z**4 - 1
    result = algebraic_decompose(F, (x, z), candidate=F, ambient=A, parameter=y)
    assert result.verify() and result.decomposition.inner.degree(z) == 4


@pytest.mark.parametrize(
    "modulus,reason",
    [(a * a - z**3 + z, "positive_genus"), (a * a + z * z - 3, "no_ground_field_point")],
)
def test_candidate_rejection_is_not_global_indecomposability(modulus, reason):
    A = QuotientAlgebra.from_polynomials([modulus], (a,), domain=sp.QQ.frac_field(z))
    g = x * x + a * x + z
    F = sp.resultant(g, modulus, a)
    result = algebraic_decompose(F, (x, z), candidate=g, ambient=A, parameter=y)
    assert result.status == "candidate_rejected" and result.reason == reason
    assert not result.complete and not result.decompositions and result.verify()
    assert not replace(result, reason="other").verify()


def test_reducible_and_trivial_reconstructed_components_are_rejected():
    A = QuotientAlgebra.from_polynomials([a**3 - z], (a,), domain=sp.QQ.frac_field(z))
    # F itself becomes reducible over K(a), so this is not a minimal component.
    result = algebraic_decompose(x**3 - z, (x, z), candidate=x**3 - a**3, ambient=A, parameter=y)
    # Coefficients reduce to z: here G=x^3+y is irreducible, but H is a unit map.
    assert result.reason == "trivial_component" and result.verify()
    linear = algebraic_decompose(x**3 - z, (x, z), candidate=x - a, ambient=A, parameter=y)
    assert linear.reason == "trivial_component" and linear.verify()


@pytest.mark.parametrize(
    "options", [{"candidate": x * x + a}, {"candidate": 1}, {"parameter": x}, {"ambient": None}]
)
def test_invalid_algebraic_candidates(options):
    A = QuotientAlgebra.from_polynomials([a * a - z], (a,), domain=sp.QQ.frac_field(z))
    kw = {"candidate": x - a, "ambient": A, "parameter": y}
    kw.update(options)
    with pytest.raises(FunctionalDecompositionError):
        algebraic_decompose(x * x - z, (x, z), **kw)


def test_resultant_extra_components():
    D = AlgebraicDecomposition(
        sp.Poly(x * x - z, x, z), sp.Poly(x * x - y, x, y), sp.Poly(y - z, y, z), 1, sp.Integer(1)
    )
    assert not D.verify()  # H is an invertible intermediate coordinate.
    bad = replace(D, inner=sp.Poly(y * y - z, y, z))
    assert not bad.verify()  # Resultant is x^4-z, not the asserted input.
    assert not RationalDecomposition(x**4, x, y, sp.QQ, y, x**4).verify()
    assert not replace(bad, unit=sp.Float(1)).verify()


def test_reducible_reconstructed_factor_is_rejected():
    A = QuotientAlgebra.from_polynomials([a**4 - z], (a,), domain=sp.QQ.frac_field(z))
    result = algebraic_decompose(
        x**4 - z,
        (x, z),
        candidate=x**3 + a * x * x + a * a * x + a**3,
        ambient=A,
        parameter=y,
    )
    assert result.reason == "reducible_candidate" and result.verify()


def test_candidate_is_normalized_in_the_ambient_field_before_degree_checks():
    modulus = a * a - z * a + z**3
    A = QuotientAlgebra.from_polynomials([modulus], (a,), domain=sp.QQ.frac_field(z))
    F = x**4 - z * x * x * (x + 1) + z**3 * (x + 1) ** 2
    g = x * x - a * (x + 1)
    result = algebraic_decompose(
        F,
        (x, z),
        candidate=modulus * x**3 + 2 * a * g,
        ambient=A,
        parameter=y,
    )
    assert result.candidate == g.expand() and result.verify()


def test_algebraic_frontend_preserves_number_field_constants():
    K = sp.QQ.algebraic_field(sp.sqrt(2))
    A = QuotientAlgebra.from_polynomials([a - z], (a,), domain=K.frac_field(z))
    F = x * x - z**4 - sp.sqrt(2)
    result = algebraic_decompose(F, (x, z), candidate=F, ambient=A, parameter=y, constants=K)
    assert result.verify() and result.decomposition.constants == K


def test_complete_verifier_does_not_replay_candidate_recovery(monkeypatch):
    import algroots.functional_decomposition as search

    examples = (x**6, x**4 + x, (x + 1 / x) ** 2)
    answers = tuple(rational_decompose(f, x, parameter=y) for f in examples)
    assert all(answer.complete and answer.verify() for answer in answers)

    def unexpected_recovery(*_args, **_kwargs):
        raise AssertionError("the independent verifier called candidate recovery")

    monkeypatch.setattr(search, "_recover_rational", unexpected_recovery)
    assert all(answer.verify() for answer in answers)
    # A producer bug dropping an admissible candidate cannot certify absence.
    assert not replace(answers[0], decompositions=()).verify()
    assert not replace(answers[2], decompositions=answers[2].decompositions[:1]).verify()


def test_complete_verifier_rejects_incorrect_factor_coverage():
    answer = rational_decompose(x**6, x, parameter=y)
    assert answer.complete and answer.verify()
    assert not replace(answer, candidate_count=answer.candidate_count + 1).verify()
    assert not replace(answer, factors=answer.factors[:-1]).verify()
    duplicate = (*answer.decompositions, answer.decompositions[0])
    assert not replace(answer, decompositions=duplicate).verify()
