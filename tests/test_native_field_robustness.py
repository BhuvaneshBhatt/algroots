"""Unlucky characteristics, embedding invariants, multiplicities, and guarded fibers."""

from dataclasses import replace
from itertools import product

import pytest
import sympy as sp
from sympy.polys.rings import PolyRing

from algroots import NativeF4Limits, QuotientAlgebra, groebner_basis, parametric_groebner_basis
from algroots.native_f4 import compiled_available
from algroots.native_fields import native_field_basis

x, y, t, u = sp.symbols("x y t u")


@pytest.mark.skipif(not compiled_available(), reason="native F4 is not installed")
@pytest.mark.parametrize(
    "alpha,prime,source,reason",
    [
        (sp.sqrt(2), 2, x - sp.sqrt(2), "squarefree"),
        (sp.sqrt(2) / 2, 2, x - sp.sqrt(2) / 2, "squarefree"),
        (sp.sqrt(2), 3, x - sp.sqrt(2) / 3, "denominator"),
        (sp.I, 5, (sp.I - 2) * x - 1, "different leading|denominator"),
    ],
)
def test_unlucky_algebraic_images(alpha, prime, source, reason):
    pytest.importorskip("algroots._native_extension")
    ring = PolyRing((x,), sp.QQ.algebraic_field(alpha), "lex")
    result = native_field_basis(
        [ring.from_expr(source)], ring, limits=NativeF4Limits(first_prime=prime), certify=True
    )
    assert result.basis == tuple(
        ring.from_expr(p)
        for p in groebner_basis([source], (x,), domain=ring.domain, order="lex", backend="sympy")
    )
    assert result.stats["rejected_primes"][0]["prime"] == prime
    import re

    assert re.search(reason, result.stats["rejected_primes"][0]["reason"])


@pytest.mark.skipif(not compiled_available(), reason="native F4 is not installed")
@pytest.mark.parametrize("order", ["lex", "grlex", "grevlex"])
def test_repeated_roots(order):
    pytest.importorskip("algroots._native_extension")
    domain = sp.QQ.algebraic_field(sp.sqrt(2)).frac_field(t)
    source = [(x - t - sp.sqrt(2)) ** 3, y - x]
    algebra = QuotientAlgebra.from_polynomials(
        source, (x, y), domain=domain, order=order, groebner_backend="native"
    )
    assert algebra.dimension == 3
    assert algebra.geometric_solution_count == 1
    assert algebra.normal_form((x - t - sp.sqrt(2)) ** 2) != (0)
    assert algebra.normal_form((x - t - sp.sqrt(2)) ** 3) == 0


@pytest.mark.parametrize("backend", ["sympy", "native"])
@pytest.mark.parametrize(
    "source", [[t * x, u * y, x - y], [(t - u) * x**2, t * y, x * y], [(t - 1) * x, u * (x - 1)]]
)
def test_guard_intersections(backend, source):
    if backend == "native" and not compiled_available():
        pytest.skip("native F4 is not installed")
    proof = parametric_groebner_basis(source, (x, y), (t, u), backend=backend)
    assert proof.verify()
    for a, b in product([0, 1, 2], repeat=2):
        expected = groebner_basis([p.subs({t: a, u: b}) for p in source], (x, y), backend="sympy")
        assert proof.specialize({t: a, u: b}) == expected


def test_invalid_proof_coefficient():
    proof = parametric_groebner_basis([t * x], (x,), (t,))
    assert not replace(proof, source=(y,)).verify()


def test_zero_and_unit_guards():
    for source, expected in [([], []), ([0], []), ([1], [1]), ([t, 1 - t], [1])]:
        proof = parametric_groebner_basis(source, (x,), (t,))
        assert proof.verify()
        for value in [0, 1, 2]:
            assert list(proof.specialize({t: value})) == expected


@pytest.mark.skipif(not compiled_available(), reason="native F4 is not installed")
def test_characteristic_cache_separation():
    domain = sp.QQ.frac_field(t)
    ring = PolyRing((x, y), domain, "grevlex")
    source = [1009 * x + y, y * y - t]
    result = native_field_basis([ring.from_expr(p) for p in source], ring, certify=True)
    assert 1009 not in result.primes
    assert {p.as_expr() for p in result.basis} == set(
        groebner_basis(source, (x, y), domain=domain, backend="sympy")
    )


@pytest.mark.skipif(not compiled_available(), reason="native F4 is not installed")
def test_native_input_representations():
    from algroots.native_f4 import native_prime_basis

    ring = PolyRing((x, y), sp.GF(101), "grevlex")
    foreign = PolyRing((x, y), sp.GF(101), "lex")
    expressions = [x * x - y, y * y - 1]
    original = tuple(ring.from_expr(p) for p in expressions)
    saved = tuple(p.copy() for p in original)
    for source in (original, tuple(foreign.from_expr(p) for p in expressions), expressions):
        actual, _ = native_prime_basis(source, ring, 101)
        assert actual == original
        assert original == saved
