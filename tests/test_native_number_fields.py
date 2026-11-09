"""FLINT extension-field images, factor CRT, and exact number-field lifting."""

import pytest
import sympy as sp
from sympy.polys.rings import PolyRing

from algroots import GroebnerLimitError, NativeF4Limits, QuotientAlgebra, groebner_basis
from algroots.native_fields import native_field_basis

extension = pytest.importorskip("algroots._native_extension")
x, y = sp.symbols("x y")


@pytest.mark.parametrize("order", ["lex", "grlex", "grevlex"])
@pytest.mark.parametrize(
    "alpha", [sp.sqrt(2), -sp.sqrt(2), sp.I, sp.sqrt(2) + sp.sqrt(3), sp.CRootOf(x**3 - x - 1, 0)]
)
def test_number_field_basis(order, alpha):
    domain = sp.QQ.algebraic_field(alpha)
    equations = [x * x - alpha * y, y * y - 1]
    basis = groebner_basis(
        equations,
        (x, y),
        domain=domain,
        order=order,
        backend="native",
        limits=NativeF4Limits(first_prime=5),
    )
    assert basis == groebner_basis(equations, (x, y), domain=domain, order=order, backend="sympy")
    quotient = QuotientAlgebra.from_polynomials(
        equations,
        (x, y),
        domain=domain,
        groebner_backend="native",
        groebner_limits=NativeF4Limits(first_prime=5),
    )
    assert quotient.dimension == 4
    assert quotient.normal_form(x * x - alpha * y) == 0


def test_extension_arithmetic():
    ring = PolyRing((x, y), sp.QQ.algebraic_field(sp.sqrt(2)), "grevlex")
    result = native_field_basis(
        (ring.from_expr(p) for p in [x * y - sp.sqrt(2), x * x - y]),
        ring,
        limits=NativeF4Limits(first_prime=5, collect_batch_metrics=True),
        certify=True,
    )
    assert result.verified
    assert result.stats["extension_calls"] > 0
    assert any(len(image["factor"]) == 3 for image in result.stats["images"])


def test_no_inert_primes():
    alpha = sp.sqrt(2) + sp.sqrt(3)
    ring = PolyRing((x,), sp.QQ.algebraic_field(alpha), "lex")
    result = native_field_basis(
        [ring.from_expr(x - alpha)],
        ring,
        limits=NativeF4Limits(first_prime=5, collect_batch_metrics=True),
        certify=True,
    )
    assert result.basis == (ring.from_expr(x - alpha),)
    assert all(len(image["factor"]) <= 3 for image in result.stats["images"])


def test_extension_limits():
    with pytest.raises(GroebnerLimitError, match="degree"):
        groebner_basis(
            [x - sp.sqrt(2)],
            (x,),
            domain=sp.QQ.algebraic_field(sp.sqrt(2)),
            backend="native",
            limits=NativeF4Limits(max_field_degree=1),
        )
    with pytest.raises(extension.LimitError, match="term"):
        extension.compute([{(2, 0): (1, 0), (0, 1): (0, 1)}], 5, [3, 0, 1], 2, max_terms=1)


def test_invalid_modulus():
    with pytest.raises(ValueError, match="irreducible"):
        extension.compute([{(1,): (1,)}], 5, [4, 0, 1], 1)
