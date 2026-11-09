"""Reconstruct parameter functions in a fixed algebraic power basis."""

import pytest
import sympy as sp
from sympy.polys.rings import PolyRing

from algroots import NativeF4Limits, QuotientAlgebra, groebner_basis
from algroots.native_fields import native_field_basis

pytest.importorskip("algroots._native_extension")
x, y, t, u = sp.symbols("x y t u")


@pytest.mark.parametrize("order", ["lex", "grlex", "grevlex"])
@pytest.mark.parametrize("alpha", [sp.sqrt(2), -sp.sqrt(2), sp.I, sp.CRootOf(x**3 - x - 1, 0)])
def test_mixed_basis(order, alpha):
    domain = sp.QQ.algebraic_field(alpha).frac_field(t)
    equations = [(t + alpha) * x - 1, y - alpha * x]
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
        equations, (x, y), domain=domain, groebner_backend="native"
    )
    assert quotient.dimension == 1
    assert sp.simplify(quotient.normal_form(x) - 1 / (t + alpha)) == 0


def test_two_parameters():
    alpha = sp.sqrt(2)
    domain = sp.QQ.algebraic_field(alpha).frac_field(t, u)
    equations = [(t + alpha * u) * x - 1, y - x]
    ring = PolyRing((x, y), domain, "grevlex")
    result = native_field_basis([ring.from_expr(p) for p in equations], ring, certify=True)
    assert result.verified
    assert result.stats["constant_field_degree"] == 2
    assert result.basis == tuple(
        ring.from_expr(p.as_expr())
        for p in groebner_basis(equations, (x, y), domain=domain, backend="sympy").polys
    )
