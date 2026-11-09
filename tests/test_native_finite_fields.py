"""Public finite-extension routing preserves the declared modulus and characteristic."""

import pytest
import sympy as sp
from sympy.polys.rings import PolyRing

from algroots import GroebnerLimitError, NativeF4Limits, QuotientAlgebra, groebner_basis
from algroots.backends import _extension_available
from algroots.finite_fields import finite_field
from algroots.groebner import compute_groebner_basis
from algroots.profiling import GroebnerProfiler

pytestmark = pytest.mark.skipif(
    not _extension_available(), reason="FLINT native extension unavailable"
)
x, y = sp.symbols("x y")


@pytest.mark.parametrize("prime,degree", [(2, 2), (3, 2), (5, 2), (2, 3)])
@pytest.mark.parametrize("order", ["lex", "grlex", "grevlex"])
def test_extension_basis(prime, degree, order):
    domain = finite_field(prime, degree)
    a = domain.symbol
    source = [x * y - a, y * y + (a + 1) * x + 1]
    with GroebnerProfiler() as profile:
        basis = groebner_basis(source, (x, y), domain=domain, order=order)
    assert basis == groebner_basis(source, (x, y), domain=domain, order=order, backend="sympy")
    assert basis.domain == domain
    assert basis.domain.characteristic() == prime
    assert all(basis.reduce(p)[1] == 0 for p in source)
    assert profile.records[0]["backend"] == "native"
    assert profile.records[0]["native"][0]["kind"] == "finite_extension"
    algebra = QuotientAlgebra.from_polynomials(source, (x, y), domain=domain, order=order)
    assert algebra.dimension == 3
    assert algebra.normal_form(x * y) == a


def test_expert_verification():
    domain = finite_field(3, 2)
    ring = PolyRing((x, y), domain, "lex")
    inputs = tuple(ring.from_expr(p) for p in (x * x - domain.symbol, y - x))
    before = tuple(p.copy() for p in inputs)
    assert compute_groebner_basis(inputs, ring) == compute_groebner_basis(
        inputs, ring, backend="sympy"
    )
    assert inputs == before


def test_extension_limits():
    domain = finite_field(3, 2)
    for limits in (NativeF4Limits(max_terms=1), NativeF4Limits(max_field_degree=1)):
        with pytest.raises(GroebnerLimitError):
            groebner_basis([x * x - domain.symbol, y - x], (x, y), domain=domain, limits=limits)
