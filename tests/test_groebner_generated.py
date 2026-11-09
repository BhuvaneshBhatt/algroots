"""Bounded finite ideals under exact invertible changes of their generators."""

import pytest
import sympy as sp
from hypothesis import given, settings
from hypothesis import strategies as st

from algroots import QuotientAlgebra, groebner_backends, groebner_basis, polysolve

x, y = sp.symbols("x y")


@pytest.mark.parametrize("domain", [sp.QQ, sp.GF(2), sp.GF(5)])
@pytest.mark.parametrize("order", ["lex", "grlex", "grevlex"])
@settings(max_examples=12, deadline=None)
@given(a=st.integers(-2, 2), b=st.integers(-2, 2), c=st.integers(-2, 2), k=st.integers(-2, 2))
def test_generator_changes(domain, order, a, b, c, k):
    f, g = x * x - a, y - b * x - c
    reference = sp.groebner([f, g], x, y, domain=domain, order=order)
    scale = 3 if domain != sp.GF(5) else 2
    for inputs in ([f, g], [g, scale * (f + k * g)]):
        result = groebner_basis(inputs, (x, y), domain=domain, order=order, backend="sympy")
        assert result == reference
        quotient = QuotientAlgebra.from_polynomials(
            inputs, (x, y), domain=domain, order=order, groebner_backend="sympy"
        )
        assert quotient.dimension == 2
        assert quotient.normal_form(x * x) == domain.to_sympy(domain.convert(a))
        assert quotient.normal_form(y - b * x - c) == 0
    u, v = sp.symbols("u v")
    renamed = groebner_basis(
        [g.xreplace({x: u, y: v}), f.xreplace({x: u, y: v})],
        (v, u),
        domain=domain,
        order=order,
        backend="sympy",
    )
    assert all(renamed.reduce(p)[1] == 0 for p in (u * u - a, v - b * u - c))
    assert QuotientAlgebra.from_groebner_basis(renamed).dimension == 2


@pytest.mark.parametrize("backend", ["sympy", "native"])
@pytest.mark.parametrize("multiple", [False, True])
def test_construction_chain(backend, multiple):
    if not groebner_backends()[backend].available:
        pytest.skip(f"{backend} is not installed")
    equations = [x * x if multiple else x * x - 1, y - x]
    basis = groebner_basis(equations, (x, y), backend=backend)
    quotient = QuotientAlgebra.from_polynomials(equations, (x, y), groebner_backend=backend)
    result = polysolve(equations, (x, y), groebner_backend=backend, recognize=False, presolve=False)
    assert quotient.dimension == 2
    assert result.quotient_dimension == 2
    assert len(result.roots) == (1 if multiple else 2)
    assert all(basis.reduce(p)[1] == quotient.normal_form(p) == 0 for p in equations)
    for root in result.roots:
        assert abs(complex(root[0] - root[1])) < 1e-30
        assert abs(complex(root[0] ** 2 - (0 if multiple else 1))) < 1e-30
