"""Finite-field coefficients and quotient factors, including bad trace characteristics."""

import pytest
import sympy as sp

from algroots.domains import field_element
from algroots.finite_decomposition import (
    frobenius_matrix,
    nilradical_basis,
    primary_decomposition,
    reduced_algebra,
)
from algroots.finite_fields import factor_univariate, field_elements, field_order, finite_field
from algroots.quotient import QuotientAlgebra

x, y, t, a = sp.symbols("x y t a")


@pytest.mark.parametrize("p,degree", [(2, 1), (3, 1), (2, 2), (3, 2), (2, 3)])
def test_coefficient_fields(p, degree):
    field = finite_field(p, degree)
    elements = tuple(field_elements(field))
    assert len(set(elements)) == field_order(field) == p**degree
    for value in elements:
        assert value ** (p**degree) == value
        if value:
            assert field.exquo(field.one, value) * value == field.one


@pytest.mark.parametrize("p,degree", [(2, 1), (3, 1), (2, 2), (3, 2)])
def test_finite_local_factors(p, degree):
    field = finite_field(p, degree)
    algebra = QuotientAlgebra.from_polynomials((x**p, y * y), (x, y), domain=field)
    assert algebra.geometric_solution_count == 1
    assert len(nilradical_basis(algebra)) == 2 * p - 1
    assert reduced_algebra(algebra).dimension == 1
    assert algebra.trace_pairing == sp.zeros(2 * p)
    ((component, residue),) = primary_decomposition(algebra)
    assert component.dimension == 2 * p and residue.dimension == 1 and residue.is_field
    q = field_order(field)
    operator = frobenius_matrix(algebra)
    assert operator.shape == (2 * p, 2 * p)
    assert algebra.normal_form(x**q) == 0


def test_nonmonogenic_product():
    algebra = QuotientAlgebra.from_polynomials((x * x - x, y * y - y), (x, y), domain=sp.GF(2))
    factors = primary_decomposition(algebra)
    assert len(factors) == 4
    assert sum(c.dimension for c, r in factors) == algebra.dimension
    assert all(c.dimension == r.dimension == 1 and r.is_field for c, r in factors)
    with pytest.raises(NotImplementedError, match="Frobenius"):
        algebra.separating_element(t)


def test_residue_extension():
    algebra = QuotientAlgebra.from_polynomials(((x * x + x + 1) ** 2,), (x,), domain=sp.GF(2))
    ((component, residue),) = primary_decomposition(algebra)
    assert component.dimension == 4 and residue.dimension == 2 and residue.is_field
    assert algebra.geometric_solution_count == 2
    assert len(nilradical_basis(algebra)) == 2


def test_factorization_and_scalars():
    field = finite_field(3, 2)
    polynomial = sp.Poly((x * x + 1) ** 3 * (x - 1), x, domain=field)
    unit, factors = factor_univariate(polynomial)
    rebuilt = sp.Poly(unit, x, domain=field)
    for factor, exponent in factors:
        rebuilt *= factor**exponent
    assert rebuilt == polynomial
    assert sorted((f.degree(), e) for f, e in factors) == [(1, 1), (1, 3), (1, 3)]
    value = field_element((field.symbol + 1) / 2, field)
    assert value * field.convert(2) == field.convert(field.symbol + 1)
    algebra = QuotientAlgebra.from_polynomials((x * x - field.to_sympy(value),), (x,), domain=field)
    assert algebra.minimal_polynomial(x, t) == sp.Poly(
        t * t - field.to_sympy(value), t, domain=field
    )
    assert algebra.trace(x) == 0 and algebra.norm(x) == field.to_sympy(-value)
    assert algebra.multiplication_matrix(x**4) == sp.eye(2) * field.to_sympy(value**2)


def test_imperfect_constants():
    field = sp.GF(2).frac_field(t)
    algebra = QuotientAlgebra.from_polynomials((x * x - t,), (x,), domain=field)
    assert algebra.minimal_polynomial(x, y) == sp.Poly(y * y - t, y, domain=field)
    with pytest.raises(NotImplementedError, match="finite"):
        nilradical_basis(algebra)
    with pytest.raises(NotImplementedError, match="finite"):
        primary_decomposition(algebra)


@pytest.mark.parametrize("arguments", [(9,), (3, 0), (True,), (3, 2, {"modulus": a * a - 1})])
def test_invalid_field(arguments):
    if isinstance(arguments[-1], dict):
        args, kwargs = arguments[:-1], arguments[-1]
    else:
        args, kwargs = arguments, {}
    with pytest.raises(ValueError):
        finite_field(*args, **kwargs)


@pytest.mark.parametrize(
    "domain",
    [sp.QQ, sp.ZZ, sp.GF(9), sp.QQ.frac_field(t)],
    ids=["rationals", "integers", "composite-modulus", "rational-functions"],
)
def test_factor_univariate_rejects_nonfields(domain):
    polynomial = sp.Poly(x**2 - 1, x, domain=domain)
    with pytest.raises(TypeError, match="finite coefficient field"):
        factor_univariate(polynomial)


def test_factor_univariate_requires_univariate_poly():
    with pytest.raises(ValueError, match="univariate Poly"):
        factor_univariate(x**2 - 1)
    with pytest.raises(ValueError, match="univariate Poly"):
        factor_univariate(sp.Poly(x + y, x, y, domain=sp.GF(2)))
