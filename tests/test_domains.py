"""Exact coefficient embeddings across polynomial, fraction, and matrix arithmetic."""

import pytest
import sympy as sp

from algroots.domains import domain_matrix, field_element, groebner, polynomial, polynomial_domain
from algroots.quotient import QuotientAlgebra

x, z, t = sp.symbols("x z t")


@pytest.mark.parametrize(
    "alpha", [sp.sqrt(-3), sp.sqrt(2), sp.real_root(2, 3), sp.sqrt(2) + sp.sqrt(3)]
)
def test_embeddings(alpha):
    ground = sp.QQ.algebraic_field(alpha)
    field = ground.frac_field(z)
    value = (z + alpha) / (z - alpha)
    native = field_element(value, field)
    assert sp.cancel(field.to_sympy(native) - value) == 0
    poly = polynomial(x * x - z - alpha, (x,), field)
    assert poly.domain == field and poly.all_coeffs() == [1, 0, -z - alpha]
    basis = groebner([poly], (x,), field)
    assert basis.reduce(poly)[1] == 0
    matrix = domain_matrix(sp.Matrix([[alpha, z], [0, alpha]]), field)
    assert sp.cancel(field.to_sympy(matrix.det()) - alpha * alpha) == 0
    algebra = QuotientAlgebra.from_polynomials([x * x - z - alpha], (x,), domain=field)
    assert algebra.trace(x) == 0
    assert sp.cancel(algebra.norm(x) + z + alpha) == 0
    assert algebra.minimal_polynomial(x, t).all_coeffs() == [1, 0, -z - alpha]
    assert algebra.normal_form(x * algebra.inverse(x)) == 1


def test_exact_inputs():
    with pytest.raises(ValueError, match="exact"):
        field_element(0.1, sp.QQ.frac_field(z))
    assert domain_matrix(sp.zeros(0, 3), sp.QQ).shape == (0, 3)


def test_large_unit():
    alpha = sp.sqrt(2)
    field = sp.QQ.algebraic_field(alpha)
    denominator = sp.expand((1 + alpha) ** 80)
    expected = (field.convert(alpha) - field.one) ** 80
    assert field_element(1 / denominator, field) == expected
    rational = field.frac_field(z)
    value = polynomial(x / denominator + z, (x,), rational)
    assert value.rep.to_dict()[(1,)] - rational.convert(expected, field) == rational.zero


def test_fraction_normalization():
    alpha = sp.sqrt(-3)
    ground = sp.QQ.algebraic_field(alpha)
    declared = ground.frac_field(z)
    arithmetic = polynomial_domain(declared)
    value = field_element((z + alpha) / (alpha * z + 1), arithmetic)
    assert ground.one == value.denom.LC
    assert value / value == arithmetic.one
    assert ground.one == (value + 1).denom.LC
    assert (
        field_element(declared.to_sympy(declared.convert(value, arithmetic)), arithmetic) == value
    )
    assert polynomial_domain(arithmetic) is arithmetic
