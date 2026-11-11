"""Fields generated over constants: rebasing and constructive Luroth maps."""

from dataclasses import replace

import pytest
import sympy as sp

from algroots.errors import QuotientAlgebraError
from algroots.function_fields import coefficient_field
from algroots.quotient import QuotientAlgebra

x, z, u, v, T = sp.symbols("x z u v T")


def test_cusp_generator_and_bidirectional_maps_with_poles():
    E = coefficient_field(
        [z * z, z**3], ambient=sp.QQ.frac_field(z), generator_symbols=(u, v), parameter=T
    )
    assert E.generator == z and E.generator_expression == v / u
    assert E.extension_degree == 1 and E.verify()
    assert E.express(z).expression == T and E.express(1 / (z + 1)).verify()
    P = E.rational_parametrization()
    assert P.coefficient_images == (T * T, T**3)
    assert P.to_source(z + 1) == T + 1 and P.to_ambient(T + 1) == z + 1
    assert P.forward_guard is sp.true and P.inverse_guard == sp.Ne(u, 0)
    assert P.verify()
    assert not replace(E.express(z), expression=T + 1).verify()
    assert not replace(E, generator_expression=u).verify()


def test_constructive_nonmembership():
    E = coefficient_field([z * z], ambient=sp.QQ.frac_field(z), parameter=T)
    assert E.generator == z * z and E.extension_degree == 2
    assert E.express(1 / (z * z + 1)).expression == 1 / (T + 1)
    assert not E.contains(z) and E.express(z).verify()
    assert E.verify()
    with pytest.raises(QuotientAlgebraError, match="does not belong"):
        E.rational_parametrization().to_source(z)
    with pytest.raises(QuotientAlgebraError, match="parameter"):
        E.to_ambient(z)


def test_nonpolynomial_composite_generator_and_saturated_image():
    h = z + 1 / z
    E = coefficient_field(
        [h * h, h**3], ambient=sp.QQ.frac_field(z), generator_symbols=(u, v), parameter=T
    )
    assert sp.cancel(E.generator - h) == 0 and E.extension_degree == 2
    assert E.verify() and not E.contains(z)
    assert E.express(h + 1 / h).verify()
    F = coefficient_field(
        [1 / z, 1 / (z - 1)], ambient=sp.QQ.frac_field(z), generator_symbols=(u, v)
    )
    assert F.verify() and F.contains(z) and F.extension_degree == 1


def test_algebraic_coefficient_field_does_not_silently_adjoin_z():
    A = QuotientAlgebra.from_polynomials([x * x - z * x + z**3], (x,), domain=sp.QQ.frac_field(z))
    E = coefficient_field([x], ambient=A, generator_symbols=(u,))
    assert E.transcendence_degree == 1 and E.degree == 1 and E.presentation == sp.QQ.frac_field(u)
    assert E.verify() and E.contains(x) and E.express(1 / (x + 1)).verify()
    assert not E.contains(z) and E.express(z).verify()
    assert A.subfield([x]).contains(z)  # Explicitly different base-field semantics.
    with pytest.raises(QuotientAlgebraError, match="coordinates"):
        E.to_ambient(z)


def test_rebased_nonlinear_presentation_and_membership():
    A = QuotientAlgebra.from_polynomials([x * x - z], (x,), domain=sp.QQ.frac_field(z))
    E = coefficient_field([x * x, x**3], ambient=A, generator_symbols=(u, v))
    assert E.relations == (u**3 - v * v,) and E.degree == 2 and E.verify()
    assert E.express(x).expression == v / u
    assert E.express(1 / (x + 1)).verify()
    wrong = QuotientAlgebra.from_polynomials([v * v - 2 * u**3], (v,), domain=sp.QQ.frac_field(u))
    assert not replace(E, presentation=wrong).verify()
    F = coefficient_field([1 / x], ambient=A, generator_symbols=(u,))
    assert F.verify() and F.contains(z)


@pytest.mark.parametrize("generators", [[], [sp.Integer(2)], [x]])
def test_algebraic_constants_vs_generators(generators):
    A = QuotientAlgebra.from_polynomials([x * x - 2], (x,), domain=sp.QQ.frac_field(z))
    E = coefficient_field(generators, ambient=A)
    assert E.transcendence_degree == 0 and E.verify()
    assert not E.contains(z) and E.express(z).verify()
    assert E.contains(x) == bool(generators and generators[0] == x)
    if E.contains(x):
        assert E.express(1 / x).verify() and E.degree == 2


def test_rational_constants_and_algebraic_number_constants():
    E = coefficient_field([], ambient=sp.QQ.frac_field(z))
    assert E.verify() and E.contains(2) and not E.contains(z)
    with pytest.raises(QuotientAlgebraError, match="constant"):
        E.rational_parametrization()
    K = sp.QQ.algebraic_field(sp.sqrt(2))
    F = coefficient_field([z * z + sp.sqrt(2)], ambient=K.frac_field(z), constants=K, parameter=T)
    assert F.verify() and F.extension_degree == 2
    assert F.express(z * z).verify() and not F.contains(z)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ambient": sp.QQ},
        {"ambient": sp.QQ.poly_ring(z, T)},
        {"ambient": sp.QQ.frac_field(z), "constants": sp.QQ.frac_field(T)},
        {"ambient": sp.QQ.frac_field(z), "parameter": z},
        {"ambient": sp.QQ.frac_field(z), "generator_symbols": (z,)},
    ],
)
def test_invalid_constant_field_and_coordinate_contracts(kwargs):
    with pytest.raises(QuotientAlgebraError):
        coefficient_field([z * z], **kwargs)
