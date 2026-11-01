"""Constructive fixed-base field membership and both directions of embeddings."""

from dataclasses import replace

import pytest
import sympy as sp

from algroots.errors import QuotientAlgebraError
from algroots.quotient import QuotientAlgebra

x, y, z, u, v, T = sp.symbols("x y z u v T")


def test_proper_subfield_and_constructive_negative_membership():
    A = QuotientAlgebra.from_polynomials([x**4 - z], (x,), domain=sp.QQ.frac_field(z))
    E = A.subfield([x * x], generator_symbols=(u,))
    assert E.dimension == 2 and E.verify()
    member = E.express(1 / (x * x + 1))
    assert member.belongs and member.verify()
    assert A.normal_form((x * x + 1) * E.to_ambient(member.expression) - 1) == 0
    negative = E.express(x)
    assert not negative and negative.expression is None and negative.verify()
    assert not replace(member, coordinates=(sp.Integer(0), sp.Integer(0))).verify()
    assert E.contains(z)  # The fixed-base construction includes the base parameter.
    with pytest.raises(QuotientAlgebraError, match="formal"):
        E.to_ambient(x)
    with pytest.raises(QuotientAlgebraError, match="over="):
        A.subfield([x * x], over=sp.QQ)
    P = E.primitive_element(T)
    assert P.field.dimension == 2 and P.verify()
    assert A.normal_form(P.to_ambient(P.to_source(x * x + 1)) - (x * x + 1)) == 0
    assert P.field.normal_form(P.to_source(P.to_ambient(1 / (T + 1))) - P.field.inverse(T + 1)) == 0
    with pytest.raises(QuotientAlgebraError, match="does not belong"):
        P.to_source(x)
    assert not replace(P, generator_images=(sp.Integer(0),)).verify()
    wrong = QuotientAlgebra.from_polynomials([T * T - 2 * z], (T,), domain=A.domain)
    assert not replace(P, field=wrong).verify()


def test_multigenerator_primitive_element_exhausts_bad_first_choice():
    A = QuotientAlgebra.from_polynomials([x * x - 2, y * y - 3], (x, y))
    E = A.subfield([x, y], generator_symbols=(u, v))
    assert E.dimension == 4 and E.verify()
    P = E.primitive_element(T)
    assert P.field.dimension == 4 and P.verify()
    assert P.ambient_generator == x + y  # choice=0 generates only QQ(sqrt(2)).
    assert P.to_ambient(P.generator_images[0]) == x
    assert P.to_ambient(P.generator_images[1]) == y
    assert E.express(x * y).verify()


@pytest.mark.parametrize("generators", [[], [sp.Integer(2)], [x * x]])
def test_constant_subfields_and_empty_generating_set(generators):
    A = QuotientAlgebra.from_polynomials([x * x - 2], (x,))
    E = A.subfield(generators)
    assert E.dimension == 1 and E.verify()
    assert E.contains(sp.Rational(3, 5)) and not E.contains(x)
    P = E.primitive_element(T)
    assert P.verify() and P.to_source(3) == 3


def test_number_constants_and_invalid_inputs():
    K = sp.QQ.algebraic_field(sp.sqrt(2))
    A = QuotientAlgebra.from_polynomials([x * x - z], (x,), domain=K.frac_field(z))
    E = A.subfield([x], generator_symbols=(u,))
    assert E.verify() and E.express(sp.sqrt(2) / (x + 1)).verify()
    assert E.to_ambient(u + sp.sqrt(2) * z) == x + sp.sqrt(2) * z
    for symbols in [(z,), (x,), (), (u, u)]:
        with pytest.raises(QuotientAlgebraError):
            A.subfield([x], generator_symbols=symbols)
    with pytest.raises(QuotientAlgebraError, match="exact"):
        A.subfield([sp.Float(0.5)])
    for relation in [x * x, x * x - 1]:
        B = QuotientAlgebra.from_polynomials([relation], (x,))
        with pytest.raises(QuotientAlgebraError, match="field"):
            B.subfield([x])
