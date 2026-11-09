"""Exact rational FLINT normal forms and explicitly managed worker pools."""

from random import Random

import pytest
import sympy as sp
from sympy.polys.rings import PolyRing

from algroots.basis_verification import verify_rational_basis
from algroots.qq_normal_form import FLINTQQNormalForm, QQNormalFormLimitError


@pytest.mark.parametrize("order", ["lex", "grlex", "grevlex"])
def test_rational_normal_forms_randomized(order):
    rng = Random(1349)
    xs = sp.symbols("x y z")
    ring = PolyRing(xs, sp.QQ, order)
    # Monic triangular basis allows an external, independent exact oracle.
    basis = tuple(
        ring.from_expr(p)
        for p in (
            xs[0] ** 2 - xs[1],
            xs[1] ** 2 - xs[2],
            xs[2] ** 2 - 2,
        )
    )
    engine = FLINTQQNormalForm(basis, ring)
    for _ in range(18):
        terms = {}
        for _ in range(10):
            mono = tuple(rng.randrange(4) for _ in xs)
            terms[mono] = sp.Rational(rng.randrange(-9, 10), rng.randrange(1, 6))
        polynomial = ring.from_dict(terms)
        expected = polynomial.rem(basis)
        actual = engine.remainder(polynomial)
        assert {k: sp.Rational(str(v)) for k, v in actual.to_dict().items()} == {
            tuple(k): sp.Rational(str(v)) for k, v in expected.terms()
        }
    assert not engine.remainder(ring.zero)


def test_certifying_flintrational_spairs_and_adversarial_basis():
    x, y = sp.symbols("x y")
    ring = PolyRing((x, y), sp.QQ, "grevlex")
    f = ring.from_expr(x**2 - y)
    g = ring.from_expr(y**2 - 2)
    basis = (f, g)
    witnesses = ((ring.one, ring.zero), (ring.zero, ring.one))
    assert verify_rational_basis(
        basis, basis, ring, membership_witnesses=witnesses, rational_normal_form_backend="flint"
    )
    assert verify_rational_basis(
        basis, basis, ring, membership_witnesses=witnesses, rational_normal_form_backend="sympy"
    )
    forged = (ring.from_expr(x**2 - y + 1), g)
    assert not verify_rational_basis(
        basis, forged, ring, membership_witnesses=witnesses, rational_normal_form_backend="flint"
    )
    invalid = (ring.from_expr(x * y - 1), ring.from_expr(x**2 - y))
    assert not verify_rational_basis(
        invalid,
        invalid,
        ring,
        membership_witnesses=((ring.one, ring.zero), (ring.zero, ring.one)),
        rational_normal_form_backend="flint",
    )


def test_explicit_qq_pass_budget():
    x = sp.symbols("x")
    ring = PolyRing((x,), sp.QQ, "lex")
    reducer = FLINTQQNormalForm((ring.from_expr(x - 1),), ring, max_passes=1)
    with pytest.raises(QQNormalFormLimitError):
        reducer.remainder(ring.from_expr(x**2))
