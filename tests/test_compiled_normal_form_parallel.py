"""Exact GF(p) normal forms and deterministic parallel prime certification."""

import random

import pytest
import sympy as sp
from sympy.polys.rings import PolyRing

from algroots.ff_normal_form import compiled_available, normal_form


def _dict_poly(poly):
    return {tuple(m): int(c) for m, c in poly.terms()}


@pytest.mark.parametrize("order", ["lex", "grlex", "grevlex"])
@pytest.mark.parametrize("prime", [2, 3, 101, 1009])
def test_native_exact_division_and_remainder(order, prime):
    rng = random.Random(1729 + prime + len(order))
    x, y, z = sp.symbols("x y z")
    ring = PolyRing((x, y, z), sp.GF(prime), order)
    for _ in range(18):

        def random_poly():
            terms = {
                (rng.randrange(4), rng.randrange(4), rng.randrange(3)): rng.randrange(1, prime)
                for _ in range(7)
            }
            return ring.from_dict(terms)

        f = random_poly()
        gs = [g for g in (random_poly(), random_poly()) if g]
        expected = normal_form(
            _dict_poly(f), [_dict_poly(g) for g in gs], prime, order, backend="python"
        )
        actual = normal_form(
            _dict_poly(f),
            [_dict_poly(g) for g in gs],
            prime,
            order,
            backend="compiled" if compiled_available() else "auto",
        )
        assert actual == expected
        qs, r = actual
        assert (
            sum((ring.from_dict(q) * g for q, g in zip(qs, gs, strict=True)), ring.zero)
            + ring.from_dict(r)
            == f
        )
        assert all(
            not any(all(a >= b for a, b in zip(m, tuple(g.LM), strict=True)) for g in gs) for m in r
        )


def test_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="prime"):
        normal_form({(1,): 1}, [], 9, "lex")
    with pytest.raises(ValueError, match="exponent"):
        normal_form(
            {(-1,): 1},
            [{(1,): 1}],
            101,
            "lex",
            backend="compiled" if compiled_available() else "python",
        )
    with pytest.raises(ZeroDivisionError):
        normal_form(
            {(1,): 1}, [{}], 101, "lex", backend="compiled" if compiled_available() else "python"
        )
