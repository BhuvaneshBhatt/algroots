"""Independent finite-extension differential checks for the FLINT F4 kernel."""

from random import Random

import pytest
import sympy as sp
from sympy.polys.groebnertools import groebner
from sympy.polys.rings import PolyRing

from algroots.finite_fields import finite_field

native = pytest.importorskip("algroots._native_extension")


@pytest.mark.parametrize("order", ["lex", "grlex", "grevlex"])
def test_finite_extension_reference(order):
    x, y, a = sp.symbols("x y a")
    field = finite_field(5, 2, name=a, modulus=a * a + 2)
    ring = PolyRing((x, y), field, order)
    rng = Random(984)
    for iteration in range(12):
        expressions = [
            (x * x if iteration % 2 else x * y)
            + sum((rng.randrange(5) + rng.randrange(5) * a) * m for m in [x, y, 1]),
            y * y + sum((rng.randrange(5) + rng.randrange(5) * a) * m for m in [x, y, 1]),
        ]
        inputs = [ring.from_expr(f) for f in expressions]
        sparse = []
        for poly in inputs:
            row = {}
            for monomial, coefficient in poly.items():
                low = [int(c) % 5 for c in reversed(coefficient.rep.to_list())]
                row[tuple(monomial)] = tuple(low + [0] * (2 - len(low)))
            sparse.append(row)
        basis, stats = native.compute(
            sparse, 5, [2, 0, 1], 2, order, max_reduction_steps=1000000, collect_batch_metrics=True
        )
        actual = tuple(
            ring.from_dict(
                {
                    m: field.convert(sum(c * a**i for i, c in enumerate(value)))
                    for m, value in row.items()
                }
            )
            for row in basis
        )
        expected = tuple(groebner(inputs, ring))
        assert actual == expected
        assert stats["algorithm"] == "flint_sparse_f4"
        assert stats["field_degree"] == 2
        assert sum(batch["pairs"] for batch in stats["batch_metrics"]) == stats["processed_pairs"]


def test_direct_input_validation():
    with pytest.raises(ValueError, match="prime"):
        native.compute([{(1,): (1, 0)}], 9, [1, 0, 1], 1)
    with pytest.raises(ValueError, match="width"):
        native.compute([{(1, 2): (1, 0)}], 5, [2, 0, 1], 1)
    with pytest.raises(ValueError, match="canonical"):
        native.compute([{(1,): (1, 0, 1)}], 5, [2, 0, 1], 1)


def test_parallel_fields():
    from concurrent.futures import ThreadPoolExecutor

    inputs = [{(1, 1): (0, 1), (0, 0): (1, 0)}, {(2, 0): (1, 0), (0, 1): (1, 1)}]

    def compute(_):
        return native.compute(inputs, 5, [2, 0, 1], 2)[0]

    reference = compute(0)
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(compute, range(16))) == [reference] * 16
