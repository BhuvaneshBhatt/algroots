"""Compiled GF(p), sparse F4 row selection, and module-layout regressions."""

import random

import pytest
import sympy as sp
from sympy.polys.rings import PolyRing

from algroots.ff_sparse_matrix import compiled_available, reduce_rows
from algroots.module_f4 import module_f4_batch


@pytest.mark.parametrize("prime", [2, 3, 101, 1009, 65537])
def test_compiled_sparse_matches_python(prime):
    if not compiled_available():
        pytest.skip("optional C extension is not installed")
    rng = random.Random(417 + prime)
    for _ in range(12):
        ncols = rng.randrange(1, 45)
        rows = []
        for index in range(rng.randrange(1, 30)):
            if rows and index % 6 == 0:
                rows.append(dict(rows[-1]))
            else:
                rows.append(
                    {
                        i: rng.randrange(-prime * 3, prime * 3)
                        for i in rng.sample(range(ncols), rng.randrange(ncols + 1))
                    }
                )
        expected = reduce_rows(rows, ncols, prime, backend="python")
        actual = reduce_rows(rows, ncols, prime, backend="compiled")
        assert actual == expected
        for index, (row, operations, inverse) in enumerate(actual):
            replay = {k: v % prime for k, v in rows[index].items() if v % prime}
            for earlier, factor in operations:
                for column, value in actual[earlier][0].items():
                    new = (replay.get(column, 0) - factor * value) % prime
                    if new:
                        replay[column] = new
                    else:
                        replay.pop(column, None)
            assert {k: v * inverse % prime for k, v in replay.items()} == row


@pytest.mark.parametrize("strategy", ["full", "frontier"])
def test_module_layout_corruption_relearns_in_current_prime(strategy):
    x, y = sp.symbols("x y")
    cache = {}
    outputs = []
    for prime in (1009, 1013, 1019):
        ring = PolyRing((x, y), sp.GF(prime), "grevlex")
        basis = ((ring.from_expr(x + 1), ring.zero), (ring.from_expr(y + 1), ring.zero))
        first = module_f4_batch(
            [(0, 1)], basis, ring, symbolic_cache=cache, reducer_policy=strategy
        )
        outputs.append(tuple(tuple(poly.as_expr() for poly in row) for row in first))
        if prime == 1009:
            key = next(iter(cache))
            cache[key] = (((0, (999, 999)),), ())
        if prime == 1013:
            assert cache[key] != (((0, (999, 999)),), ())
    assert outputs[0] == outputs[1] == outputs[2]
