"""Independent correctness and resource contracts for the native F4 kernel."""

import random

import pytest
import sympy as sp
from sympy.polys.groebnertools import groebner
from sympy.polys.rings import PolyRing

from algroots.native_f4 import (
    NativeF4LimitError,
    NativeF4Limits,
    compiled_available,
    native_prime_basis,
)

pytestmark = pytest.mark.skipif(not compiled_available(), reason="native F4 extension unavailable")


def canonical(polynomials):
    return {frozenset(p.items()) for p in polynomials}


@pytest.mark.parametrize("prime", [2, 3, 101, 1009, 1000000007])
@pytest.mark.parametrize("order", ["lex", "grlex", "grevlex"])
def test_exact_basis(prime, order):
    x, y, z = sp.symbols("x y z")
    ring = PolyRing((x, y, z), sp.GF(prime), order)
    cases = [
        (),
        (0,),
        (1, x),
        (x, x - 1),
        (x, 2 * x),
        (x + y, x - y, y),
        (x * y - z, y * z - x, x * z - y),
        (x * x - y, y * y - z, z * z - x),
        (x * x - y, x * x - y, y * z - x),
        (x * x + y * z - 1, x * z + y * y - 1, z * z + x * y - 1),
    ]
    for case in cases:
        inputs = tuple(ring.from_expr(p) for p in case)
        original = tuple(p.copy() for p in inputs)
        expected = tuple(groebner([p for p in inputs if p], ring))
        result, stats = native_prime_basis(
            inputs,
            ring,
            prime,
            limits=NativeF4Limits(
                max_pairs=15000, max_basis=350, max_terms=100000, max_reduction_steps=3000000
            ),
        )
        assert canonical(result) == canonical(expected)
        assert inputs == original
        assert stats["pre_interreduction_basis"] >= len(result)


@pytest.mark.parametrize("order", ["lex", "grlex", "grevlex"])
def test_random_systems(order):
    rng = random.Random(9924)
    x, y, z = sp.symbols("x y z")
    ring = PolyRing((x, y, z), sp.GF(101), order)
    monomials = (1, x, y, z, x * x, y * y, z * z, x * y, x * z, y * z)
    for _ in range(15):
        inputs = tuple(
            ring.from_expr(sum(rng.randrange(-3, 4) * rng.choice(monomials) for _ in range(5)))
            for _ in range(3)
        )
        result, _ = native_prime_basis(inputs, ring, 101)
        assert canonical(result) == canonical(groebner([p for p in inputs if p], ring))


@pytest.mark.parametrize("nvars, exponent", [(2, 300), (9, 3), (64, 2)])
def test_unpacked_monomials(nvars, exponent):
    variables = sp.symbols(f"x0:{nvars}")
    ring = PolyRing(variables, sp.GF(101), "grevlex")
    inputs = tuple(
        ring.from_expr(p) for p in (variables[0] ** exponent - 1, variables[-1] - variables[0])
    )
    result, _ = native_prime_basis(
        inputs, ring, 101, limits=NativeF4Limits(max_reduction_steps=3000000)
    )
    assert canonical(result) == canonical(groebner(inputs, ring))


@pytest.mark.parametrize(
    "limit", [{"max_terms": 1}, {"max_basis": 1}, {"max_pairs": 1}, {"max_reduction_steps": 1}]
)
def test_budget(limit):
    x, y, z = sp.symbols("x y z")
    ring = PolyRing((x, y, z), sp.GF(101), "grevlex")
    inputs = tuple(ring.from_expr(p) for p in (x * y - z, y * z - x, x * z - y))
    with pytest.raises(NativeF4LimitError):
        native_prime_basis(inputs, ring, 101, limits=NativeF4Limits(**limit))


def test_matrix_metrics():
    x, y, z = sp.symbols("x y z")
    ring = PolyRing((x, y, z), sp.GF(101), "grevlex")
    inputs = tuple(ring.from_expr(p) for p in (x * y - z, y * z - x, x * z - y))
    result, stats = native_prime_basis(
        inputs, ring, 101, limits=NativeF4Limits(collect_batch_metrics=True)
    )
    assert canonical(result) == canonical(groebner(inputs, ring))
    batches = stats["batch_metrics"]
    assert sum(row[0] for row in batches) == stats["pairs_processed"]
    assert max(row[3] for row in batches) == stats["matrix_peak_rows"]
    assert max(row[4] for row in batches) == stats["matrix_peak_width"]
    assert all(len(row) == 11 and row[1] <= row[2] for row in batches if row[3])


@pytest.mark.parametrize("prime", [0, 1, 4, 15, 1000000009])
def test_direct_prime_validation(prime):
    from algroots import _native_f4

    with pytest.raises(RuntimeError, match="prime|limits"):
        _native_f4.compute([{(1,): 1}], prime, 1)


@pytest.mark.parametrize(
    "option",
    ["algorithm", "row_backend", "symbolic_rounds", "pivot_trace", "trace", "arena_polynomials"],
)
def test_removed_options(option):
    from algroots import _native_f4

    with pytest.raises(TypeError):
        _native_f4.compute([{(1,): 1}], 101, 1, **{option: None})
