"""Differential tests for compiled monomials, S-pair scheduling and exact DAG CSE."""

import random

import pytest
import sympy as sp
from sympy.polys.domains import QQ
from sympy.polys.rings import ring

from algroots.compact_rational_proof import (
    RationalProofDAG,
    canonicalize_rational_dag,
    verify_rational_dag,
)
from algroots.monomial_ops import (
    first_divisors,
    native_available,
    pair_lcms,
    sort_monomials,
)


@pytest.mark.parametrize("seed", range(15))
def test_batch_monomials(seed):
    rng = random.Random(seed)
    leaders = tuple(tuple(rng.randrange(6) for _ in range(5)) for _ in range(12))
    queries = tuple(tuple(rng.randrange(9) for _ in range(5)) for _ in range(70))
    pairs = tuple((rng.randrange(len(leaders)), rng.randrange(len(leaders))) for _ in range(70))
    orders = ("lex", "grlex", "grevlex")
    for backend in ("python", "compiled") if native_available() else ("python",):
        assert first_divisors(leaders, queries, backend=backend) == first_divisors(
            leaders, queries, backend="python"
        )
        assert pair_lcms(leaders, pairs, backend=backend) == pair_lcms(
            leaders, pairs, backend="python"
        )
        for order in orders:
            for reverse in (True, False):
                assert sort_monomials(queries, order, reverse=reverse, backend=backend) == (
                    sort_monomials(queries, order, reverse=reverse, backend="python")
                )


def test_reject_invalid_exponents():
    with pytest.raises(ValueError):
        first_divisors([(1, 0)], [(1, -1)])
    with pytest.raises(ValueError):
        first_divisors([(1,)], [(1, 2)])
    with pytest.raises(IndexError):
        pair_lcms([(1, 2)], [(0, 1)])


def test_semantic_circuit_deduplication():
    r, x, y = ring("x,y", QQ, "grevlex")
    zero = (0, 0)
    original = (x, y)
    dag = RationalProofDAG(
        nodes=(
            ((-1, {zero: sp.Rational(1)}),),  # x
            ((-1, {zero: sp.Rational(2)}),),  # 2x
            ((0, {zero: sp.Rational(2)}),),  # 2x, different circuit
            (
                (1, {zero: sp.Rational(1)}),
                (2, {zero: sp.Rational(-1)}),
                (-2, {zero: sp.Rational(1)}),
            ),  # y
        ),
        roots=(3,),
        scales=(1, 1),
        generator_count=2,
    )
    assert verify_rational_dag(dag, original, (y,), r)
    result = canonicalize_rational_dag(dag, original, r)
    assert len(result.nodes) == 1
    assert result.roots == (0,)
    assert verify_rational_dag(result, original, (y,), r)
    forged = RationalProofDAG(dag.nodes, (1,), dag.scales, dag.generator_count)
    assert not verify_rational_dag(forged, original, (y,), r)


def test_module_divisibility_respects_position():
    from algroots.module_f4 import _matching_reducers

    leads = ((1, (1, 0), 1), (0, (2, 0), 1), (1, (0, 1), 1))
    queries = ((0, (1, 9)), (1, (2, 0)), (1, (0, 2)), (0, (3, 0)))
    for backend in ("python", "compiled") if native_available() else ("python",):
        assert _matching_reducers(leads, queries, monomial_backend=backend) == (-1, 0, 2, 1)
