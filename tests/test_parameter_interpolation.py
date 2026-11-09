"""Sparse modular interpolation independently of native Gröbner construction."""

from random import Random

import pytest
import sympy as sp

from algroots import NativeF4Limits
from algroots.native_f4 import NativeF4LimitError, compiled_available
from algroots.parameter_interpolation import (
    UnluckyInterpolation,
    _prony,
    evaluate,
    sparse_interpolate,
)


def test_sparse_rational_functions():
    prime = 1000003
    t, u = sp.symbols("t u")
    rng = Random(2901)
    for degree in range(1, 4):
        for _ in range(4):
            numerator = {
                (degree, 0): rng.randrange(1, prime),
                (0, degree): rng.randrange(1, prime),
                (0, 0): 1,
            }
            denominator = {(1, 0): 1, (0, 1): 2, (0, 0): 3}

            def sample(point, numerator=numerator, denominator=denominator):
                return {
                    "value": evaluate(numerator, point, prime)
                    * pow(evaluate(denominator, point, prime), -1, prime)
                    % prime
                }

            stats = {"sparse_lines": 0, "sparse_terms": 0}
            actual, divisor = sparse_interpolate(
                sample, 2, prime, NativeF4Limits(max_parameter_degree=degree), stats
            )["value"]

            def expression(terms):
                return sum(c * t**a * u**b for (a, b), c in terms.items())

            identity = expression(actual) * expression(denominator) - expression(
                numerator
            ) * expression(divisor)
            assert sp.Poly(identity, t, u, modulus=prime).is_zero
            assert actual
            assert divisor


def test_encoding_collision():
    with pytest.raises(UnluckyInterpolation, match="encodings"):
        _prony([1] * 6, 101, {1: None}, 4)


def test_sparse_degree_limit():
    with pytest.raises(NativeF4LimitError, match="degree"):
        sparse_interpolate(
            lambda p: {0: p[0] ** 2 % 101},
            2,
            101,
            NativeF4Limits(max_parameter_degree=0),
            {"sparse_lines": 0, "sparse_terms": 0},
        )


@pytest.mark.skipif(not compiled_available(), reason="native F4 is not installed")
def test_sparse_native_parameters():
    from sympy.polys.rings import PolyRing

    from algroots import groebner_basis
    from algroots.native_fields import native_field_basis

    x = sp.Symbol("x")
    parameters = sp.symbols("t:5")
    a, b, c, d, e = parameters
    domain = sp.QQ.frac_field(*parameters)
    coefficient = (a**6 + b**5 + c * d + e) / (1 + a * b)
    source = [x - coefficient]
    ring = PolyRing((x,), domain, "lex")
    result = native_field_basis([ring.from_expr(p) for p in source], ring, certify=True)
    assert result.basis == (ring.from_expr(source[0]),)
    assert result.stats["parameter_interpolation"] == "sparse"
    assert result.stats["sparse_lines"] > 0
    with pytest.raises(NativeF4LimitError, match="term|point"):
        groebner_basis(
            source,
            (x,),
            domain=domain,
            backend="native",
            limits=NativeF4Limits(parameter_interpolation="dense"),
        )
    for limits, message in [
        (NativeF4Limits(max_evaluation_points=4), "point"),
        (NativeF4Limits(max_sparse_monomials=10), "decoding"),
    ]:
        with pytest.raises(NativeF4LimitError, match=message):
            native_field_basis([ring.from_expr(p) for p in source], ring, limits=limits)


@pytest.mark.skipif(not compiled_available(), reason="native F4 is not installed")
def test_sparse_mixed_embedding():
    pytest.importorskip("algroots._native_extension")
    from algroots import groebner_basis

    x = sp.Symbol("x")
    t, u = sp.symbols("t u")
    for alpha in (sp.sqrt(2), -sp.sqrt(2)):
        domain = sp.QQ.algebraic_field(alpha).frac_field(t, u)
        source = [(t + alpha * u) * x - 1]
        actual = groebner_basis(
            source,
            (x,),
            domain=domain,
            backend="native",
            limits=NativeF4Limits(parameter_interpolation="sparse"),
        )
        assert actual == groebner_basis(source, (x,), domain=domain, backend="sympy")


@pytest.mark.parametrize("choice", ["unknown", 1, None])
def test_interpolation_choice(choice):
    with pytest.raises(ValueError, match="parameter_interpolation"):
        NativeF4Limits(parameter_interpolation=choice)


def test_line_matrix_limit():
    limits = NativeF4Limits(
        parameter_interpolation="sparse", max_parameter_degree=20, max_interpolation_terms=4
    )
    with pytest.raises(NativeF4LimitError, match="term"):
        sparse_interpolate(
            lambda point: {0: pow(point[0], 3, 1009)},
            1,
            1009,
            limits,
            {"sparse_lines": 0, "sparse_terms": 0},
        )
