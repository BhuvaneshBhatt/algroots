"""Exact generic RURs, coefficient domains, separator names, and profiling."""

import pytest
import sympy as sp

from algroots import groebner_basis
from algroots.native_f4 import NativeF4Limits, compiled_available
from algroots.profiling import GroebnerProfiler
from algroots.rational_univariate import (
    compute_rational_univariate_representation,
    solve_rur_points,
)
from algroots.rational_univariate.representation import RationalUnivariateError
from algroots.rational_univariate.solve import numerical_rur_roots

x, y, t, u = sp.symbols("x y t u")


@pytest.mark.parametrize("backend", ["sympy", "native"])
@pytest.mark.parametrize(
    "domain,coefficient",
    [
        (sp.QQ.frac_field(t), t),
        (sp.QQ.frac_field(t, u), t + u),
        (sp.QQ.algebraic_field(sp.sqrt(2)).frac_field(t), t + sp.sqrt(2)),
    ],
)
@pytest.mark.parametrize("multiplicity", [1, 2])
def test_generic_rur(backend, domain, coefficient, multiplicity):
    if backend == "native" and not compiled_available():
        pytest.skip("native F4 is not installed")
    if backend == "native" and domain.domain.is_AlgebraicField:
        pytest.importorskip("algroots._native_extension")
    source = [(x * x - coefficient) ** multiplicity, y - x]
    result = compute_rational_univariate_representation(
        source, (x, y), domain=domain, groebner_backend=backend
    )
    assert result.dimension == 2 * multiplicity
    assert result.solution_count == 2
    assert result.defining_polynomial.domain == domain
    substitution = dict(
        zip((x, y), (p.as_expr() for p in result.normalized_coordinate_polynomials()), strict=True)
    )
    for value in source:
        expression = sp.cancel(value.subs(substitution, simultaneous=True))
        assert (
            sp.Poly(expression, result.parameter, domain=domain)
            .rem(result.defining_polynomial)
            .is_zero
        )
    with pytest.raises(RationalUnivariateError, match="specialize"):
        solve_rur_points(result)
    with pytest.raises(RationalUnivariateError, match="specialize"):
        numerical_rur_roots(result, digits=30)


def test_separator_names():
    coefficient = sp.Symbol("_rur_t")
    domain = sp.QQ.frac_field(coefficient)
    result = compute_rational_univariate_representation([x - coefficient], (x,), domain=domain)
    assert result.parameter.name != "_rur_t"
    assert result.normalized_coordinate_polynomials()[0].as_expr() == coefficient
    for parameter in [coefficient, x, sp.Symbol("x", real=True), "z"]:
        with pytest.raises(RationalUnivariateError, match="distinct"):
            compute_rational_univariate_representation(
                [x - coefficient], (x,), parameter, domain=domain
            )


def test_rur_characteristic():
    with pytest.raises(RationalUnivariateError, match="characteristic-zero"):
        compute_rational_univariate_representation([x * x - 1], (x,), domain=sp.GF(5))


def test_profiler_contexts():
    with GroebnerProfiler() as outer:
        first = groebner_basis([x * x - y, y * y - 1], (x, y), backend="sympy")
        with GroebnerProfiler() as inner:
            second = groebner_basis([x * x - y, y * y - 1], (x, y), backend="sympy")
        with pytest.raises(ValueError, match="backend"):
            groebner_basis([x], (x,), backend="unknown")
        third = groebner_basis([x - 1], (x,), backend="sympy")
    assert first == second
    assert list(third) == [x - 1]
    assert [r["completed"] for r in outer.records] == [True, False, True]
    assert [r["basis_size"] for r in inner.records] == [2]
    assert outer.records[0]["verification_seconds"] == 0
    assert outer.records[0]["candidate_seconds"] > 0
    assert outer.records[0]["seconds"] >= outer.records[0]["candidate_seconds"]


@pytest.mark.skipif(not compiled_available(), reason="native F4 is not installed")
def test_profiled_native_fields():
    with GroebnerProfiler() as profile:
        basis = groebner_basis(
            [t * x - 1],
            (x,),
            domain=sp.QQ.frac_field(t),
            backend="native",
            limits=NativeF4Limits(collect_batch_metrics=True),
        )
    assert list(basis) == [x - 1 / t]
    record = profile.records[0]
    assert record["completed"]
    assert record["verification_seconds"] > 0
    stats = record["native"][0]
    assert stats["native_calls"] == len(stats["images"])
    assert all(image["profile"]["native_call_seconds"] > 0 for image in stats["images"])
