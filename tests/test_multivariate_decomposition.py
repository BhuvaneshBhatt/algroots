"""GRS02 decomposition identities, exhaustive classes, limits and replay."""

import os
import subprocess
import sys
from dataclasses import replace
from itertools import combinations
from pathlib import Path

import pytest
import sympy as sp

from algroots.function_fields import luroth_generator
from algroots.functional_decomposition import (
    DecompositionLimitError,
    FunctionalDecompositionError,
    MultivariateRationalDecomposition,
    MultivariateRationalDecompositionResult,
    RationalDecompositionResult,
    multivariate_rational_decompose,
    rational_decompose,
)

x, y, z, t = sp.symbols("x y z t")


def assert_compositions(result):
    assert result.verify()
    for decomposition in result.decompositions:
        assert decomposition.verify()
        assert (
            sp.cancel(decomposition.outer.subs(t, decomposition.inner) - result.original_expression)
            == 0
        )


@pytest.mark.parametrize(
    "function,degrees",
    [
        ((x * x + y * y) ** 2, {(2, 2)}),
        ((x * y + z) ** 3, {(3, 2)}),
        ((x + y) ** 6, {(3, 2), (2, 3)}),
        (
            ((x * x + y * y) ** 2 + (x * y + 1) ** 2)
            / ((x * x + y * y) ** 2 + (x * x + y * y) * (x * y + 1) + 2 * (x * y + 1) ** 2),
            {(2, 2)},
        ),
    ],
)
def test_polynomial_and_rational_multivariate_components(function, degrees):
    result = multivariate_rational_decompose(function, (x, y, z), parameter=t)
    assert isinstance(result, MultivariateRationalDecompositionResult)
    assert result.complete and result.status == "decomposable"
    assert {d.degrees for d in result.decompositions} == degrees
    assert_compositions(result)
    assert not replace(result, decompositions=()).verify()


def test_diagonal_factor_is_not_required_to_be_linear():
    result = multivariate_rational_decompose((x * x + y * y) ** 2, (x, y), parameter=t)
    assert result.candidate_count == 1 and len(result.factors) == 2
    assert all(p.total_degree() == 2 for p in result.factors)
    assert result.decompositions[0].inner == x * x + y * y
    assert_compositions(result)


def test_grs02_published_example_2_1():
    a, b = x * y + x * z * z - y**5, 9 * x + 25 * y * z
    function = (a * a - b * b) / (a * a - 2 * b * b)
    result = multivariate_rational_decompose(function, (x, y, z), parameter=t)
    assert result.complete and result.status == "decomposable"
    assert result.candidate_count == 1
    assert [d.degrees for d in result.decompositions] == [(2, 5)]
    assert_compositions(result)
    field = luroth_generator([result.decompositions[0].inner], variables=(x, y, z))
    assert field.contains(a / b)  # Published and returned generators are Möbius-equivalent.


def test_all_three_mobius_classes_of_a_rational_laurent_composition():
    result = multivariate_rational_decompose((x / y + y / x) ** 2, (x, y), parameter=t)
    assert result.complete and len(result.decompositions) == 3
    assert {d.degrees for d in result.decompositions} == {(2, 2)}
    assert_compositions(result)
    fields = [luroth_generator([d.inner], variables=(x, y)) for d in result.decompositions]
    assert all(
        not field.contains(other.inner)
        for i, field in enumerate(fields)
        for j, other in enumerate(result.decompositions)
        if i != j
    )


def test_products_of_irreducibles_and_nonnear_candidates_are_both_visited():
    result = multivariate_rational_decompose((x + y) ** 6, (x, y), parameter=t)
    assert result.candidate_count == 3 and len(result.decompositions) == 2
    assert len(result.factors) == 4
    # No irreducible factor alone contains all the degree-three near polynomial.
    assert any(d.degrees == (2, 3) for d in result.decompositions)
    assert_compositions(result)


def test_candidate_count_matches_independent_small_powerset_oracle():
    result = multivariate_rational_decompose((x + y) ** 12, (x, y), parameter=t, max_candidates=1)
    factors, expected = result.factors, 0
    targets = {2, 3, 4, 6}
    comparison = result.comparison_variables
    for length in range(1, len(factors) + 1):
        for subset in combinations(factors, length):
            product = sp.Poly(sp.prod(p.as_expr() for p in subset), *comparison, x, y)
            dx = max(sum(m[:2]) for m in product.monoms())
            dy = max(sum(m[2:]) for m in product.monoms())
            diagonal = sp.Poly(
                product.as_expr().xreplace(dict(zip(comparison, (x, y), strict=True))), x, y
            )
            expected += int(dx == dy and dx in targets and diagonal.is_zero)
    assert result.candidate_count == expected and expected > 1
    assert result.examined == 1 and not result.complete and result.verify()


def test_candidate_budgets_are_replayable_and_never_prove_false_negatives(monkeypatch):
    import algroots._multivariate_decomposition as implementation

    empty = multivariate_rational_decompose((x + y) ** 6, (x, y), parameter=t, max_candidates=0)
    assert empty.status == "incomplete" and not empty.complete and empty.verify()
    calls, original = [], implementation._recover

    def count(*args):
        calls.append(args[-1])
        return original(*args)

    with monkeypatch.context() as patch:
        patch.setattr(implementation, "_recover", count)
        partial = multivariate_rational_decompose(
            (x + y) ** 6, (x, y), parameter=t, max_candidates=1
        )
        assert len(calls) == 1
    assert partial.status == "decomposable" and not partial.complete and partial.verify()
    assert partial.examined == 1 and partial.candidate_count == 3
    assert not replace(partial, examined=3).verify()


@pytest.mark.parametrize("function", [x**4 + y, (x**4 + y) / (x + y + 1), x**4 + y**4])
def test_composite_degree_functions_can_have_complete_negative_results(function):
    result = multivariate_rational_decompose(function, (x, y), parameter=t)
    assert result.complete and result.status == "indecomposable" and not result.decompositions
    assert result.factors and result.verify()


@pytest.mark.parametrize("function", [0, 2, x + y, (x + y) / (x - y)])
def test_constant_and_degree_one_results(function):
    result = multivariate_rational_decompose(function, (x, y), parameter=t)
    assert result.status == "degree_below_two" and result.complete and result.verify()
    assert not result.factors and not result.decompositions


def test_prime_total_degree_and_degree_limits_do_not_factor(monkeypatch):
    def no_factor(*args, **kwargs):
        raise AssertionError("factorization must not be reached")

    with monkeypatch.context() as patch:
        patch.setattr(sp.Poly, "factor_list", no_factor)
        prime = multivariate_rational_decompose((x + y) ** 3, (x, y), parameter=t)
        assert prime.complete and prime.status == "indecomposable" and prime.verify()
        assert prime.factors == () and prime.candidate_count == 0
        with pytest.raises(DecompositionLimitError, match="total degree"):
            multivariate_rational_decompose((x + y) ** 12, (x, y), max_degree=8)
    # GRS02's nontriviality convention excludes linear multivariate inner maps.
    quadratic = multivariate_rational_decompose((x + y) ** 2, (x, y))
    assert quadratic.status == "indecomposable" and quadratic.verify()


@pytest.mark.parametrize("constant", [sp.sqrt(2), sp.I])
def test_exact_number_field_constants(constant):
    constants = sp.QQ.algebraic_field(constant)
    h = x * x + constant * x * y + y * y
    result = multivariate_rational_decompose(
        h * h + constant, (x, y), constants=constants, parameter=t
    )
    assert result.complete and result.status == "decomposable"
    assert result.decompositions[0].constants == constants
    assert_compositions(result)


def test_rational_poles_and_reduction():
    function = (x * x + y * y) ** 2
    original = sp.Mul(function * (x - y), sp.Pow(x - y, -1, evaluate=False), evaluate=False)
    result = multivariate_rational_decompose(original, (x, y), parameter=t)
    decomposition = result.decompositions[0]
    assert decomposition.original_expression == original
    assert sp.expand(decomposition.function - function) == 0
    assert decomposition.input_guard == sp.Ne(x - y, 0)
    assert_compositions(result)


@pytest.mark.parametrize("function", [x**6, (x + 1 / x) ** 2, (x**4 + x) / (x + 1)])
def test_univariate_parity_and_sequence_dispatch_preserve_legacy_contract(function):
    scalar_route = rational_decompose(function, x, parameter=t)
    direct = multivariate_rational_decompose(function, (x,), parameter=t)
    dispatch = rational_decompose(function, (x,), parameter=t)
    assert isinstance(scalar_route, RationalDecompositionResult)
    assert isinstance(dispatch, MultivariateRationalDecompositionResult)
    assert direct.status == dispatch.status == scalar_route.status
    assert {d.degrees for d in direct.decompositions} == {
        d.degrees for d in scalar_route.decompositions
    }
    assert len(direct.decompositions) == len(scalar_route.decompositions)
    assert_compositions(direct)
    assert dispatch.verify()


@pytest.mark.parametrize("variables", [(x, y), (y, x), (z, y, x)])
def test_variable_order_unused_variables_and_symbol_identity(variables):
    result = rational_decompose((x * x + y * y) ** 2, list(variables), parameter=t)
    assert result.variables == variables and result.parameter is t
    assert_compositions(result)
    a = sp.Dummy("x", positive=True)
    b = sp.Symbol("x", real=True)
    result = multivariate_rational_decompose((a * a + b * b) ** 2, (a, b), parameter=t)
    assert result.variables[0] is a and result.variables[1] is b
    assert_compositions(result)


def test_generator_variable_sequence_and_polynomial_input():
    result = multivariate_rational_decompose(
        sp.Poly((x * x + y * y) ** 2, x, y), (v for v in (x, y)), parameter=t
    )
    assert_compositions(result)


@pytest.mark.parametrize("expression", [x + z, x + sp.sqrt(2), sp.sin(x), x + 0.5, sp.oo, sp.zoo])
def test_invalid_rational_input(expression):
    with pytest.raises(FunctionalDecompositionError):
        multivariate_rational_decompose(expression, (x, y))


@pytest.mark.parametrize(
    "options",
    [
        {"variables": ()},
        {"variables": (x, x)},
        {"variables": (x, 1)},
        {"variables": x},
        {"variables": (x, sp.Symbol("n", commutative=False))},
        {"parameter": x},
        {"parameter": 1},
        {"max_candidates": -1},
        {"max_candidates": True},
        {"max_degree": 0},
        {"max_degree": False},
        {"constants": sp.RR},
        {"constants": sp.GF(7)},
        {"constants": sp.QQ.frac_field(z)},
    ],
)
def test_invalid_options(options):
    kwargs = {"variables": (x, y)}
    kwargs.update(options)
    with pytest.raises(ValueError):
        multivariate_rational_decompose((x * x + y * y) ** 2, **kwargs)


def test_unsupported_operations_propagate_without_indecision_certificates(monkeypatch):
    def unsupported(*args, **kwargs):
        raise NotImplementedError("backend does not support this exact operation")

    with monkeypatch.context() as patch:
        patch.setattr(sp.Poly, "factor_list", unsupported)
        with pytest.raises(NotImplementedError):
            multivariate_rational_decompose((x * x + y * y) ** 2, (x, y))


def test_search_and_composition_tampering_are_rejected():
    result = multivariate_rational_decompose((x * x + y * y) ** 2, (x, y), parameter=t)
    decomposition = result.decompositions[0]
    for changes in (
        {"outer": t * t + 1},
        {"inner": x + y},
        {"outer": t * t + x},
        {"inner": x * x + z * z},
        {"parameter": x},
        {"variables": (x, x)},
        {"constants": sp.ZZ},
    ):
        assert not replace(decomposition, **changes).verify()
    for changes in (
        {"decompositions": ()},
        {"factors": result.factors[:-1]},
        {"factors": (result.factors[0] * 2, *result.factors[1:])},
        {"examined": -1},
        {"examined": True},
        {"examined": 0},
        {"candidate_count": result.candidate_count + 1},
        {"comparison_variables": (x, z)},
        {"comparison_variables": ()},
        {"comparison_variables": (z, z)},
        {"parameter": x},
        {"constants": sp.ZZ},
    ):
        assert not replace(result, **changes).verify()
    assert not MultivariateRationalDecomposition(x + y, (x, y), t, sp.QQ, t, x + y).verify()


@pytest.mark.parametrize(
    "first", ["algroots._multivariate_decomposition", "algroots.functional_decomposition"]
)
def test_fresh_import_orders_and_public_namespace(first, tmp_path):
    source = str(Path(__file__).resolve().parents[1] / "src")
    program = f"""import {first}
import algroots
from algroots.functional_decomposition import multivariate_rational_decompose
import sympy as sp
x,y=sp.symbols("x y")
assert multivariate_rational_decompose((x*x+y*y)**2,(x,y)).verify()
assert not hasattr(algroots,"multivariate_rational_decompose")
"""
    process = subprocess.run(
        [sys.executable, "-c", program],
        cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH=source),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert process.returncode == 0, process.stdout + process.stderr
