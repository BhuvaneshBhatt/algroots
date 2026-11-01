"""Discover coefficient extensions, certify factor orbits and replay search scope."""

import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import sympy as sp

from algroots.curves import CurveError
from algroots.functional_decomposition import (
    AlgebraicCandidateAttempt,
    AlgebraicDiscoveryIssue,
    DecompositionLimitError,
    FunctionalDecompositionError,
    algebraic_decompose,
    discover_algebraic_factors,
)
from algroots.subfields import field_normal_form

x, z, t = sp.symbols("x z t")
PAPER = x**4 - z * x * x * (x + 1) + z**3 * (x + 1) ** 2


@pytest.mark.parametrize(
    "first", ["algroots._algebraic_discovery", "algroots.functional_decomposition"]
)
def test_import_orders_work_in_a_fresh_process(first, tmp_path):
    source = str(Path(__file__).resolve().parents[1] / "src")
    program = f"""import {first}
import sympy as sp
from algroots.functional_decomposition import discover_algebraic_factors
x,z=sp.symbols('x z')
assert discover_algebraic_factors(x*x+z*x+z,(x,z)).verify()
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


def test_paper_factor_orbits_cover_all_six_quadratic_factors():
    discovery = discover_algebraic_factors(PAPER, (x, z), degrees=(2,))
    assert discovery.complete and discovery.verify()
    (level,) = discovery.levels
    assert level.geometric_factor_count == 6
    assert [o.extension_degree for o in level.orbits] == [2, 4]
    assert sum(o.extension_degree for o in level.orbits) == 6
    for orbit in level.orbits:
        assert orbit.verify()
        difference = sp.Poly(orbit.factor * orbit.cofactor - PAPER, x)
        assert all(field_normal_form(orbit.ambient, c) == 0 for c in difference.all_coeffs())
    assert not replace(
        level, orbits=level.orbits[:1], component_polynomials=level.component_polynomials[:1]
    ).verify()
    bad = level.coordinate_polynomials[0] + 1
    assert not replace(
        level, coordinate_polynomials=(bad, *level.coordinate_polynomials[1:])
    ).verify()
    assert not replace(level.orbits[0], cofactor=level.orbits[0].cofactor + 1).verify()
    assert not replace(discovery, requested_degrees=(2, 3)).verify()


def test_automatic_paper_decomposition_needs_no_supplied_data():
    result = algebraic_decompose(PAPER, (x, z), parameter=t)
    assert result.status == "decomposable" and result.verify()
    assert result.examined == 1 and result.candidate_count == 4
    assert result.discovery.complete and not result.complete
    (D,) = result.decompositions
    assert D.verify() and D.exponent == 1
    assert not result.attempts[0].result.coefficient_field.contains(z)
    bad = replace(result.attempts[0].result, decomposition=replace(D, unit=2))
    assert not replace(result, attempts=(AlgebraicCandidateAttempt(0, bad),)).verify()
    assert not replace(result, attempts=(replace(result.attempts[0], index=1),)).verify()


def test_discovery_reaches_a_factor_extension_not_inside_a_root_field():
    discovery = discover_algebraic_factors(x**4 + x + z, (x, z), degrees=(2,))
    assert discovery.complete and discovery.verify()
    (orbit,) = discovery.orbits
    assert orbit.extension_degree == 6 and orbit.verify()
    # An extension of degree six cannot lie in the original quartic root field.
    assert 4 % orbit.extension_degree != 0


def test_complete_search_preserves_nontrivial_resultant_powers():
    result = algebraic_decompose(x**3 - 3 * x - z, (x, z), parameter=t, exhaustive=True)
    assert result.status == "decomposable" and result.complete and result.verify()
    assert result.examined == result.candidate_count == 2
    assert result.decompositions[0].exponent == 2
    assert result.attempts[1].result.reason == "trivial_component"


@pytest.mark.parametrize("polynomial", [x * x + z * x + z, x**3 + z * x + z**3 + z])
def test_exhaustive_negative_search_replays_every_rejection(polynomial):
    result = algebraic_decompose(polynomial, (x, z), parameter=t, exhaustive=True)
    assert result.complete and result.status == "indecomposable" and result.verify()
    assert all(a.result.reason in {"positive_genus", "trivial_component"} for a in result.attempts)
    assert not replace(result.discovery, levels=()).verify()


def test_missing_factor_degrees_and_candidate_limits_are_incomplete():
    F = x**3 + z * x + z**3 + z
    limited = algebraic_decompose(F, (x, z), max_candidates=0)
    assert limited.status == "incomplete" and not limited.complete and limited.verify()
    limited = algebraic_decompose(F, (x, z), max_candidates=1, exhaustive=True)
    assert limited.status == "incomplete" and limited.verify()
    scoped = algebraic_decompose(F, (x, z), factor_degrees=(2,), exhaustive=True)
    assert scoped.discovery.complete and scoped.status == "incomplete" and scoped.verify()
    assert not scoped.complete


def test_factor_dimension_limits_are_precomputed_and_accounted_for():
    discovery = discover_algebraic_factors(PAPER, (x, z), degrees=(2, 3, 4), max_factor_dimension=5)
    assert not discovery.complete and discovery.verify()
    assert discovery.issues == (AlgebraicDiscoveryIssue(2, "factor_dimension_limit"),)
    assert [level.degree for level in discovery.levels] == [3, 4]
    assert not replace(discovery, max_factor_dimension=6).verify()
    result = algebraic_decompose(PAPER, (x, z), max_factor_dimension=1, max_candidates=0)
    assert result.status == "incomplete" and result.verify()
    with pytest.raises(DecompositionLimitError):
        discover_algebraic_factors(x**7 + z, (x, z))


def test_separator_collision_has_a_real_budget_and_can_be_retried():
    F = PAPER.subs(x, -x)
    failed = discover_algebraic_factors(F, (x, z), degrees=(2,), max_separating_attempts=1)
    assert not failed.complete and failed.verify()
    assert failed.issues == (AlgebraicDiscoveryIssue(2, "separator_limit"),)
    retried = discover_algebraic_factors(F, (x, z), degrees=(2,), max_separating_attempts=2)
    assert retried.complete and retried.verify()


def test_nonmonic_input_and_all_requested_degrees_are_preserved():
    F = z * x * x + x + 1
    result = discover_algebraic_factors(F, (x, z))
    assert result.requested_degrees == (1, 2) and result.complete and result.verify()
    for orbit in result.orbits:
        assert all(
            field_normal_form(orbit.ambient, c) == 0
            for c in sp.Poly(z * orbit.factor * orbit.cofactor - F, x).all_coeffs()
        )


def test_exact_number_field_constants_and_internal_symbol_freshness():
    K = sp.QQ.algebraic_field(sp.sqrt(2))
    result = discover_algebraic_factors(x * x + sp.sqrt(2) * z * x + z, (x, z), constants=K)
    assert result.complete and result.verify()
    assert all(o.ambient.domain == K.frac_field(z) for o in result.orbits)
    u = sp.Dummy("factor_extension", positive=True)
    v = sp.Symbol("factor_coefficient_0", real=True)
    result = discover_algebraic_factors(u * u + v * u + v, (u, v))
    assert result.complete and result.verify()


def test_unresolved_reconstruction_is_not_an_indecomposability_proof(monkeypatch):
    import algroots.functional_decomposition as implementation

    def unsupported(*args, **kwargs):
        raise CurveError("unsupported test witness")

    monkeypatch.setattr(implementation, "algebraic_decompose", unsupported)
    result = algebraic_decompose(x * x + z * x + z, (x, z), exhaustive=True)
    assert result.examined == result.candidate_count == 1
    assert result.status == "incomplete" and not result.complete and result.verify()
    assert "CurveError" in result.attempts[0].error


def test_linear_output_has_no_nontrivial_minimal_outer_factor():
    result = algebraic_decompose(x - z**4, (x, z))
    assert result.status == "degree_below_two" and result.complete and result.verify()


@pytest.mark.parametrize("degrees", [(0,), (2, 1), (2, 2), (5,), (True,)])
def test_invalid_degree_requests(degrees):
    with pytest.raises(FunctionalDecompositionError):
        discover_algebraic_factors(PAPER, (x, z), degrees=degrees)


@pytest.mark.parametrize(
    "options", [{"max_factor_dimension": 0}, {"max_separating_attempts": 0}, {"max_degree": True}]
)
def test_invalid_discovery_budgets(options):
    with pytest.raises(FunctionalDecompositionError):
        discover_algebraic_factors(PAPER, (x, z), **options)


@pytest.mark.parametrize(
    "options",
    [
        {"max_candidates": -1},
        {"max_candidates": True},
        {"exhaustive": 1},
        {"point": (0, 0)},
        {"factor_degrees": (1,)},
    ],
)
def test_invalid_automatic_search_options(options):
    with pytest.raises(FunctionalDecompositionError):
        algebraic_decompose(PAPER, (x, z), **options)
