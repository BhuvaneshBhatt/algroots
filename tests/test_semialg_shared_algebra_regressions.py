"""Algebraic regressions ported from semialg's former shared quotient/RUR layer."""

from __future__ import annotations

import pytest
import sympy as sp

from algroots.quotient import QuotientAlgebra
from algroots.rational_univariate import (
    compute_rational_univariate_representation,
    solve_rur_representation,
    solve_zero_dimensional_system_with_rur,
)


@pytest.mark.parametrize(
    ("equations", "quotient_dimension", "solution_count"),
    [
        (lambda x, y: (x**2, y**2), 4, 1),
        (lambda x, y: (x**2, y**3), 6, 1),
        (lambda x, y: ((x - 1) ** 3, y - x), 3, 1),
        (lambda x, y: ((x**2 - 2) ** 3, y - x), 6, 2),
    ],
)
def test_nonreduced_rur_separates_multiplicity_from_geometric_branches(
    equations, quotient_dimension, solution_count
) -> None:
    x, y = sp.symbols("x y", real=True)
    system = equations(x, y)

    representation = compute_rational_univariate_representation(system, (x, y))
    solutions = solve_zero_dimensional_system_with_rur(system, (x, y), real=True)

    assert representation.quotient_dimension == quotient_dimension
    assert representation.geometric_solution_count == solution_count
    assert representation.solution_count == solution_count
    assert len(solutions) == solution_count


@pytest.mark.parametrize("alpha", [sp.sqrt(2), sp.sqrt(3), sp.sqrt(2) + sp.sqrt(3)])
def test_nonreduced_algebraic_fields_preserve_nilpotent_dimension(alpha) -> None:
    x, y = sp.symbols("x y", real=True)
    system = ((x - alpha) ** 2, (y - x) ** 2)

    representation = compute_rational_univariate_representation(system, (x, y))
    solutions = solve_zero_dimensional_system_with_rur(system, (x, y), real=True)

    assert getattr(representation.defining_polynomial.domain, "is_AlgebraicField", False)
    assert representation.quotient_dimension == 4
    assert representation.geometric_solution_count == 1
    assert solutions == ((alpha, alpha),)


def test_multiple_algebraic_generators_use_one_exact_field() -> None:
    x, y = sp.symbols("x y", real=True)
    representation = compute_rational_univariate_representation(
        (x - sp.sqrt(2), y - sp.sqrt(3)), (x, y)
    )
    solutions = solve_zero_dimensional_system_with_rur(
        (x - sp.sqrt(2), y - sp.sqrt(3)), (x, y), real=True
    )

    assert getattr(representation.defining_polynomial.domain, "is_AlgebraicField", False)
    assert solutions == ((sp.sqrt(2), sp.sqrt(3)),)


def test_nonradical_two_branch_system_returns_geometric_roots_once() -> None:
    x, y = sp.symbols("x y", real=True)
    system = ((x**2 - 2) ** 2, y - x)

    assert solve_zero_dimensional_system_with_rur(system, (x, y), real=True) == (
        (-sp.sqrt(2), -sp.sqrt(2)),
        (sp.sqrt(2), sp.sqrt(2)),
    )


def test_existing_rur_representation_replays_without_reconstruction() -> None:
    x, y = sp.symbols("x y")
    system = (x**2 + y**2 - 1, x - y)
    representation = compute_rational_univariate_representation(system, (x, y))

    assert solve_rur_representation(representation, real=True) == (
        (-sp.sqrt(2) / 2, -sp.sqrt(2) / 2),
        (sp.sqrt(2) / 2, sp.sqrt(2) / 2),
    )


def test_rur_points_replay_against_original_polynomial_system() -> None:
    x, y = sp.symbols("x y", real=True)
    system = (x**2 + y**2 - 1, x - y)
    points = solve_zero_dimensional_system_with_rur(system, (x, y), real=True)

    assert points
    for point in points:
        assignment = dict(zip((x, y), point, strict=True))
        assert all(sp.simplify(poly.subs(assignment)) == 0 for poly in system)


def test_equation_scaling_permutation_and_recombination_preserve_finite_variety() -> None:
    x, y = sp.symbols("x y", real=True)
    f = x**2 - 2
    g = y - x
    systems = (
        (f, g),
        (-7 * g, 3 * f),
        (f + 5 * g, g),
        (g, f),
    )

    baseline = solve_zero_dimensional_system_with_rur(systems[0], (x, y), real=True)
    for system in systems:
        quotient = QuotientAlgebra.from_polynomials(system, (x, y))
        representation = compute_rational_univariate_representation(system, (x, y))
        points = solve_zero_dimensional_system_with_rur(system, (x, y), real=True)
        assert quotient.dimension == representation.quotient_dimension == 2
        assert points == baseline


def test_symbol_renaming_commutes_with_quotient_structure() -> None:
    x, y = sp.symbols("x y")
    u, v = sp.symbols("u v")
    original = QuotientAlgebra.from_polynomials((x**2 - 2, y - x), (x, y))
    renamed = QuotientAlgebra.from_polynomials((u**2 - 2, v - u), (u, v))

    assert original.dimension == renamed.dimension == 2
    assert original.geometric_solution_count == renamed.geometric_solution_count == 2
    assert original.standard_exponents == renamed.standard_exponents


def test_unit_ideal_has_zero_quotient_dimension_and_no_solutions() -> None:
    x = sp.Symbol("x")
    quotient = QuotientAlgebra.from_polynomials((x, x - 1), (x,))
    representation = compute_rational_univariate_representation((x, x - 1), (x,))

    assert quotient.dimension == 0
    assert representation.quotient_dimension == 0
    assert representation.geometric_solution_count == 0
    assert solve_zero_dimensional_system_with_rur((x, x - 1), (x,), real=False) == tuple()
