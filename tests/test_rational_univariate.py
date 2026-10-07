import pytest
import sympy as sp

from algroots.errors import RationalUnivariateError
from algroots.rational_univariate import (
    compute_rational_univariate_representation,
    solve_rur_points,
    solve_rur_representation,
    solve_zero_dimensional_system_with_rur,
)


def test_rur_solves_two_point_system_exactly():
    x, y = sp.symbols("x y")
    rep = compute_rational_univariate_representation([x**2 + y**2 - 1, x - y], [x, y])
    roots = solve_rur_representation(rep, real=True)
    expected = {
        (-sp.sqrt(2) / 2, -sp.sqrt(2) / 2),
        (sp.sqrt(2) / 2, sp.sqrt(2) / 2),
    }
    assert set(roots) == expected
    assert rep.solution_count == 2
    assert rep.dimension == 2


def test_rur_points_expose_assignments():
    x = sp.Symbol("x")
    rep = compute_rational_univariate_representation([x**2 - 2], [x])
    points = solve_rur_points(rep)
    assert {point.assignment[x] for point in points} == {-sp.sqrt(2), sp.sqrt(2)}


def test_rur_complex_irreducible_cubic_uses_crootof():
    x = sp.Symbol("x")
    roots = solve_zero_dimensional_system_with_rur([x**3 - x + 1], [x], real=False)
    assert len(roots) == 3
    assert all(isinstance(root[0], sp.CRootOf) for root in roots)


def test_rur_records_quotient_multiplicity_separately_from_geometric_roots():
    x = sp.Symbol("x")
    rep = compute_rational_univariate_representation([x**2], [x])
    assert rep.dimension == 2
    assert rep.solution_count == 1
    assert solve_rur_representation(rep) == ((sp.Integer(0),),)


def test_rur_rejects_positive_dimensional_system():
    x, y = sp.symbols("x y")
    with pytest.raises(RationalUnivariateError):
        compute_rational_univariate_representation([x * y], [x, y])


def test_rur_uses_native_number_field_for_algebraic_coefficients():
    x = sp.Symbol("x")
    rep = compute_rational_univariate_representation([x**2 - sp.sqrt(2)], [x])
    assert rep.defining_polynomial.domain.is_AlgebraicField
    assert set(solve_rur_representation(rep, real=True)) == {
        (-sp.root(2, 4),),
        (sp.root(2, 4),),
    }


def test_rur_builds_compositum_for_multiple_algebraic_coefficients():
    x, y = sp.symbols("x y")
    rep = compute_rational_univariate_representation([x - sp.sqrt(2), y - sp.sqrt(3)], [x, y])
    assert rep.defining_polynomial.domain.is_AlgebraicField
    domain = rep.defining_polynomial.domain
    assert domain.to_sympy(domain.from_sympy(sp.sqrt(2))) == sp.sqrt(2)
    assert domain.to_sympy(domain.from_sympy(sp.sqrt(3))) == sp.sqrt(3)
    assert solve_rur_representation(rep, real=False) == ((sp.sqrt(2), sp.sqrt(3)),)


def test_rur_supports_rootof_coefficients():
    x, z = sp.symbols("x z")
    alpha = sp.CRootOf(z**3 - z - 1, 0)
    rep = compute_rational_univariate_representation([x - alpha], [x])
    roots = solve_rur_representation(rep, real=False)
    assert len(roots) == 1
    assert sp.simplify(roots[0][0] - alpha) == 0


def test_rur_supports_complex_algebraic_coefficient_fields():
    x = sp.Symbol("x")
    rep = compute_rational_univariate_representation([x**2 - (1 + sp.I)], [x])
    assert rep.defining_polynomial.domain.is_AlgebraicField
    roots = solve_rur_representation(rep, real=False)
    assert len(roots) == 2
    assert all(abs(complex(sp.N(root[0] ** 2 - (1 + sp.I), 50))) < 1e-40 for root in roots)
