import pytest
import sympy as sp

from algroots import polysolve
from algroots.presolve import _bounded_substitution, polynomial_presolve

x, y, z = sp.symbols("x y z")


@pytest.mark.parametrize("coefficient", [sp.Integer(1), sp.Rational(2, 3), sp.sqrt(2)])
def test_cancellation_unlocks_guarded_substitution(coefficient):
    expression = coefficient * (y**2 + z**2)
    equations = (x - expression, x**2 - expression**2 + y**2 - 2, z**2 - 1)
    reduced = polynomial_presolve(equations, (x, y, z), max_terms=2, growth_factor=1)
    assert reduced.substitutions[0][0] == x
    assert reduced.cancellation_probes >= 1 and reduced.cancellation_acceptances >= 1
    assert x not in reduced.variables
    result = polysolve(
        equations,
        (x, y, z),
        presolve_max_terms=2,
        presolve_growth_factor=1,
        recognize=False,
        digits=35,
    )
    oracle = polysolve(
        equations, (x, y, z), presolve=False, method="rur", recognize=False, digits=35
    )
    assert len(result.roots) == len(oracle.roots) == 4
    assert result.total_multiplicity == oracle.total_multiplicity
    for root in result.roots:
        assert (
            min(
                max(abs(complex(a - b)) for a, b in zip(root, other, strict=True))
                for other in oracle.roots
            )
            < 1e-25
        )


def test_cancellation_preserves_singular_multiplicity():
    expression = y**2 + z**2
    equations = (x - expression, x**2 - expression**2 + y**2, z**2)
    reduced = polysolve(
        equations, (x, y, z), presolve_max_terms=2, presolve_growth_factor=1, recognize=False
    )
    oracle = polysolve(equations, (x, y, z), presolve=False, method="rur", recognize=False)
    assert len(reduced.roots) == 1 and reduced.total_multiplicity == oracle.total_multiplicity == 4


def test_probe_never_relaxes_final_term_or_degree_limits():
    poly = sp.Poly(x**2 + y, (x, y, z))
    assert _bounded_substitution(poly, 0, y + z, (y, z), term_limit=2, degree_limit=10) is None
    assert _bounded_substitution(poly, 0, y + z, (y, z), term_limit=10, degree_limit=1) is None


def test_probe_work_budget_declines_before_full_expansion():
    poly = sp.Poly(x**1000 + y, (x, y, z))
    assert (
        _bounded_substitution(
            poly, 0, y + z, (y, z), term_limit=10, degree_limit=10000, work_limit=8
        )
        is None
    )


def test_probe_coefficient_growth_guard():
    poly = sp.Poly(x**1000, (x, y))
    assert (
        _bounded_substitution(
            poly, 0, sp.Integer(2) ** 100 * y, (y,), term_limit=10, degree_limit=10000
        )
        is None
    )


def test_zero_remainder_and_constant_pivot():
    poly = sp.Poly(x**20 - 1, x)
    assert _bounded_substitution(poly, 0, sp.S.One, (), term_limit=1, degree_limit=0) == 0


def test_probe_exact_against_expansion_with_partial_cancellation():
    expression = y + z
    row = x**3 - 3 * y * z * x - y**3 - z**3 + y**2
    poly = sp.Poly(row, x, y, z)
    actual = _bounded_substitution(poly, 0, expression, (y, z), term_limit=1, degree_limit=2)
    assert actual == sp.expand(row.subs(x, expression)) == y**2


def test_probe_huge_exponent_declines_without_deep_recursion():
    # SymPy's dense Poly constructor itself cannot represent this degree.
    # Feed its sparse term protocol to exercise our pre-expansion guard.
    from types import SimpleNamespace

    poly = SimpleNamespace(terms=lambda: [((2**100, 0), sp.S.One)])
    assert _bounded_substitution(poly, 0, y + 1, (y,), term_limit=2, degree_limit=2**101) is None


def test_probe_bounds_products_with_original_coefficients():
    poly = sp.Poly(sp.Integer(2) ** 8190 * x + 1, x, y)
    assert _bounded_substitution(poly, 0, 4 * y, (y,), term_limit=2, degree_limit=2) is None
