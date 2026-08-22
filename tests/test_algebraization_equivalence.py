import pytest
import sympy as sp

from algroots import algebraize_system, algsolve, polysolve

x, y = sp.symbols("x y")


def _close_tuple(left, right, tol=1e-12):
    return max(abs(complex(sp.N(a - b, 20))) for a, b in zip(left, right, strict=True)) < tol


@pytest.mark.parametrize(
    ("equations", "variables"),
    [
        ((sp.sqrt(x) - (x - 2),), (x,)),
        ((x ** sp.Rational(2, 3) - 4,), (x,)),
        ((sp.sqrt(x + y) - x, y - 2), (x, y)),
        ((sp.Pow(x, sp.Rational(-1, 2)) - 2,), (x,)),
    ],
)
def test_high_level_roots_are_subset_of_projected_algebraized_candidates(equations, variables):
    algebraized = algebraize_system(equations, variables)
    polynomial = polysolve(
        algebraized.polynomial_equations,
        algebraized.augmented_variables,
        digits=45,
    )
    high = algsolve(equations, variables, digits=45)

    projected = tuple(root[: len(variables)] for root in polynomial.roots)
    for root in high.roots:
        assert any(_close_tuple(root, candidate) for candidate in projected)


def test_polynomial_input_high_and_low_level_agree_exactly_as_root_sets():
    equations = (x**2 + y**2 - 1, x * y - sp.Rational(1, 4))
    high = algsolve(equations, (x, y), digits=40)
    low = polysolve(equations, (x, y), digits=40)
    assert len(high.roots) == len(low.roots)
    for root in high.roots:
        assert any(_close_tuple(root, other) for other in low.roots)
