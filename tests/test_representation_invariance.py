import pytest
import sympy as sp

from algroots import polysolve

x, y = sp.symbols("x y")
BASE = (x**2 + y**2 - 1, x * y - sp.Rational(1, 4))


def _match_sets(left, right, tol=1e-10):
    assert len(left) == len(right)
    for root in left:
        assert any(
            max(abs(complex(sp.N(a - b, 18))) for a, b in zip(root, candidate, strict=True)) < tol
            for candidate in right
        )


@pytest.mark.parametrize(
    "variant",
    [
        lambda: (BASE, (x, y)),
        lambda: ((BASE[1], BASE[0]), (x, y)),
        lambda: ((sp.expand(BASE[0]), sp.factor(BASE[1])), (x, y)),
        lambda: ((sp.Eq(BASE[0], 0), sp.Eq(BASE[1], 0)), (x, y)),
        lambda: ((7 * BASE[0], -3 * BASE[1]), (x, y)),
        lambda: ((BASE[0] + 2 * BASE[1], BASE[1]), (x, y)),
    ],
)
def test_equivalent_equation_representations_preserve_roots(variant):
    reference = polysolve(BASE, (x, y), digits=35)
    equations, variables = variant()
    other = polysolve(equations, variables, digits=35)
    _match_sets(reference.roots, other.roots)


def test_variable_permutation_preserves_geometric_root_set():
    reference = polysolve(BASE, (x, y), digits=35)
    permuted = polysolve(BASE, (y, x), digits=35)
    swapped = tuple((root[1], root[0]) for root in permuted.roots)
    _match_sets(reference.roots, swapped)
