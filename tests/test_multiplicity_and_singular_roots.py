import pytest
import sympy as sp

from algroots import polysolve
from algroots.errors import ActionMatrixError
from algroots.quotient import QuotientAlgebra

x, y = sp.symbols("x y")


@pytest.mark.parametrize(
    ("equations", "variables", "expected_distinct", "expected_dimension"),
    [
        ((x**2,), (x,), 1, 2),
        ((x**3,), (x,), 1, 3),
        ((x**2, y**2), (x, y), 1, 4),
        (((x - 1) ** 2 * (x + 2),), (x,), 2, 3),
    ],
)
def test_nonradical_distinct_roots_differ_from_quotient_dimension(
    equations, variables, expected_distinct, expected_dimension
):
    auto = polysolve(equations, variables, method="auto", digits=35)
    assert len(auto.roots) == expected_distinct

    basis = sp.groebner(equations, *variables, order="lex")
    assert basis.is_zero_dimensional
    quotient_basis = QuotientAlgebra.from_groebner_basis(
        basis, variables, max_dimension=64
    ).standard_exponents
    assert len(quotient_basis) == expected_dimension
    assert expected_dimension >= expected_distinct


def test_action_backend_rejects_defective_nonradical_ideal():
    with pytest.raises(ActionMatrixError):
        polysolve([x**2, y], (x, y), method="action", digits=35)


def test_repeated_root_is_not_returned_multiple_times():
    result = polysolve([(x - 3) ** 5], (x,), digits=35)
    assert len(result.roots) == 1
    assert abs(complex(sp.N(result.roots[0][0] - 3, 20))) < 1e-15


def test_singular_jacobian_at_repeated_root():
    equations = (x**2, y**2)
    jacobian = sp.Matrix(equations).jacobian((x, y))
    assert jacobian.subs({x: 0, y: 0}).rank() == 0
