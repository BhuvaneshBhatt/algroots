import pytest
import sympy as sp

from algroots import polysolve

x, y = sp.symbols("x y")


@pytest.mark.parametrize(
    "equations",
    [
        (
            10**30 * x**2 - 2 * 10**30,
            10**25 * y**2 - 3 * 10**25,
        ),
        (
            x**2 - sp.Rational(123456789012345678901, 9876543210987654321),
            y - sp.Rational(3141592653589793238, 10**18),
        ),
        (
            x - sp.sqrt(2) * y,
            y**2 - (2 + sp.sqrt(3)),
        ),
        (
            x**2 - (1 + sp.sqrt(2) + sp.sqrt(3)),
            y - x**2,
        ),
    ],
)
def test_exact_large_and_algebraic_coefficients(equations):
    result = polysolve(equations, (x, y), digits=55)
    assert result.roots
    assert result.max_relative_residual < sp.Float("1e-25")
