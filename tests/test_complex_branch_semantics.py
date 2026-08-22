import pytest
import sympy as sp

from algroots import algsolve

x = sp.symbols("x")


@pytest.mark.parametrize(
    ("power", "target"),
    [
        (sp.Rational(1, 3), 2),
        (sp.Rational(2, 3), 4),
        (sp.Rational(1, 4), 2),
        (sp.Rational(3, 5), 8),
        (sp.Rational(-1, 3), sp.Rational(1, 2)),
    ],
)
def test_rational_power_roots_satisfy_original_principal_expression(power, target):
    equation = x**power - target
    result = algsolve([equation], (x,), digits=55)
    for (value,) in result.roots:
        residual = sp.N(equation.subs(x, value), 40)
        assert abs(complex(residual)) < 1e-25


def test_negative_real_branch_rejected_for_two_thirds_power():
    result = algsolve(
        [x ** sp.Rational(2, 3) - 4],
        (x,),
        digits=50,
    )
    real_values = [complex(sp.N(root[0], 30)) for root in result.roots]
    assert any(abs(value - 8) < 1e-20 for value in real_values)
    assert not any(abs(value + 8) < 1e-20 for value in real_values)


def test_nested_principal_radicals_are_checked_at_original_expression():
    equation = sp.sqrt(1 + sp.sqrt(x)) - 2
    result = algsolve([equation], (x,), digits=50)
    assert len(result.roots) == 1
    assert abs(complex(sp.N(result.roots[0][0] - 9, 30))) < 1e-20
