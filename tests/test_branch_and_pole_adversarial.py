import pytest
import sympy as sp

from algroots import algsolve

x, y = sp.symbols("x y")


@pytest.mark.parametrize(
    ("equation", "expected"),
    [
        (sp.sqrt(x) - (x - 2), (4,)),
        (x ** sp.Rational(2, 3) - 4, (8,)),
        (x ** sp.Rational(-1, 2) - sp.Rational(1, 2), (4,)),
        (sp.sqrt(sp.sqrt(x)) - 2, (16,)),
    ],
)
def test_principal_branch_adversarial_cases(equation, expected):
    result = algsolve([equation], (x,), digits=35, verification_digits=20)
    assert len(result.roots) == len(expected)
    values = [complex(sp.N(root[0], 20)) for root in result.roots]
    for wanted in expected:
        assert any(abs(value - wanted) < 1e-15 for value in values)


def test_cleared_denominator_does_not_restore_a_pole():
    # Construct unevaluated so x=1 remains an excluded point.
    expr = sp.Mul(x**2 - 1, sp.Pow(x - 1, -1, evaluate=False), evaluate=False)
    result = algsolve([expr], (x,), digits=35)
    assert len(result.roots) == 1
    assert abs(complex(sp.N(result.roots[0][0] + 1, 20))) < 1e-15


def test_pole_component_is_removed_before_zero_dimensionality_check():
    denominator = sp.Mul(x, y - 1, evaluate=False)
    rational = sp.Mul(x * (y - 2), sp.Pow(denominator, -1, evaluate=False), evaluate=False)
    # Clearing the denominator naively introduces x=0 as a whole component.
    # Saturation removes that component, leaving y=2 together with x=1.
    result = algsolve([rational, x - 1], (x, y), digits=30)
    assert len(result.roots) == 1
    root = result.roots[0]
    assert abs(complex(sp.N(root[0] - 1, 18))) < 1e-13
    assert abs(complex(sp.N(root[1] - 2, 18))) < 1e-13


def test_coupled_radical_system_filters_wrong_square_root_branch():
    result = algsolve([sp.sqrt(x + y) - x, y - 2], (x, y), digits=35)
    # sqrt(x+2)=x -> x^2-x-2=0 gives x=2,-1 after squaring; only x=2 has principal sqrt=x.
    assert len(result.roots) == 1
    assert abs(complex(sp.N(result.roots[0][0] - 2, 18))) < 1e-13
