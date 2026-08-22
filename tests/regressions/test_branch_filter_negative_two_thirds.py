import sympy as sp

from algroots import algsolve

x = sp.symbols("x")


def test_negative_eight_is_not_accepted_for_principal_two_thirds_power():
    result = algsolve([x ** sp.Rational(2, 3) - 4], (x,), digits=50)
    values = [complex(sp.N(root[0], 30)) for root in result.roots]
    assert any(abs(value - 8) < 1e-20 for value in values)
    assert not any(abs(value + 8) < 1e-20 for value in values)
