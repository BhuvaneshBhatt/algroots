import sympy as sp

from algroots import algsolve

x, y = sp.symbols("x y")


def test_pole_component_saturation_removes_spurious_positive_dimensional_component():
    denominator = sp.Mul(x, y - 1, evaluate=False)
    rational = sp.Mul(
        x * (y - 2),
        sp.Pow(denominator, -1, evaluate=False),
        evaluate=False,
    )
    result = algsolve([rational, x - 1], (x, y), digits=40)
    assert len(result.roots) == 1
    root = result.roots[0]
    assert abs(complex(sp.N(root[0] - 1, 20))) < 1e-15
    assert abs(complex(sp.N(root[1] - 2, 20))) < 1e-15
