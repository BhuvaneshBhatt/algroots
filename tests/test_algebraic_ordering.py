import pytest
import sympy as sp

from algroots import polysolve
from algroots.root_output import _exact_sign


@pytest.mark.parametrize(
    "value, expected",
    [
        (3 * sp.real_root(2, 3) ** 2 / 2, 1),
        (-3 * sp.real_root(2, 3) ** 2 / 2, -1),
        (sp.sqrt(2) - sp.Rational(7, 5), 1),
        (sp.Rational(7, 5) - sp.sqrt(2), -1),
    ],
)
def test_scaled_root_sign(value, expected):
    assert _exact_sign(value, 64) == expected


@pytest.mark.recognition
def test_recognized_root_order():
    x, y = sp.symbols("x y")
    result = polysolve([x - y**2, y**3 - 2], (x, y), digits=60)
    assert len(result.roots) == 3
    assert result.recognition_attempted
    for a, b in result.roots:
        assert abs(complex(a - b * b)) < 1e-50
        assert abs(complex(b**3 - 2)) < 1e-50
