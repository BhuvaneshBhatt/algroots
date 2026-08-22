import os

import pytest
import sympy as sp

from algroots import polysolve
from algroots.numerical import flint_available

x, y = sp.symbols("x y")


def _require_flint():
    if flint_available():
        return
    if os.environ.get("ALGROOTS_REQUIRE_FLINT") == "1":
        pytest.fail("python-flint is required in CI but could not be imported")
    pytest.skip("python-flint unavailable on this developer machine")


@pytest.mark.flint
def test_action_backend_runs_through_acb_eigensolver():
    _require_flint()
    result = polysolve([x**2 - 2, y**2 - 3], (x, y), method="action", digits=45)
    assert result.method == "action_matrix"
    assert len(result.roots) == 4
    assert result.quotient_dimension == 4


@pytest.mark.flint
def test_shape_backend_runs_through_acb_polynomial_roots():
    _require_flint()
    result = polysolve([x - y**2, y**5 - y - 1], (x, y), method="shape", digits=45)
    assert result.method == "shape_position"
    assert len(result.roots) == 5


@pytest.mark.flint
def test_acb_action_roots_agree_with_shape_roots():
    _require_flint()
    equations = [x - y**2, y**3 - 2]
    action = polysolve(equations, (x, y), method="action", digits=45)
    shape = polysolve(equations, (x, y), method="shape", digits=45)
    assert len(action.roots) == len(shape.roots) == 3
    for root in action.roots:
        assert any(
            max(abs(complex(sp.N(a - b, 20))) for a, b in zip(root, candidate, strict=True)) < 1e-14
            for candidate in shape.roots
        )
