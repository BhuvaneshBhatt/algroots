import os

import pytest
import sympy as sp

from algroots import polysolve
from algroots.numerical import flint_available

x, y = sp.symbols("x y")


@pytest.mark.flint
def test_colliding_separator_is_not_accepted(monkeypatch):
    if not flint_available():
        if os.environ.get("ALGROOTS_REQUIRE_FLINT") == "1":
            pytest.fail("python-flint required in CI")
        pytest.skip("python-flint unavailable")
    import algroots.solver as solver

    monkeypatch.setattr(solver, "_separator_candidates", lambda count: ((1, 1), (1, 2)))
    result = polysolve([x**2 - 1, y**2 - 1], (x, y), method="action", digits=40)
    assert result.separator_coeffs == (1, 2)
    assert len(result.roots) == 4
