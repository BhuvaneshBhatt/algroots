import os

import pytest
import sympy as sp

from algroots import polysolve
from algroots.numerical import flint_available

x, y = sp.symbols("x y")


@pytest.mark.flint
def test_action_retries_after_nonseparating_linear_form(monkeypatch):
    if not flint_available():
        if os.environ.get("ALGROOTS_REQUIRE_FLINT") == "1":
            pytest.fail("python-flint required for separator retry test")
        pytest.skip("python-flint unavailable")

    import algroots.solver as solver

    monkeypatch.setattr(
        solver,
        "_separator_candidates",
        lambda count: ((1, 1), (1, 2)) if count == 2 else ((1,) * count,),
    )
    result = polysolve(
        [x**2 - 1, y**2 - 1],
        (x, y),
        method="action",
        digits=40,
        max_precision_digits=180,
    )
    assert len(result.roots) == 4
    assert result.quotient_dimension == 4
    assert result.separator_coeffs == (1, 2)


def test_action_result_preserves_each_completeness_stage():
    result = polysolve(
        [x**2 - 2, y**2 - 3],
        (x, y),
        method="action",
        digits=35,
    )
    assert result.quotient_dimension == 4
    assert result.standard_monomials is not None
    assert len(result.standard_monomials) == result.quotient_dimension
    assert len(result.roots) == result.quotient_dimension
    assert len(result.diagnostics) == result.quotient_dimension
    assert all(
        diag.final_relative_residual <= result.max_relative_residual for diag in result.diagnostics
    )


def test_completeness_is_not_inferred_from_small_residual_subset():
    polynomial = x**4 - 5 * x**2 + 4
    three_exact_roots = (-2, -1, 1)
    assert all(sp.expand(polynomial.subs(x, value)) == 0 for value in three_exact_roots)
    full = polysolve([polynomial], (x,), digits=35)
    assert len(full.roots) == 4
