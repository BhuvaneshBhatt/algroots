import pytest
import sympy as sp

from algroots import polysolve
from algroots.cost_policy import prefer_action
from algroots.quotient import QuotientAlgebra

x, y = sp.symbols("x y")


def test_multivariate_repeated_factor_avoids_failed_action(monkeypatch):
    import algroots.solver as solver

    equations = ((x - y) ** 2, y**2 - 1)
    monkeypatch.setattr(
        solver, "_solve_action_matrix", lambda *a, **k: pytest.fail("wasted action solve")
    )
    result = polysolve(equations, (x, y), presolve=False, recognize=False)
    assert result.method == "rational_univariate"
    assert result.total_multiplicity == 4 and len(result.roots) == 2
    assert dict(result.cost_diagnostics.structural_estimates)["repeated_input_factor"]
    assert result.cost_diagnostics.policy == "calibrated-v2"


def test_repeated_factor_is_cost_hint_not_radicality_proof():
    equations = (x**2, x, y**2 - 1)
    q = QuotientAlgebra.from_polynomials(equations, (x, y))
    assert not prefer_action(q, 64, equations)[0]
    result = polysolve(equations, (x, y), presolve=False, recognize=False)
    assert result.is_radical and result.total_multiplicity == 2


@pytest.mark.parametrize(
    "equations",
    [(x**2 - sp.sqrt(2), y**2 - 3), (x**3 - 10**30 * y, y**2 - 2), (x**2 + y**2 - 3, x * y - 1)],
)
def test_calibrated_regular_families_keep_action(equations):
    result = polysolve(equations, (x, y), presolve=False, recognize=False, digits=35)
    assert result.method == "action_matrix"
    assert dict(result.cost_diagnostics.structural_estimates)["dispatch_reason"] == "small_action"
    oracle = polysolve(equations, (x, y), method="rur", presolve=False, recognize=False, digits=35)
    assert len(result.roots) == len(oracle.roots)
    for point in result.roots:
        assert (
            min(
                max(abs(complex(a - b)) for a, b in zip(point, other, strict=True))
                for other in oracle.roots
            )
            < 1e-20
        )


def test_factor_hint_budget_does_not_factor_large_inputs(monkeypatch):
    equations = (x**9 + y, y**2 - 2)
    q = QuotientAlgebra.from_polynomials(equations, (x, y))
    original = sp.Poly.sqf_list

    def guarded(self, *args, **kwargs):
        assert self.total_degree() <= 8
        return original(self, *args, **kwargs)

    monkeypatch.setattr(sp.Poly, "sqf_list", guarded)
    assert prefer_action(q, 64, equations)[0]
