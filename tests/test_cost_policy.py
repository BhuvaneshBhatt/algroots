import pytest
import sympy as sp

from algroots import polysolve
from algroots.cost_policy import graph_cost, ordered_variables, prefer_action
from algroots.quotient import QuotientAlgebra

x, y, z, w = sp.symbols("x y z w")


@pytest.mark.parametrize(
    "equations,variables",
    [
        ((x**2 - y, y**2 - z, z**2 - w, w**2 - 2), (x, y, z, w)),
        ((x**2 - 2, y**2 - x, z**2 - x, w**2 - y - z), (w, z, y, x)),
        ((x**2 - y, y**2 - z, z**2 - x - 1), (z, x, y)),
    ],
)
def test_ordering_is_deterministic_and_no_worse_graph_cost(equations, variables):
    baseline = ordered_variables(equations, variables, policy="frequency")
    actual = ordered_variables(equations, variables)
    assert set(actual) == set(variables)
    assert ordered_variables(equations, variables) == actual
    assert graph_cost(equations, actual) <= graph_cost(equations, baseline)
    result = polysolve(equations, variables, presolve=False, recognize=False, digits=30)
    oracle = polysolve(
        equations, variables, presolve=False, variable_order="input", recognize=False, digits=30
    )
    assert len(result.roots) == len(oracle.roots)
    for root in result.roots:
        assert (
            min(
                max(abs(complex(a - b)) for a, b in zip(root, other, strict=True))
                for other in oracle.roots
            )
            < 1e-20
        )


def test_nonreduced_cost_hint_skips_action(monkeypatch):
    import algroots.solver as solver

    monkeypatch.setattr(
        solver, "_solve_action_matrix", lambda *a, **k: pytest.fail("unnecessary action attempt")
    )
    result = polysolve((x**2, y**2), (x, y), presolve=False, recognize=False)
    assert result.method == "rational_univariate"
    assert result.total_multiplicity == 4 and result.geometric_solution_count == 1
    assert dict(result.cost_diagnostics.structural_estimates)["repeated_univariate_relation"]


def test_backend_hint_respects_explicit_budget():
    quotient = QuotientAlgebra.from_polynomials((x**3 - 2, y**3 - 3), (x, y))
    assert prefer_action(quotient, 9)[0]
    assert not prefer_action(quotient, 8)[0]
    assert prefer_action(quotient, 9)[1]["dense_action_entries"] == 162


def test_rur_first_retains_action_fallback(monkeypatch):
    import algroots.cost_policy as policy
    import algroots.solver as solver
    from algroots.errors import RationalUnivariateError

    monkeypatch.setattr(
        policy, "prefer_action", lambda *args: (False, {"preferred_backend": "rur"})
    )

    def fail(*args, **kwargs):
        raise RationalUnivariateError("forced RUR decline")

    monkeypatch.setattr(solver, "_solve_rur", fail)
    result = polysolve((x**2 - 2, y**2 - 3), (x, y), presolve=False, recognize=False)
    assert result.method == "action_matrix" and len(result.roots) == 4
