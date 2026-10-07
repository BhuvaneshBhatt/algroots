"""Inspect cancellation-aware presolve and observed costs.

Guarantee: Exact substitution; timings are observations.
Run from the repository root: python examples/09_presolve_and_costs.py
"""

import sympy as sp

from algroots import polysolve


def main():
    x, y, z = sp.symbols("x y z")
    expression = y**2 + z**2
    equations = (x - expression, x**2 - expression**2 + y**2 - 2, z**2 - 1)
    result = polysolve(
        equations, (x, y, z), recognize=False, presolve_max_terms=2, presolve_growth_factor=1
    )
    assert len(result.roots) == 4 and result.variables == (x, y, z)
    costs = result.cost_diagnostics
    assert dict(costs.structural_estimates)["presolve_cancellation_acceptances"] >= 1
    print("Substitutions:", result.affine_substitutions)
    print("Observed phase seconds:", dict(costs.phase_seconds))
    print("Cost hints:", dict(costs.structural_estimates))


if __name__ == "__main__":
    main()
