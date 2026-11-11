"""Reuse solver settings and supervise a complete verified calculation."""

import sympy as sp

from algroots import (
    ExecutionLimits,
    PrecisionOptions,
    SolverLimits,
    SolverOptions,
    groebner_basis,
    polysolve,
)


def main():
    x, y = sp.symbols("x y")
    equations = [x**2 - y, y**2 - 1]
    settings = SolverOptions(
        precision=PrecisionOptions(digits=40, recognize=False),
        limits=SolverLimits(max_action_dimension=16),
        execution=ExecutionLimits(timeout=30),
    )
    result = polysolve(equations, (x, y), options=settings, groebner_backend="sympy")
    basis = groebner_basis(
        equations, (x, y), backend="sympy", execution=ExecutionLimits(timeout=30)
    )
    assert len(result.roots) == 4
    assert basis.reduce(x**4 - 1)[1] == 0
    print("Verified basis:", basis)
    print("Number of roots:", len(result.roots))


if __name__ == "__main__":
    main()
