# Solver settings

Start with `polysolve(equations, variables)` for exact polynomial equations and `algsolve` for supported rational functions and principal rational powers. Both require finitely many complex solutions. The defaults cover ordinary calls; use reusable settings when you need more precision, tighter resource bounds, or supervised execution.

<!-- algroots: execute -->
```python
import sympy as sp
from algroots import PrecisionOptions, SolverLimits, SolverOptions, polysolve

x, y = sp.symbols("x y")
options = SolverOptions(
    precision=PrecisionOptions(digits=70, recognize=False),
    limits=SolverLimits(max_action_dimension=16, max_returned_roots=8),
)
result = polysolve([x**2 - y, y**2 - 1], [x, y], options=options)
assert len(result.roots) == 4
assert result.quotient_dimension == 4
```

`PrecisionOptions` controls working digits, verification digits, guard digits, precision ceilings, and exact recognition. `SolverLimits` controls iteration, solution, action-matrix, homotopy-path, returned-root, and ordering-refinement bounds. Both validate their values when constructed. Settings are immutable and reusable across calls to either solver.

Flat keywords such as `digits=70` remain available for individual calls. When `options=` is supplied, keep the grouped settings in that object: a nondefault flat precision or resource keyword raises an error instead of overwriting a grouped setting. Algorithm choices such as `method`, `groebner_backend`, `presolve`, and output mode remain explicit call keywords. Native Gröbner operation limits use `groebner_limits`; they are distinct from solver limits.

For bounded endpoint recovery, pass `SolverOptions(recovery=HomotopyRecoveryOptions(...))` to `polysolve(..., method="homotopy")`. This enables recovery with the supplied budgets. Do not also set flat recovery keywords. `algsolve` does not expose this recovery setting because branch projection has a separate contract. See [Recovery workflow](recovery-workflow.md) for the mathematical guarantees and a complete example.

Set `SolverOptions(execution=ExecutionLimits(timeout=30))` to supervise the entire solve, including presolve, basis verification, root extraction, and recognition. Use a guarded script as described in [Supervised execution](supervised-execution.md). An operation-budget refusal or deadline is not evidence that the equations have no roots.

See the generated [API reference](api.md#backend-discovery-and-solver-settings) for every setting and default. For basis selection and verification costs, use the [Gröbner guide](groebner-bases.md#choose-a-backend).
