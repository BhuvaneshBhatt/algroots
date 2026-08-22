# algroots

`algroots` is an all-roots solver for supported **exact, zero-dimensional algebraic equation systems**. For the core polynomial and algebraic solvers, exact Gröbner/quotient-algebra structure determines the finite algebraic problem and FLINT/Arb arbitrary-precision numerics extracts and verifies the distinct complex roots. Exact algebraic recognition through [`algrecognize`](https://pypi.org/project/algrecognize/) is attempted by default after solving; pass `recognize=False` to disable it.

The high-level solver accepts polynomial equations, rational functions, rational powers, and nested radicals. So-called "algebraization" can introduce polynomial branches, so every projected candidate is checked against the **original algebraic equations** and retained domain constraints before it is returned.

> **Guarantee boundary.** A small numerical residual verifies that a returned approximation satisfies the equations to the requested numerical standard; residual verification alone is not a proof that no roots were missed. The core all-roots routes combine numerical extraction with exact zero-dimensional structural information. The separate continuation/monodromy subsystem is experimental: trace and capture-recapture stopping provide numerical/statistical evidence and are not exact completeness certificates.

`algroots` also exposes exact quotient-algebra tools for rational polynomial systems: rational univariate representations (RURs) and exact border bases with multiplication-matrix commutation checks. These complement the numerical shape/action/triangular/RUR/total-degree-homotopy and monodromy routes rather than replacing them.

For exact rational- or algebraic-coefficient polynomial systems, `polysolve(..., method="rur")` uses a native number-field RUR as a root-extraction backend and then applies the same numerical verification/result pipeline.

## Installation

```bash
python -m pip install algroots
```

For development:

```bash
python -m pip install -e ".[test]"
python -m pytest
```

## Quick example

<!-- algroots: execute -->
```python
import sympy as sp
from algroots import algsolve

x = sp.symbols("x")

result = algsolve(
    [sp.sqrt(x) - (x - 2)],
    (x,),
    digits=60,
)

print(result.roots)
# approximately ((4.0,),)
```

Squaring the equation also produces `x = 1`, but `x = 1` violates the original principal-square-root equation. `algroots` rejects that extraneous branch.

For input that is already polynomial, use:

<!-- algroots: execute -->
```python
from algroots import polysolve

x, y = sp.symbols("x y")
result = polysolve(
    [x - y**2, y**3 - 2],
    (x, y),
    digits=60,
)
```

By default, successful exact reconstructions are available as `result.recognized_roots`. The numerical roots remain in `result.roots`; if automatic recognition cannot certify a low-complexity relation within the default degree-8 budget, the numerical solve still succeeds and `result.recognition_error` records the recognition failure. Use `recognize=False` to skip this post-processing step.

## Documentation

The documentation is organized from user-facing basics to implementation details:

1. [Quick Start](https://github.com/BhuvaneshBhatt/algroots/blob/main/docs/quick-start.md)
2. [Supported Problems](https://github.com/BhuvaneshBhatt/algroots/blob/main/docs/supported-problems.md)
3. [Feature and Support Matrix](https://github.com/BhuvaneshBhatt/algroots/blob/main/docs/feature-support-matrix.md)
4. [Guarantees and Result Semantics](https://github.com/BhuvaneshBhatt/algroots/blob/main/docs/guarantees-and-result-semantics.md)
5. [Choosing a Backend](https://github.com/BhuvaneshBhatt/algroots/blob/main/docs/choosing-a-backend.md)
6. [Algorithms](https://github.com/BhuvaneshBhatt/algroots/blob/main/docs/algorithms.md)
7. [Numerical Reliability](https://github.com/BhuvaneshBhatt/algroots/blob/main/docs/numerical-reliability.md)
8. [API Reference](https://github.com/BhuvaneshBhatt/algroots/blob/main/docs/api.md)

For a compact capability overview, see the [Feature and Support Matrix](https://github.com/BhuvaneshBhatt/algroots/blob/main/docs/feature-support-matrix.md). For deeper topics, see:
- [Total-Degree Homotopy Continuation](https://github.com/BhuvaneshBhatt/algroots/blob/main/docs/homotopy-continuation.md),
- [Exact Border Bases](https://github.com/BhuvaneshBhatt/algroots/blob/main/docs/exact-border-bases.md), 
- [Multiplicity and Singular Roots](https://github.com/BhuvaneshBhatt/algroots/blob/main/docs/multiplicity-and-singular-roots.md), 
- [Precision and Conditioning](https://github.com/BhuvaneshBhatt/algroots/blob/main/docs/precision-and-conditioning.md), 
- [Branch Semantics](https://github.com/BhuvaneshBhatt/algroots/blob/main/docs/branch-semantics.md), 
- [Completeness versus Verification](https://github.com/BhuvaneshBhatt/algroots/blob/main/docs/completeness-versus-verification.md), 
- [Exact Recognition](https://github.com/BhuvaneshBhatt/algroots/blob/main/docs/exact-recognition.md), 
- [Performance Model](https://github.com/BhuvaneshBhatt/algroots/blob/main/docs/performance-model.md), and 
- [Troubleshooting](https://github.com/BhuvaneshBhatt/algroots/blob/main/docs/troubleshooting.md).

The [Examples Gallery](https://github.com/BhuvaneshBhatt/algroots/blob/main/docs/examples-gallery.md) provides practical systems, while the [End-to-End Worked Example](https://github.com/BhuvaneshBhatt/algroots/blob/main/docs/end-to-end-example.md) follows one algebraic system from input through exact algebraization, quotient structure, FLINT/Arb numerical extraction, branch filtering, and default best-effort exact certification.

## Design in one diagram

```text
                     exact/algebraic input
                              │
                 algebraize when necessary
                              │
              ┌───────────────┴───────────────┐
              │                               │
     quotient-algebra routes          continuation routes
 shape / action / triangular / RUR   total-degree homotopy
              │                       monodromy (seeded)
              └───────────────┬───────────────┘
                              │
                 numerical refinement +
                  scaled residual checks
                              │
              original branch/domain filtering
                              │
           best-effort exact recognition (default)
```

Exact RUR and border-basis objects are also available directly as quotient-algebra representations; they are not merely numerical backends.

## Total-degree homotopy backend

`polysolve(..., method="homotopy")` is an explicit total-degree continuation backend. For equation degrees $d_1,\ldots,d_n$, it tracks the $\prod_i d_i$ roots of the start system $x_i^{d_i}-1=0$. It retries a short deterministic sequence of exact complex gamma values when a path set is singular or numerically unusable, and can track independent paths in worker processes with one symbolic compilation per worker. `method="auto"` does not select it.

The scope for now is deliberately conservative: regular square polynomial systems whose required total-degree paths all end at distinct finite nonsingular roots. If a path fails, tends toward infinity, or coalesces at a singular endpoint, `HomotopySolveError` is raised rather than returning a potentially incomplete set. Singular endgames and projective path tracking are future work. See [Total-Degree Homotopy Continuation](https://github.com/BhuvaneshBhatt/algroots/blob/main/docs/homotopy-continuation.md) for the start system, path-count semantics, gamma retries, parallelism, and endpoint diagnostics.

## Experimental continuation and monodromy

The continuation core serves two uses: an explicit `polysolve(..., method="homotopy")` total-degree backend and the separate experimental monodromy subsystem. Its high-level `discover_monodromy_orbit` entry point accepts the same exact algebraic equation subset as `algsolve`: polynomial equations, rational functions, rational powers, nested radicals, and exact algebraic coefficients. Algebraic seeds are lifted to the augmented polynomial system on the principal sheet; tracked endpoints are projected back to the original variables and rechecked against the original branch/domain semantics before they are returned. The lower-level `closed_additive_loop` and `track_loop` APIs remain polynomial-system primitives. There is currently no `polysolve(..., method="monodromy")` route and no automatic monodromy seed generation.

Monodromy can stop from an independently supplied exact root count, a second-order trace test, or a capture-recapture estimate. Only the independently exact count supplies an exact-count basis; trace and statistical stopping remain numerical/statistical evidence. Endpoint matching also refuses ambiguous tolerance matches rather than assigning an arbitrary permutation. Discovered roots are recognized automatically by default using each root's own verification precision; pass `recognize=False` to disable this. `recognize_system_roots` remains available for manual recognition with custom bounds. Recognition does not strengthen the orbit's completeness evidence. See [Continuation and Monodromy](https://github.com/BhuvaneshBhatt/algroots/blob/main/docs/continuation-and-monodromy.md).

## Scope

`algroots` is intended for finite solution sets of algebraic equalities. Generic transcendental equations, inequalities, and positive-dimensional varieties are outside its current scope. Multiple or non-radical roots might require the triangular fallback; multiplicity reporting is not yet a first-class feature.

## License

GPL-3.0-only.
