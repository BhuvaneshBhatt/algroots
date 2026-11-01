# algroots

`algroots` provides exact algebraic computation for polynomial systems, rational functions, finite quotient algebras, function fields, functional decomposition, and algebraic curves. It combines exact Gröbner and quotient-algebra structure with arbitrary-precision FLINT/Arb root extraction. It also provides exact rational univariate representations, border bases, isolated-root certificates and targeted deflation.

Use `polysolve` for polynomials and `algsolve` for rational functions, principal rational powers and nested radicals. For now, coefficients must be exact: use `sp.Rational(1, 10)`, rather than `0.1`. `algroots` does not support positive-dimensional varieties, inequalities, noisy coefficients or transcendental equations.

## Public API

Use `polysolve` or `algsolve` from the package root. Manual certification, quotient algebra, RURs, border bases and continuation tools use their public submodules. See the [namespace map](https://github.com/BhuvaneshBhatt/algroots/blob/main/docs/api.md#root-and-specialized-namespaces) for the full API.

## Install and run

Python 3.11 or newer is required. The dependencies include SymPy, mpmath, python-flint and algrecognize.

```bash
python -m pip install algroots
```

## Solve a polynomial system

<!-- algroots: execute -->
```python
import sympy as sp
from algroots import polysolve

x, y = sp.symbols("x y")
result = polysolve((x**2 + y**2 - 3, x * y - 1), (x, y), digits=40, recognize=False)
assert len(result.roots) == result.geometric_solution_count == 4
print(result.roots)
print(result.completeness)
```

In the result each tuple follows the supplied variable order. `.roots` stores distinct geometric points with aligned per-root metadata. Use `root_mode="with_multiplicity"` for repeated-root iteration and `.output_roots`; `.multiplicities` reports established local multiplicities. Results use canonical coordinate ordering, with `.ordering` distinguishing exact ordering from numerical fallback. `root_order="required"` requires exact ordering within the proof budgets. `digits` requests output precision; working precision may increase, and `verification_digits` controls the numerical residual standard. A small residual by itself is not an exact root or completeness proof.

The automatic dispatch applies guarded exact presolve, deterministic ordering and a grevlex (graded reverse lexicographic) Gröbner basis. Action and RUR share quotient work; conversion to lexicographic with FGLM occurs only when needed. Cost hints preserve exact fallback checks. See [Choosing a Backend](https://github.com/BhuvaneshBhatt/algroots/blob/main/docs/choosing-a-backend.md).

## Preserve branches and poles

<!-- algroots: execute -->
```python
import sympy as sp
from algroots import algsolve

x = sp.Symbol("x")
result = algsolve((sp.sqrt(x) - (x - 2),), (x,), digits=40, recognize=False)
assert len(result.roots) == 1
assert abs(result.roots[0][0] - 4) < sp.Rational(1, 10**30)
```

Squaring also produces `x=1`, which fails the original principal-square-root equation. `algsolve` filters such candidates and retains denominator restrictions. Obviously, it cannot recover domain information already removed by SymPy before the input is supplied.

## Certify roots and inspect multiplicities

<!-- algroots: execute -->
```python
import sympy as sp
from algroots import polysolve

x = sp.Symbol("x")
result = polysolve((x**2,), (x,), recognize=False, certify="required")
assert len(result.roots) == 1
assert result.total_multiplicity == 2 and not result.is_radical
certificate = result.root_certifications[0].certificate
assert certificate.multiplicity == 2 and certificate.verify()
```

`certify="auto"` records best-effort proof attempts; `certify="required"` raises when a proof can't be established. Automatic boxes currently require rational coefficients and a globally finite ideal. Exact-point certification also supports algebraic coefficients.

Recognition is a different operation. By default, solvers attempt exact coordinate recognition through `algrecognize`; successful jointly certified tuples appear in `recognized_roots`. Recognition failure leaves the numerical result available. `recognize=False` disables that post-processing. See [Certification Workflow](https://github.com/BhuvaneshBhatt/algroots/blob/main/docs/certification-workflow.md) and [Reading Results](https://github.com/BhuvaneshBhatt/algroots/blob/main/docs/reading-results.md).

| Evidence | What it establishes |
|---|---|
| Residual diagnostics | Numerical consistency at the verification standard |
| `completeness.status == "conditional"` | Exact finite count/structure, with numerical extraction assumptions |
| Exact isolated-root certificate | The point satisfies the original equations, is isolated, and has the reported rank/multiplicity |
| Certified bounded recovery | Distinct exact endpoint proofs account for the exact finite-root count |
| Numerical paths / monodromy stopping | Numerical or statistical evidence; no certified path or infinity accounting |

## Exact algebra and advanced solving

| Task | API | Guide |
|---|---|---|
| Quotient basis, normal forms, matrices, traces | `QuotientAlgebra` | [Quotient tools](https://github.com/BhuvaneshBhatt/algroots/blob/main/docs/quotient-tools.md) |
| Exact RUR and exact algebraic points | `compute_rational_univariate_representation`, `solve_rur_representation` | [RUR](https://github.com/BhuvaneshBhatt/algroots/blob/main/docs/rational-univariate-representation.md) |
| Exact rational border bases | `compute_border_basis`, `compute_border_basis_linear` | [Border bases](https://github.com/BhuvaneshBhatt/algroots/blob/main/docs/exact-border-bases.md) |
| Singular-root proof and exact local deflation | `certify_isolated_root`, `deflate_isolated_root` | [Singular certification](https://github.com/BhuvaneshBhatt/algroots/blob/main/docs/singular-certification-and-charts.md) |
| Bounded endgame/deflation/chart recovery | `polysolve(method="homotopy", homotopy_recovery=True)` | [Recovery](https://github.com/BhuvaneshBhatt/algroots/blob/main/docs/recovery-workflow.md) |
| Numerical projective tracking | `projective_homotopy`, `track_projective_path` | [Chart switching](https://github.com/BhuvaneshBhatt/algroots/blob/main/docs/singular-certification-and-charts.md) |
| Seeded experimental monodromy | `discover_monodromy_orbit` | [Monodromy](https://github.com/BhuvaneshBhatt/algroots/blob/main/docs/continuation-and-monodromy.md) |


## Functional decomposition, fields, and curves

Algebraic methods also address structural questions for which enumerating roots is not the goal:

| Question | Expert API | Guide |
|---|---|---|
| Can a rational function be written as a composition of simpler rational functions? | `algroots.functional_decomposition.rational_decompose`, `rational_decomposition_chain` | [Functional decomposition](https://github.com/BhuvaneshBhatt/algroots/blob/main/docs/functional-decomposition.md) |
| Does an algebraic relation admit a compositional factorization, possibly over a coefficient extension? | `algroots.functional_decomposition.algebraic_decompose`, `discover_algebraic_factors` | [Functional decomposition](https://github.com/BhuvaneshBhatt/algroots/blob/main/docs/functional-decomposition.md) |
| What subfield is generated by a collection of algebraic or rational elements? | `QuotientAlgebra.subfield`, `algroots.function_fields.coefficient_field`, `luroth_generator` | [Generated subfields](https://github.com/BhuvaneshBhatt/algroots/blob/main/docs/function-fields.md) |
| What is the geometric genus of a plane curve, and does it have a rational parametrization? | `algroots.curves.PlaneCurve` | [Curve geometry](https://github.com/BhuvaneshBhatt/algroots/blob/main/docs/curves.md) |
| How can a finite quotient algebra be decomposed into local factors or reduced? | `algroots.finite_decomposition.primary_decomposition`, `reduced_algebra`, `nilradical_basis` | [Finite algebras](https://github.com/BhuvaneshBhatt/algroots/blob/main/docs/finite-algebras.md) |
| How can one construct a finite coefficient field and factor polynomials over it? | `algroots.finite_fields.finite_field`, `factor_univariate` | [Finite algebras](https://github.com/BhuvaneshBhatt/algroots/blob/main/docs/finite-algebras.md) |

These specialized APIs use exact coefficients and explicit search limits. Successful decompositions carry identities that can be checked. Plane-curve operations are distinct from solving a positive-dimensional system by listing all its points.

Ordinary `method="homotopy"` targets regular square systems whose required Bézout paths reach finite nonsingular endpoints. Recovery is opt-in, sequential, rational-coefficient and bounded. It adds a finite quotient for proof accounting, preserves successful endpoint proofs across retries, and reports unresolved paths. Neither of those modes certifies numerical paths or roots at infinity. Homotopy is not selected by `auto`.

## Numerical precision and stopping

Numerical precision and integer budgets follow a common contract: nonzero coordinate components are preserved, root comparisons are relative to coordinate scale, and univariate Arb retries respect the working precision ceiling. Monodromy curvature stopping is a heuristic and gives no completeness evidence. See [precision and conditioning](https://github.com/BhuvaneshBhatt/algroots/blob/main/docs/precision-and-conditioning.md) and [continuation and monodromy](https://github.com/BhuvaneshBhatt/algroots/blob/main/docs/continuation-and-monodromy.md).

## Examples and documentation

The [examples catalog](https://github.com/BhuvaneshBhatt/algroots/blob/main/examples/README.md) contains 16 scripts covering solving, branch filtering, recognition, quotient caches, RUR, algebraic fields, multiplicities, clusters, diagnostics, border bases, deflation, recovery, charts, monodromy and failures.

```bash
python examples/run_all.py
python examples/run_all.py 07_multiplicity_and_boxes.py 15_limits_and_failures.py
```

Start with [Quick Start](https://github.com/BhuvaneshBhatt/algroots/blob/main/docs/quick-start.md), [Supported Problems](https://github.com/BhuvaneshBhatt/algroots/blob/main/docs/supported-problems.md), [Support Matrix](https://github.com/BhuvaneshBhatt/algroots/blob/main/docs/feature-support-matrix.md) and [Reading Results](https://github.com/BhuvaneshBhatt/algroots/blob/main/docs/reading-results.md). The [documentation index](https://github.com/BhuvaneshBhatt/algroots/blob/main/docs/index.md) links every topic, including the [API reference](https://github.com/BhuvaneshBhatt/algroots/blob/main/docs/api.md), [limitations](https://github.com/BhuvaneshBhatt/algroots/blob/main/docs/limitations.md) and [troubleshooting](https://github.com/BhuvaneshBhatt/algroots/blob/main/docs/troubleshooting.md).

## Development

```bash
python -m pytest
python -m ruff check .
python -m ruff format --check .
```

Tests execute marked documentation and every catalog example, check navigation and API defaults, compare exact algebra against independent oracles, and exercise adversarial/resource limits. Benchmarks retain exact correctness checks but do not enforce wall-clock thresholds. See [Testing and Validation](https://github.com/BhuvaneshBhatt/algroots/blob/main/docs/testing-and-validation.md) and [Contributing](https://github.com/BhuvaneshBhatt/algroots/blob/main/CONTRIBUTING.md).

## License

The package is distributed under the [GPL-3.0 license](https://github.com/BhuvaneshBhatt/algroots/blob/main/LICENSE).
