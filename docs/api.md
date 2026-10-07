# API Reference

## Root and specialized namespaces

The root exports only `polysolve`, `algsolve`, `PolynomialSystemRoots`, `AlgebraicSystemRoots`, `CompletenessEvidence`, `HomotopyRecoveryOptions`, `PolynomialSystemError`, `PolynomialSystemInputError`, `NotZeroDimensionalError`, `SystemSolveLimitError`, `__version__`.

Advanced imports use the namespaces below. Types remain public for annotations and result inspection.

| Namespace | Public names |
|---|---|
| `algroots.algebraization` | `algebraize_system`, `AlgebraizedSystem` |
| `algroots.border_basis` | `BorderBasisDiagnostics`, `BorderBasisResult`, `compute_border_basis`, `compute_border_basis_linear` |
| `algroots.certification` | `RootCertificationAttempt`, `certify_numerical_roots`, `IsolatedRootCertificate`, `RationalComplexBox`, `certify_isolated_root`, `certify_root_box` |
| `algroots.continuation` | `HomotopySystem`, `SympyHomotopy`, `PathTrackerOptions`, `PathStep`, `PathResult`, `track_path` |
| `algroots.deflation` | `DeflatedRefinement`, `DeflationResult`, `DeflationStage`, `deflate_isolated_root` |
| `algroots.endgames` | `EndgameResult`, `ProjectiveHomotopy`, `cauchy_endgame`, `projective_homotopy` |
| `algroots.errors` | `RootCertificationError`, `PathTrackingError`, `PathStepError`, `ExactCertificationError`, `QuotientAlgebraError`, `ActionMatrixError`, `HomotopySolveError`, `ShapePositionError`, `TriangularSolveError`, `NumericalRootError`, `RationalUnivariateError`, `BorderBasisError` |
| `algroots.monodromy` | `MonodromyLoop`, `MonodromyPermutation`, `MonodromyOrbitResult`, `MonodromyRootInfo`, `closed_additive_loop`, `track_loop`, `monodromy_permutation`, `discover_monodromy_orbit` |
| `algroots.monodromy_stopping` | `CaptureRecaptureEstimate`, `capture_recapture_estimate`, `second_order_trace_test` |
| `algroots.projective_tracking` | `ChartSwitch`, `ProjectivePathResult`, `track_projective_path` |
| `algroots.quotient` | `QuotientAlgebra`, `SeparatingElement` |
| `algroots.rational_univariate` | `RationalUnivariateRepresentation`, `RationalUnivariatePoint`, `compute_rational_univariate_representation`, `solve_zero_dimensional_system_with_rur`, `solve_rur_representation`, `solve_rur_points` |
| `algroots.recognition` | `recognize_system_roots`, `RecognizedSystemRoot` |
| `algroots.recovery` | `PathRecoveryRecord` |
| `algroots.root_output` | `RootOrderingEvidence` |
| `algroots.solver` | `SolveCostDiagnostics`, `RootDiagnostics` |


This page covers the high-level root API and the specialized public namespaces.

## Public functions

| Function | Purpose | Returns | Guarantee level |
|---|---|---|---|
| `algsolve` | Solve a supported exact zero-dimensional algebraic system, including rational functions and rational powers | `AlgebraicSystemRoots` | numerically verified original-equation roots, with exact structural preprocessing |
| `polysolve` | Solve an exact zero-dimensional polynomial system | `PolynomialSystemRoots` | numerically verified polynomial roots, with exact structural preprocessing |
| `algebraize_system` | Convert supported algebraic equations to an augmented exact polynomial system | `AlgebraizedSystem` | exact transformation data; not itself a root solve |
| `recognize_system_roots` | Reconstruct exact algebraic coordinates from already-found numerical roots | `tuple[RecognizedSystemRoot, ...]` | optional scalar and exact joint certification |

## Exact quotient-algebra API

`QuotientAlgebra` is the expert-facing structural representation of a finite exact quotient algebra. It centralizes the Gröbner basis, standard-monomial staircase, quotient dimension, exact normal forms and coordinate vectors, variable and general multiplication matrices, trace pairing, and separating-element construction used by the RUR, action-matrix, and exact border-basis backends.

Construct it with `QuotientAlgebra.from_polynomials(polynomials, variables, order="grevlex", max_dimension=None)` or, when a compatible exact Gröbner basis already exists, `QuotientAlgebra.from_groebner_basis(...)`. Positive-dimensional ideals, inexact coefficients, or an exceeded dimension guard raise `QuotientAlgebraError`.

`SeparatingElement` is the exact structural record returned by `QuotientAlgebra.separating_element(...)`; it contains the successful linear form and coefficients, its squarefree defining polynomial, the coordinate denominator, trace vector, Krylov power vectors, and geometric solution count.

This API is equality-only. Boolean formulas, inequalities, real-sign filtering, and quantifier elimination belong in downstream semialgebraic packages rather than this quotient layer.

## `algsolve`

```python
algsolve(
    equations,
    variables,
    *,
    digits=50,
    verification_digits=None,
    guard_digits=10,
    maxsteps=200,
    max_solutions=10_000,
    max_action_dimension=256,
    max_precision_digits=None,
    max_auxiliary_variables=32,
    method="auto",
    recognize=True,
    recognition_max_degree=8,
    max_homotopy_paths=10_000,
    homotopy_seed=0,
    homotopy_gamma_attempts=4,
    homotopy_parallel=False,
    homotopy_max_workers=None,
    presolve=True,
    variable_order="auto",
    root_mode="distinct",
    multiplicity="auto",
    root_order="canonical",
    max_returned_roots=10_000,
    ordering_max_refinements=64,
)
```

Find all distinct verified roots of a supported zero-dimensional algebraic equation system.

### Options

| Option | Default | Meaning |
|---|---:|---|
| `equations` | required | iterable of exact equations or expressions interpreted as zero |
| `variables` | required | ordered sequence of distinct SymPy `Symbol` objects |
| `digits` | `50` | requested numerical output precision; must be at least 15 |
| `verification_digits` | `None` | residual-verification precision target; when supplied, must be in `[1, digits)` |
| `guard_digits` | `10` | extra internal numerical working digits |
| `maxsteps` | `200` | positive iteration/solver-step budget used by numerical subroutines |
| `max_solutions` | `10_000` | safety bound on solution count |
| `max_action_dimension` | `256` | maximum quotient dimension allowed for the dense action backend |
| `max_precision_digits` | `None` | cap on adaptive numerical precision; if supplied, must be at least `digits` |
| `max_auxiliary_variables` | `32` | maximum auxiliaries introduced during algebraization |
| `method` | `"auto"` | `"auto"`, `"shape"`, `"action"`, `"triangular"`, `"rur"`, or `"homotopy"` |
| `recognize` | `True` | attempt certified exact recognition after solving; set to `False` to skip |
| `recognition_max_degree` | `8` | maximum algebraic degree searched by automatic recognition |
| `max_homotopy_paths` | `10_000` | Bézout path-count safety limit for explicit total-degree homotopy |
| `homotopy_seed` | `0` | seed for the deterministic sequence of exact complex gamma candidates |
| `homotopy_gamma_attempts` | `4` | number of deterministic gamma candidates tried before homotopy fails |
| `homotopy_parallel` | `False` | track independent total-degree paths in spawned worker processes |
| `homotopy_max_workers` | `None` | optional worker-process cap for homotopy |

The result is an `AlgebraicSystemRoots`.

The final root list has passed original algebraic equation and domain/branch validation. See [Branch Semantics](branch-semantics.md).

## `polysolve`

```python
polysolve(
    equations,
    variables,
    *,
    digits=50,
    verification_digits=None,
    guard_digits=10,
    maxsteps=200,
    max_solutions=10_000,
    max_action_dimension=256,
    max_precision_digits=None,
    method="auto",
    recognize=True,
    recognition_max_degree=8,
    max_homotopy_paths=10_000,
    homotopy_seed=0,
    homotopy_gamma_attempts=4,
    homotopy_parallel=False,
    homotopy_max_workers=None,
    presolve=True,
    variable_order="auto",
    presolve_max_terms=1000,
    presolve_max_degree=16,
    presolve_growth_factor=4,
    certify=False,
    certification_max_dimension=128,
    certification_max_refinements=128,
    certification_max_box_attempts=16,
    homotopy_recovery=False,
    recovery_options=None,
    root_mode="distinct",
    multiplicity="auto",
    root_order="canonical",
    max_returned_roots=10_000,
    ordering_max_refinements=64,
)
```

This is the lower-level polynomial solver. It uses the same numerical controls as `algsolve` but does not algebraize rational powers/rational functions or perform original algebraic branch filtering.

Returns `PolynomialSystemRoots`. Exact recognition is attempted by default with `recognition_max_degree=8`; pass `recognize=False` to return only the numerical result metadata. Automatic recognition is best-effort and does not invalidate a verified numerical solve if no relation is certified.

For `method="homotopy"`, `max_homotopy_paths` limits the Bézout path count before tracking starts. `homotopy_seed` starts a deterministic sequence of exact complex gamma candidates and `homotopy_gamma_attempts` controls how many are tried before failure. `homotopy_parallel` / `homotopy_max_workers` control optional process-based path parallelism; each worker compiles the symbolic homotopy once and then tracks multiple start paths. These controls do not make `method="auto"` select homotopy.

## `algebraize_system`

```python
algebraize_system(
    equations,
    variables,
    *,
    max_auxiliary_variables=32,
)
```

Returns an `AlgebraizedSystem` containing the exact original equations and their augmented polynomial representation.

Algebraization is not itself a proof that every augmented polynomial root satisfies the original expression. Polynomialization can introduce branch candidates; use `algsolve` for solving and filtering.

## `recognize_system_roots`

```python
recognize_system_roots(
    result,
    *,
    max_degree,
    max_height=None,
    min_relation_bits=None,
    degree_cost=0,
    require_certified=False,
)
```

Run `algrecognize` on the numerical coordinates in an existing root result, reconstruct compatible exact SymPy algebraic roots, and test the reconstructed tuple against the original equations.

| Option | Default | Meaning |
|---|---:|---|
| `result` | required | a `PolynomialSystemRoots`, `AlgebraicSystemRoots`, or `MonodromyOrbitResult` |
| `max_degree` | required | positive maximum algebraic degree considered during recognition |
| `max_height` | `None` | optional recognition coefficient-height bound passed to `algrecognize` |
| `min_relation_bits` | `None` | optional relation-quality bound passed to `algrecognize` |
| `degree_cost` | `0` | degree penalty used by recognition |
| `require_certified` | `False` | require scalar certification and exact joint equation certification |

Returns a tuple of `RecognizedSystemRoot`. For monodromy results, recognition uses each root's aligned `MonodromyRootInfo.digits` and `verification_digits`; exact root recognition does not change the orbit's completeness evidence.

If `require_certified=True` and the stronger certification cannot be established, `ExactCertificationError` is raised.

See [Exact Recognition](exact-recognition.md) for the recognition pipeline, trusted precision, failure modes, and the distinction between root certification and completeness.

## Continuation and monodromy API

| Public object | Purpose |
|---|---|
| `HomotopySystem` | protocol consumed by the generic path tracker |
| `SympyHomotopy` | compiles a SymPy analytic homotopy, Jacobian, and parameter derivative |
| `PathTrackerOptions` | step-size, precision, Newton, and rejection controls |
| `track_path` | mixed-precision predictor/corrector path tracking |
| `PathStep` | diagnostics for one accepted continuation step |
| `PathResult` | complete result/diagnostics for one tracked path |
| `PathTrackingError` | base path-tracking exception |
| `PathStepError` | internal/publicly inspectable failed-step exception |
| `MonodromyLoop` | closed additive coefficient loop definition |
| `closed_additive_loop` | construct `F(x)+g(1-exp(2*pi*i*t))` |
| `track_loop` | track several roots through one loop, optionally in parallel |
| `monodromy_permutation` | match loop endpoints to a supplied root list |
| `MonodromyPermutation` | endpoints, root-index permutation, and path diagnostics |
| `discover_monodromy_orbit` | repeated random-loop orbit discovery for supported polynomial/algebraic systems from supplied seeds |
| `MonodromyOrbitResult` | discovered orbit, stopping evidence, and per-root verification metadata |
| `MonodromyRootInfo` | working precision, verification precision, scaled residual, verification status, and discovery provenance for one orbit root |
| `second_order_trace_test` | additive-family coordinate curvature; no completeness implication |
| `capture_recapture_estimate` | Chapman-corrected mark/recapture population estimate |
| `CaptureRecaptureEstimate` | estimate, standard deviation, confidence interval, and sample counts |

### `discover_monodromy_orbit`

```python
discover_monodromy_orbit(
    equations,
    variables,
    seeds,
    *,
    max_loops=10,
    radius=1.0,
    match_tolerance=1e-7,
    random_seed=None,
    options=None,
    parallel=False,
    max_workers=None,
    expected_root_count=None,
    trace_test=False,
    trace_tolerance=1e-8,
    statistical_stop=False,
    confidence=0.95,
    min_loops_before_stopping=2,
    max_auxiliary_variables=32,
    recognize=True,
    recognition_max_degree=8,
)
```

The high-level orbit-discovery function accepts the same algebraic expression subset as `algsolve`. Algebraic seeds are lifted to principal-branch auxiliary coordinates, tracking occurs in the exact augmented polynomial system, and returned roots are projected and reverified against the original equations and nonzero constraints. `expected_root_count` counts distinct returned original-system roots, not branch-invalid augmented-cover roots.

Low-level `closed_additive_loop`, `track_loop`, and `monodromy_permutation` remain polynomial-system APIs.

Monodromy results record `stopping_reason` and `completeness_basis`. Their `root_info` tuple aligns one-to-one with `roots`; each entry records the endpoint precision, an arbitrary-precision scaled residual recomputed against the original algebraic system after projection, a verification Boolean, and the loop/path that first supplied the retained representative. Only an independently supplied exact root count is an exact-count stopping basis; curvature stopping records `completeness_basis="none"`, while capture-recapture stopping records statistical evidence.

For algebraic input, `tracking_variables` and `tracking_equations` expose the augmented polynomial problem, while `auxiliary_variables` and `nonzero_constraints` expose its algebraization metadata. Perturbations are aligned with `tracking_equations`.

See [Continuation and Monodromy](continuation-and-monodromy.md).

## Result types

### `RootDiagnostics`

Per-root numerical refinement diagnostics.

| Field | Meaning |
|---|---|
| `initial_relative_residual` | relative residual before optional refinement |
| `final_relative_residual` | final accepted relative residual |
| `refinement_attempted` | whether numerical refinement was attempted |
| `refinement_succeeded` | whether refinement produced a usable candidate |
| `refinement_improved` | whether the accepted candidate improved the residual |

### `PolynomialSystemRoots`

| Field | Meaning |
|---|---|
| `roots` | tuple of distinct returned numerical root tuples |
| `variables` | variables in coordinate order |
| `equations` | normalized original polynomial equations |
| `groebner_basis` | exact Gröbner-basis expressions retained by the result |
| `precision_digits` | requested numerical precision |
| `working_digits` | working precision used for the successful numerical solve |
| `verification_digits` | residual-verification setting |
| `max_relative_residual` | largest accepted relative residual |
| `method` | backend that produced the result |
| `diagnostics` | per-root `RootDiagnostics` |
| `eliminant` | shape-position eliminant when applicable, otherwise `None` |
| `parameter_variable` | shape-position parameter when applicable |
| `quotient_dimension` | finite quotient dimension when constructed/applicable |
| `standard_monomials` | quotient staircase exponent tuples when applicable |
| `separator_coeffs` | coefficients of the successful action separator when applicable |
| `rational_univariate_representation` | exact RUR retained by the `rur` backend, otherwise `None` |
| `homotopy_paths_total` | Bézout path count for the total-degree homotopy backend, otherwise `None` |
| `homotopy_paths_succeeded` | paths that reached finite verified endpoints |
| `homotopy_paths_failed` | paths that failed to reach finite verified endpoints for the successful gamma |
| `homotopy_paths_divergent` | paths heuristically classified as likely divergent for the successful gamma |
| `homotopy_gamma` | exact complex gamma used by the successful total-degree homotopy |
| `homotopy_gamma_attempts` | deterministic gamma candidates attempted before success |
| `homotopy_endpoint_smallest_singular_values` | scale-normalized endpoint Jacobian smallest singular values |
| `homotopy_endpoint_condition_estimates` | scale-normalized endpoint Jacobian condition estimates |
| `recognized_roots` | certified exact root records from automatic recognition, or `None` if disabled/unsuccessful |
| `recognition_attempted` | whether automatic recognition was attempted |
| `completeness` | first-class enumeration evidence, separate from coordinate certification |
| `total_multiplicity`, `geometric_solution_count` | global exact quotient counts when applicable |
| `is_radical`, `has_multiple_roots` | exact quotient reducedness when applicable |
| `solver_variables`, `affine_substitutions` | reduced-coordinate provenance and exact reconstruction |
| `recognition_error` | recognition failure diagnostic when the numerical solve succeeded but recognition did not |

There is currently no public `complete` Boolean field. See [Guarantees and Result Semantics](guarantees-and-result-semantics.md).

### `AlgebraicSystemRoots`

`AlgebraicSystemRoots` extends `PolynomialSystemRoots` with:

| Field | Meaning |
|---|---|
| `polynomial_equations` | augmented polynomial equations actually passed to the polynomial backend |
| `polynomial_variables` | variables of the augmented polynomial problem |
| `auxiliary_variables` | variables introduced during algebraization |
| `nonzero_constraints` | retained denominator/base nonzero constraints |
| `homotopy_projected_candidates` | augmented homotopy endpoints that passed original-system projection checks |
| `homotopy_projected_roots_rejected` | augmented endpoints rejected or deduplicated during algebraic projection |

The inherited `roots` are projected back to the user's original variables and filtered against original algebraic semantics.

### `AlgebraizedSystem`

| Field | Meaning |
|---|---|
| `original_equations` | normalized original algebraic equations |
| `polynomial_equations` | exact augmented polynomial equations |
| `original_variables` | user variables |
| `augmented_variables` | original plus introduced variables |
| `auxiliary_variables` | introduced variables only |
| `nonzero_constraints` | retained nonzero domain constraints |
| `auxiliary_lift_expressions` | principal-sheet expressions used to lift original-variable seeds into auxiliary coordinates |

### `RecognizedSystemRoot`

| Field/property | Meaning |
|---|---|
| `numerical_root` | numerical tuple supplied to recognition |
| `coordinates` | scalar `algrecognize` results |
| `exact_coordinates` | reconstructed exact SymPy algebraic values |
| `equation_values` | exact residuals after substitution into original equations |
| `scalar_certified` | every scalar recognition is certified |
| `jointly_certified` | exact reconstructed tuple satisfies every original equation |
| `certified` | both scalar and joint certification hold |

## Package version

`__version__` (available as `algroots.__version__`) reports the installed distribution version. When the source tree is imported without installed package metadata, it falls back to `"0+unknown"`.

## Public exceptions

Common solver exceptions are available at the root. All public exception types are available from `algroots.errors`.

| Exception | Meaning / typical source |
|---|---|
| `PolynomialSystemError` | base class for polynomial-system solver errors |
| `PolynomialSystemInputError` | invalid variables/equations, unsupported algebraic expression, invalid denominator/base, or algebraization/input failure |
| `NotZeroDimensionalError` | solution set is not a finite zero-dimensional variety |
| `ShapePositionError` | forced/attempted shape-position strategy cannot be used |
| `ActionMatrixError` | quotient action construction or isolated simple eigensystem fails |
| `TriangularSolveError` | triangular solving cannot complete reliably |
| `NumericalRootError` | numerical root extraction/refinement/verification fails |
| `SystemSolveLimitError` | configured solution/action/structural safety limit is exceeded |
| `ExactCertificationError` | exact reconstruction or required joint certification fails |

Standard `ValueError`, `TypeError`, or `ImportError` can also be raised for invalid option values/types or unavailable exact-recognition integration.

See [Troubleshooting](troubleshooting.md) for recovery guidance.

## Guarantee levels

| Layer | What is established |
|---|---|
| Input/algebraization | exact algebraic transformation data and retained domain information |
| Gröbner/quotient structure | exact zero-dimensional structural information |
| Numerical root extraction | arbitrary-precision candidates |
| Numerical verification | returned candidates satisfy numerical residual/original-expression checks |
| Completeness reasoning | uses exact structural counts where applicable; not represented by a universal public Boolean |
| Exact recognition | attempted by default; certified exact algebraic coordinates are attached when found |
| Exact joint certification | reconstructed tuple satisfies original equations exactly |

Canonical coordinate ordering is reported by `result.ordering`; numerical fallback does not promise precision-independent near-ties. `result.multiplicities` and `multiplicity_evidence` report per-root counts or explicit unknowns. Exact local proofs remain available through `IsolatedRootCertificate`.

## Rational univariate representations

`RationalUnivariateError`

Raised when an exact rational univariate representation cannot be constructed.

`RationalUnivariateRepresentation`

Exact representation of a finite rational polynomial-system solution set by a separating parameter, a defining univariate polynomial, and rational coordinate functions. The `dimension` property is the quotient-algebra dimension (with multiplicity), while `solution_count` is the number of distinct geometric roots represented by the squarefree defining polynomial.

`RationalUnivariatePoint`

An exact point tied to a root of a `RationalUnivariateRepresentation`. `coordinates` and `assignment` evaluate the normalized coordinate polynomials at that parameter root.

`compute_rational_univariate_representation`

```python
compute_rational_univariate_representation(
    polynomials,
    variables,
    parameter=None,
    *,
    max_separating_attempts=64,
)
```

Construct an exact RUR for a zero-dimensional polynomial system over `QQ` or an exact algebraic number field.

`solve_rur_representation`

```python
solve_rur_representation(
    representation,
    *,
    real=True,
)
```

Return the distinct exact roots encoded by an existing RUR.

`solve_rur_points`

```python
solve_rur_points(
    representation,
    *,
    real=True,
)
```

Return the distinct roots as `RationalUnivariatePoint` objects.

`solve_zero_dimensional_system_with_rur`

```python
solve_zero_dimensional_system_with_rur(
    polynomials,
    variables,
    *,
    real=True,
    parameter=None,
    max_separating_attempts=64,
)
```

Construct an RUR and return its distinct exact roots.

## Exact border bases

`BorderBasisError`

Raised when exact border-basis construction fails.

`BorderBasisDiagnostics`

Exact structural diagnostics for border-basis construction, including quotient rank, border rank, and multiplication-matrix commutation checks.

`BorderBasisResult`

Exact border-basis data for a zero-dimensional quotient algebra. It exposes the order ideal, border polynomials, normal-form coordinates, and multiplication matrices.

`compute_border_basis`

```python
compute_border_basis(
    polynomials,
    variables,
    *,
    order_ideal=None,
    domain=QQ,
    groebner_order="grevlex",
    strict=True,
    algorithm="groebner",
    max_degree=None,
)
```

Construct an exact border basis either from a supporting Gröbner basis or by the native Macaulay linear-algebra route.

`compute_border_basis_linear`

```python
compute_border_basis_linear(
    polynomials,
    variables,
    *,
    domain=QQ,
    groebner_order="grevlex",
    max_degree=None,
    strict=True,
)
```

Construct an exact border basis using Macaulay linear algebra, with a supporting Gröbner basis retained for exact normal-form validation.

### `HomotopySolveError`

Raised when the explicit total-degree homotopy backend cannot track every required path to a distinct regular finite endpoint.


### `CompletenessEvidence`

Enumeration evidence on `PolynomialSystemRoots.completeness` separates an exact
backend count from coordinate certification. `conditional` means the backend
has an exact expected count but the coordinates are numerical residual-verified
approximations. `numerical` is used for regular homotopy path accounting.
`quotient_dimension` / `total_multiplicity` count algebraic multiplicity;
`geometric_solution_count` counts distinct points. `is_radical` and
`has_multiple_roots` compare these exact counts; no per-point local multiplicity
is claimed. Under presolve, `solver_variables`, `affine_substitutions`, staircase
exponents and Gröbner expressions describe the reduced computation. Roots and
RUR coordinates use the original variable order.

### `EndgameResult` and `cauchy_endgame`

`cauchy_endgame(system, start, target=1, radius=0.1, samples=32, max_cycle=8,
levels=4, tolerance=None, options=None)` follows shrinking Cauchy circles with
cycle detection. Supply a path point at `target-radius`. The returned endpoint,
cycle number, successive estimates and endpoint residual are numerical evidence only. The reported error estimate is a difference of sampled means, not a rigorous bound. Convergence
does not prove isolation, local multiplicity, completeness, or a rigorous error
bound. Other branch points within the disk, nonclosing cycles, and divergent
paths may prevent convergence. A failed tracking segment raises `PathTrackingError`.

### `ProjectiveHomotopy` and `projective_homotopy`

`projective_homotopy(expressions, variables, parameter, patch=...)` homogenizes
an analytic polynomial homotopy and adds a constant linear patch. Track its
`system` using `track_path`, starting from `lift(affine_point)`; recover finite
coordinates with `dehomogenize(point)`. Paths reaching the patch hyperplane
require chart switching, which this first implementation does not provide.
This expert API supports some affine paths to infinity and reports no proof of
projective coverage. These endgame and projective APIs are opt-in; `polysolve`
homotopy keeps its conservative finite regular endpoint contract.


## Rigorous singular certificates and deflation

`IsolatedRootCertificate` records an exact point, original equations and Gröbner
basis, finite quotient dimension, geometric count, exact Jacobian rank, local
multiplicity, separating characteristic polynomial and optional isolation box.
`certify_isolated_root(equations, variables, point, max_quotient_dimension=256)`
requires exact algebraic coordinates and a globally zero-dimensional ideal.
`verify()` independently replays the proof. Small numerical residuals never
replace these checks.

`RationalComplexBox` holds open exact rational rectangles per coordinate.
`certify_root_box(equations, variables, box, max_quotient_dimension=256,
max_refinements=64)` proves exactly one root lies in the box using exact RUR
parameter isolation, root counting and rational coordinate enclosures. Rational
coefficients are currently required for this box API. `RootCertificationError`
is raised when the requested proof cannot be established.

`deflate_isolated_root(equations, variables, point, max_stages=8,
max_added_equations=1024, max_quotient_dimension=256)` returns `DeflationResult`.
`DeflationStage` records exact Jacobian pivots and added bordered-minor equations.
Full column rank is proved before `regular=True`. `verify()` replays the proof.
Deflation preserves the target but may delete other roots. `.refine(approximate,
digits=50, maxsteps=50)` returns `DeflatedRefinement` with numerical original and
deflated residuals; it does not automatically certify that approximation.

`cauchy_endgame` now accepts `certify=False`, `certification_box=None`,
`deflate=False` and `max_quotient_dimension=256`. A converged endgame can carry a
unique-root box certificate and targeted deflation. `deflate=True` requires
`certify=True`. The target parameter must be exact for certification. Exact
endpoint certification does not certify path tracking or completeness.

## Automatic charts

`track_projective_path(projective, start, t_start=0, t_end=1, options=None,
segment_size=0.05, min_segment=1e-8, switch_ratio=0.5, max_segments=1000,
max_chart_switches=128)` tracks a homogeneous seed using adaptive coordinate
charts. `ProjectivePathResult` records homogeneous endpoint, current patch,
segments and `ChartSwitch` events. All path evidence remains numerical.
`classification` is `finite`, `numerically_near_infinity` or `unresolved`.
`affine_endpoint()` is available only for numerically resolved finite endpoints.

See [Singular Certification and Charts](singular-certification-and-charts.md)
for proof scope, examples, deflation's local semantics and remaining limitations.

The public entry points are `certify_isolated_root`, `certify_root_box`,
`deflate_isolated_root` and `track_projective_path`.

### Shared extraction, guarded substitution and automatic certificates (0.5.0)

`polysolve` now performs constant-unit polynomial substitution in addition to
linear elimination. `presolve_max_terms=1000`, `presolve_max_degree=16` and
`presolve_growth_factor=4` bound expansion before it happens. An input equation
already exceeding an absolute limit may keep its input size, but cannot grow
past that size. Term growth is also bounded relative to each input row. A
rejected pivot remains in the system; division by variable-dependent pivots is
never allowed. `presolve=False` disables substitution. The compatibility field
`affine_substitutions` records all eliminated variables, including polynomial
substitutions. Original coordinate order and lifted RUR maps are preserved.

`polysolve(..., certify="auto")` (or `certify=True`) attaches a
`RootCertificationAttempt` to every returned numerical root, aligned in
`result.root_certifications`. An attempt has `status`, `certificate` and `error`.
Statuses are `certified`, `failed` or `unavailable`; a failed proof never becomes
a tolerance-based certificate. `certify="required"` raises
`RootCertificationError` if any proof cannot be established. `certify=False`
(the default) disables explicit endpoint certification. The separate default
`multiplicity="auto"` policy can still request exact local proofs for nonreduced
quotients; use `multiplicity=False` to disable that added work. Empty results have no per-root
proof obligations, and return an empty attempt tuple when certification is on.

`certify_numerical_roots(result, max_quotient_dimension=128,
max_refinements=128, required=False)` runs the same batched proof independently
on an existing result. It builds one quotient from the original equations and
shares its separator, RUR, exact parameter roots and characteristic polynomial.
Rational boxes center on the exact binary values of the numerical coordinates;
the default radius is `10**(-min(12, verification_digits, precision_digits//2))`,
with a minimum exponent of six, scaled per coordinate by
`max(1, abs(real_center), abs(imag_center))`. Box uniqueness and coordinate containment are
proved exactly. Closely clustered points may require manually chosen boxes;
`certify_root_box` remains available. Automatic boxes currently require rational
coefficients and globally zero-dimensional ideals. The proof establishes an
endpoint and its multiplicity, never a numerical path or global completeness.

RUR numerical extraction shares the Arb-backed squarefree univariate extractor
with shape/triangular solving. `numerical_rur_roots` accepts
`max_precision_digits`; `polysolve` forwards the same budget. Returned coordinates
remain numerical midpoints and still undergo original-equation verification.
The extraction budget bounds Arb coefficient-ball and root-isolation precision.
Each isolation call uses the current working bit precision as its ceiling;
bounded retries increase this only up to `max_precision_digits`. Exact centroid
translation reduces cancellation before ball conversion. Triangular substitutions
retain exact rational midpoints of prior coordinates before polynomial construction,
so coefficient rounding cannot manufacture repeated factors. As with all numerical
branches, the reconstructed points must pass original-equation verification.
Algebraic coefficient conversion uses additional SymPy evaluation guard digits;
the ceiling refers to Arb working and isolation precision, not symbolic evaluation.

Trace vectors, pairings and general element actions stream basis multiplication
matrices rather than caching all of them. The explicit expert property
`basis_multiplication_matrices` still materializes them when requested. The
pairing and variable-action storage remain quadratic in quotient dimension
(per variable), and exactness is unchanged.

Additional public batched proof function: `certify_numerical_roots`.


### Adaptive certification, cache diagnostics and bounded recovery (0.6.0)

`certify_numerical_roots` now accepts `max_box_attempts=16`. The matching solver
control is `certification_max_box_attempts`. Neighbour distances propose initial
boxes only when numerical candidates match the exact geometric count; duplicate
singular path proposals do not dictate a tiny radius. Exact separator counts and
coordinate isolation remain the proof. Ambiguous images tighten by sixteen;
zero-count images widen by two. Refinement and attempt budgets are independent.
`RootCertificationAttempt.box_attempts` records the work; exhausted attempts
return explicit failure or raise under the required policy.

`QuotientAlgebra.from_polynomials` and `from_groebner_basis` accept
`normal_form_cache_size=128` (zero disables caching). Cache admission bounds both
input and remainder expression complexity and rational coefficient bit length
(8,192 bits); LRU eviction bounds entry count.
`normal_form_cache_info` returns hits, misses, evictions, entries, capacity and
reduction seconds. `clear_normal_form_cache` resets the cache and counters.
Cache values belong to one exact quotient and cannot leak across ideals.

`SolveCostDiagnostics`, available as `result.cost_diagnostics`, reports
`phase_seconds`, `structural_estimates`, `normal_form_statistics` and `policy`.
Normal-form counters describe the solving quotient; certification time includes
any separate original-system proof work.
Timings are observations, not reproducible proof evidence. Nested timing names
such as `solve.groebner` are included in `solve`, so their sums are not total time.
The calibrated portfolio preserves action solving for small regular quotients,
tries RUR first when repeated univariate relations suggest singularity, and
retains existing exact fallback validation. Ordering compares occurrence/degree
and minimum-fill candidates; it changes only when structural graph cost falls.
These estimates do not prove runtime improvement or radicality.

`HomotopyRecoveryOptions` bounds path count (64), gamma attempts (2), original
quotient dimension (128), tracker steps per segment (600), projective segments
(256), chart switches (32), endgame samples (16), cycles (4), levels (3), deflation
stages (4), and added deflation equations (128). Every limit is a positive integer;
endgame samples must be at least eight and levels at least two.

Use `polysolve(..., method="homotopy", homotopy_recovery=True,
recovery_options=HomotopyRecoveryOptions(...))` for sequential bounded recovery.
It retains the original square system, attempts an affine prefix and Cauchy
endgame, then projective chart tracking on failure. Candidates are proved against
one shared original quotient; singular certified endpoints undergo bounded
exact deflation using their existing certificates. `PathRecoveryRecord` entries
in `result.recovery_records` report every path stage, outcome, message, cycle and
chart switch count and per-path phase timings. `root_deflations` aligns with returned roots; a deflation limit
does not invalidate an already proved endpoint.

Recovery currently requires rational coefficients and a globally finite ideal.
It always requires exact endpoint proofs, regardless of the ordinary `certify`
flag. Completeness is certified only when distinct exact endpoint certificates
match the finite quotient's geometric count. This proves finite root accounting,
not numerical paths, roots at infinity or projective path completeness. Failed
budgets raise `HomotopySolveError` with `recovery_records` and
`certification_attempts` and `certified_endpoints`; incomplete root sets are never returned. Parallel
recovery is explicitly unsupported. Ordinary homotopy behavior is preserved
when recovery is disabled.


### Calibrated dispatch, diagnostic retries and cancellation probes (0.7.0)

`SolveCostDiagnostics.policy` is `calibrated-v2`. The expanded benchmark driver
and JSON cover 17 cases across nine families, including algebraic fields,
nonreduced systems, clusters and large coefficients. Repeated multivariate input
factors can favor RUR: square-free analysis is limited to 16 small inputs of degree
at most eight and at most 32 terms. It is a cost hint, not a nonradicality proof.
Explicit backend requests, dimension budgets and exact fallbacks are retained.

`HomotopyRecoveryOptions.max_retry_rounds=1` permits one additional round per
gamma; zero disables same-gamma retries. `retry_precision_growth=2` must be an
integer at least two. Retries only track unproved paths/candidates, increase
precision and correction accuracy within `max_precision_digits`, and reuse exact
certification kernels. Successful certificates are preserved. Reported projective
segment/chart budgets and numerical near-infinity classifications do not trigger
precision-only retries. A new gamma must reconsider every start: path indices
cannot be identified across different homotopies. Each gamma and retry retains
all existing path, step, segment, chart, endgame and proof budgets.

`PathRecoveryRecord` adds `retry_round`, `working_digits`, `retry_reason`,
`certification_status` and `retry_disposition`. Certification errors appear in
`message`. A numerical near-infinity decision is a scheduling hint, not a proof.
Completeness still requires exact distinct endpoint certificates and the exact
finite quotient count; partial sets are never returned as complete.

Presolve can probe substitutions rejected by its growth estimate using exact
sparse arithmetic. Final term/degree limits remain unchanged. The probe bounds
intermediate support to at most 4,096 terms, multiplication/accumulation work to
32,768 operations, rational products to 8,192 bits and exponent bit length to 64.
A refused probe leaves the pivot intact. `AffinePresolve.cancellation_probes` and
`cancellation_acceptances` count this work; high-level cost estimates expose them
as `presolve_cancellation_probes` and `presolve_cancellation_acceptances`.

`QuotientAlgebra.operation_diagnostics` returns a copy of exact-operation timing
and counter observations. `SolveCostDiagnostics.quotient_operation_statistics`
exposes these for the solving quotient, including action construction, traces,
rank, standard-basis coordinate shortcuts and streamed-action reuse. Nested
trace-pairing time includes trace-vector work if the latter was not already
cached. Rational geometric counts use exact-domain rank; algebraic fields retain
the profiled expression-rank route. Streamed multivariate traces retain at most
eight basis actions, so temporary matrix storage remains quadratic in dimension.


### Certification and continuation call signatures

These signatures are checked against the runtime public API. See the workflow
guides for proof scope, resource semantics and executable examples.

`certify_isolated_root`

```python
certify_isolated_root(
    equations,
    variables,
    point,
    *,
    max_quotient_dimension=256,
)
```

`certify_root_box`

```python
certify_root_box(
    equations,
    variables,
    box,
    *,
    max_quotient_dimension=256,
    max_refinements=64,
)
```

`certify_numerical_roots`

```python
certify_numerical_roots(
    result,
    *,
    max_quotient_dimension=128,
    max_refinements=128,
    required=False,
    max_box_attempts=16,
)
```

`deflate_isolated_root`

```python
deflate_isolated_root(
    equations,
    variables,
    point,
    *,
    max_stages=8,
    max_added_equations=1024,
    max_quotient_dimension=256,
)
```

`cauchy_endgame`

```python
cauchy_endgame(
    system,
    start,
    *,
    target=1,
    radius=0.1,
    samples=32,
    max_cycle=8,
    levels=4,
    tolerance=None,
    options=None,
    certify=False,
    certification_box=None,
    deflate=False,
    max_quotient_dimension=256,
)
```

`track_projective_path`

```python
track_projective_path(
    projective,
    start,
    *,
    t_start=0,
    t_end=1,
    options=None,
    segment_size=0.05,
    min_segment=1e-08,
    switch_ratio=0.5,
    max_segments=1000,
    max_chart_switches=128,
)
```

## Root multiplicity views and ordering

Both solvers accept `root_mode="distinct"` or `"with_multiplicity"`. `.roots`
and `.distinct_roots` always contain one record per point, with aligned diagnostics
and certificates. Iteration, `len(result)` and `.output_roots` follow the selected
mode. `.iter_roots(with_multiplicity=True)` expands lazily; `.roots_with_multiplicity`
materializes the same view. Expansion requires established local multiplicities
and checks `max_returned_roots` before allocating or yielding any duplicates.

`.multiplicities` aligns with distinct roots and contains positive integers or
`None`; `.multiplicity_evidence` distinguishes exact radical structure, exact local
quotient certificates and unknown/projected-cover data. `multiplicity="auto"`
uses existing proofs and attempts a bounded batched proof for nonreduced polynomial
systems. `False` disables extra multiplicity proof work; `"required"` raises
`RootMultiplicityError` when a local multiplicity is unavailable. Proof attempts
are in `.multiplicity_certifications`, separate from explicitly requested
`.root_certifications`. Certification dimension, refinement and box budgets bound
polynomial solver proof attempts. Automatic box proofs currently require rational
coefficients. Numerical homotopy path counts are never multiplicities.

Canonical ordering is lexicographic in `(Re(x1), Im(x1), Re(x2), Im(x2), ...)`,
using the supplied variable order after reconstruction and filtering. Exact
certified/recognized points are compared algebraically using root separation
and rational isolating intervals. All point metadata is permuted together.
Otherwise arbitrary-precision numerical coordinates provide a deterministic
order for that result; precision/backend-independent ordering remains unproved.
`.ordering.status` is `"certified"` or `"numerical"` and `.ordering.basis` explains
which route was used. Ordering evidence does not certify completeness or membership.

`root_order="required"` requests bounded exact point certification if necessary
and raises `RootOrderingError` if exact ordering cannot be established. Interval
refinement is limited by `ordering_max_refinements`; exact comparison currently
supports algebraic differences of degree at most 64 (including a conservative composite-degree guard). Opaque real/imaginary projections that cannot be resolved by certificate boxes or exact equality are reported as unresolved, rather than forcing an unbounded symbolic comparison. These are structural budgets,
not wall-clock limits. Zero/one-point order is trivially certified. Principal-branch
and pole filtering in `algsolve` never transfers polynomial-cover multiplicities;
repeated output is refused for such projected roots when multiplicities are unknown.

<!-- algroots: execute -->
```python
import sympy as sp
from algroots import polysolve

x = sp.Symbol("x")
result = polysolve(
    ((x - 1) ** 3 * (x + 2),),
    (x,),
    recognize=False,
    root_mode="with_multiplicity",
    root_order="required",
)
assert result.multiplicities == (1, 3)
assert len(result.roots) == 2 and len(result) == 4
assert len(result.roots_with_multiplicity) == 4
assert result.ordering.status == "certified"
```

`RootMultiplicityError` and `RootOrderingError` are available from `algroots.errors`.
