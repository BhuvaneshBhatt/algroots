# API Reference

This page summarizes the complete public API exported from `algroots`.

## Public functions

| Function | Purpose | Returns | Guarantee level |
|---|---|---|---|
| `algsolve` | Solve a supported exact zero-dimensional algebraic system, including rational functions and rational powers | `AlgebraicSystemRoots` | numerically verified original-equation roots, with exact structural preprocessing |
| `polysolve` | Solve an exact zero-dimensional polynomial system | `PolynomialSystemRoots` | numerically verified polynomial roots, with exact structural preprocessing |
| `algebraize_system` | Convert supported algebraic equations to an augmented exact polynomial system | `AlgebraizedSystem` | exact transformation data; not itself a root solve |
| `recognize_system_roots` | Reconstruct exact algebraic coordinates from already-found numerical roots | `tuple[RecognizedSystemRoot, ...]` | optional scalar and exact joint certification |

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
| `second_order_trace_test` | numerical second-order trace residual for a supplied root set |
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

Monodromy results record `stopping_reason` and `completeness_basis`. Their `root_info` tuple aligns one-to-one with `roots`; each entry records the endpoint precision, an arbitrary-precision scaled residual recomputed against the original algebraic system after projection, a verification Boolean, and the loop/path that first supplied the retained representative. Only an independently supplied exact root count is an exact-count stopping basis; trace and capture-recapture stopping remain numerical/statistical evidence.

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

All solver-specific exceptions are exported from `algroots`.

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

Root ordering is not stable. Multiplicity is not currently reported as a per-root field.

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
