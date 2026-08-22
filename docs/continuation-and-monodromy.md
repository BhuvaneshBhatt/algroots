# Continuation and Monodromy

**Status:** The mixed-precision continuation core is shared by the explicit total-degree homotopy backend and the experimental monodromy layer. Monodromy still requires supplied seeds, assumes nonsingular paths, and does not currently implement `method="monodromy"`. Trace/statistical stopping must not be interpreted as exact completeness certification.

`algroots` includes a reusable numerical continuation layer, an explicit `method="homotopy"` backend for regular square polynomial systems, and an experimental monodromy-orbit layer.

The monodromy layer discovers roots reachable from supplied seed roots and can stop using an externally supplied exact root count, a numerical second-order trace test, or a Chapman-corrected capture-recapture estimate. The result records the evidence used; trace/statistical stopping is not mislabeled as an exact completeness proof.


## Total-degree homotopy backend

`polysolve(..., method="homotopy")` constructs the start system $G_i=x_i^{d_i}-1$ from the total degree $d_i$ of each target equation and tracks all $\prod_i d_i$ roots of unity through $H=(1-t)\gamma G+tF$. Gamma candidates are generated deterministically from `homotopy_seed` using exact rational real and imaginary parts, so the homotopy itself does not embed binary64 constants. Because the gamma trick is generic rather than universal for one fixed value, the backend retries up to `homotopy_gamma_attempts` deterministic candidates before reporting failure.

The backend uses the same adaptive path tracker described below. Every path must succeed, every endpoint must verify against the target equations, and endpoint Jacobian regularity must be numerically resolvable after scale normalization. Very small singular values trigger higher-precision endpoint re-refinement; a small but stable nonzero singular value is retained, whereas a value that continues collapsing with precision is treated as unresolved/singular. `PolynomialSystemRoots` records the final smallest singular values and condition estimates. The current implementation intentionally raises `HomotopySolveError` when no gamma candidate yields a complete regular finite endpoint set. No singular endgame or projective continuation is implemented yet.

The Bézout path count is bounded by `max_homotopy_paths`. Independent paths can be distributed across spawned worker processes with `homotopy_parallel=True`; the symbolic homotopy is compiled once per worker process rather than once per path. Start points are generated lazily. `method="auto"` is unchanged and does not select this backend.

## Reusable mixed-precision path tracker

A continuation problem supplies an analytic homotopy

$$
H(x,t)=0,
$$

with a solution point at the starting parameter value. Differentiating along a nonsingular path gives

$$
J_xH(x,t)\,x'(t)=-\frac{\partial H}{\partial t}(x,t).
$$

`track_path` uses this tangent for prediction and then applies Newton correction at the new parameter value.

<!-- algroots: execute -->
```python
import sympy as sp
from algroots import PathTrackerOptions, SympyHomotopy, track_path

x, t = sp.symbols("x t")
homotopy = SympyHomotopy([x**2 - (1 + t)], (x,), t)

path = track_path(
    homotopy,
    (1,),
    options=PathTrackerOptions(
        initial_digits=35,
        max_digits=160,
        residual_digits=24,
    ),
)

assert path.success
# endpoint is approximately sqrt(2)
```

The tracker is not tied to `SympyHomotopy`. Any object satisfying the public `HomotopySystem` protocol can be tracked. It must provide:

- `variables`;
- `evaluate(values, t)`;
- `jacobian(values, t)`;
- `parameter_derivative(values, t)`.

### Adaptive step size

Each accepted step consists of:

1. tangent prediction;
2. Newton correction;
3. residual validation.

Easy Newton corrections allow the next step to grow. Failed or difficult corrections shrink the step. Consecutive failures are bounded by `max_step_rejections`.

### Mixed precision

Working precision can increase for two reasons:

- repeated correction failure at the minimum useful step size;
- a large estimated Jacobian condition number indicates that the requested residual accuracy is unsafe at the current precision.

The tracker reuses the same exact symbolic homotopy after increasing precision. `max_digits` bounds escalation.

This is inspired by the mixed-precision predictor/corrector philosophy used in modern polynomial path tracking: precision is increased only where the local numerical problem requires it rather than being fixed at a high value for the entire path.

### Diagnostics

`PathResult` records:

- `endpoint`;
- `success`;
- `final_t`;
- `final_digits`;
- accepted/rejected step counts;
- number of precision increases;
- maximum relative residual, retained as an arbitrary-precision `mpmath` value;
- an optional `PathStep` history whose residuals also retain arbitrary precision;
- a coarse classification (`success`, `failed`, or `likely_divergent`) used to distinguish very large near-endpoint affine paths from ordinary correction failure.

A path passing close to a singularity may shrink steps, increase precision, or terminate cleanly instead of silently jumping branches.

## Closed additive coefficient loops

The first monodromy construction is

$$
H(x,t)=F(x)+g\left(1-e^{2\pi i t}\right).
$$

At both endpoints,

$$
H(x,0)=H(x,1)=F(x),
$$

so continuation around the closed parameter loop can map one root of the original system to another root.

Construct a loop with:

<!-- algroots: execute -->
```python
from algroots import closed_additive_loop

loop = closed_additive_loop(
    [x**2 - 1],
    (x,),
    (2,),
)
```

For this example the loop winds the corresponding squared-root value around zero, so the two roots are exchanged.

<!-- algroots: execute -->
```python
from algroots import monodromy_permutation

permutation = monodromy_permutation(
    loop,
    ((1,), (-1,)),
)

assert permutation.permutation == (1, 0)
```

Likewise, the analogous loop for `x**3 - 1` produces a three-cycle when the roots are supplied in compatible cyclic order.

## Duplicate matching

Monodromy endpoints are numerical. Internal matching uses a scale-aware infinity distance rather than raw absolute distance, so matching behaves reasonably for roots whose coordinates differ greatly in magnitude. The calculation remains in arbitrary precision; it does not convert coordinates through Python binary64 `complex` values.

Near-duplicate endpoints are consolidated before they are added to an orbit. Unique endpoint labeling and deduplication are intentionally separate operations: an endpoint that is within tolerance of multiple labels is ambiguous for a permutation, but it is still not a new orbit point.

## Orbit discovery

`discover_monodromy_orbit` begins with one or more known seed roots and repeatedly draws closed additive loops. Unlike the low-level loop constructors, this high-level entry point accepts the same exact algebraic expression subset as `algsolve`: rational functions, rational powers/radicals, nested radicals, and exact algebraic coefficients as well as polynomial equations.

For algebraic input, `algroots` first algebraizes the system. Each supplied seed is lifted from the original variables to the introduced auxiliary variables using the corresponding principal-branch expressions. Monodromy then tracks the augmented polynomial system. Every retained endpoint is projected back to the original variables and checked against the original equations and nonzero constraints, so polynomial-cover roots on the wrong branch are not exposed as public orbit roots. `max_auxiliary_variables` applies to this algebraization just as it does in `algsolve`.

The result keeps both views: `variables`/`equations` and `roots` describe the original algebraic problem, while `tracking_variables`, `tracking_equations`, `auxiliary_variables`, and `nonzero_constraints` describe the polynomial tracking problem. Perturbation vectors correspond to `tracking_equations`.

<!-- algroots: execute -->
```python
from algroots import discover_monodromy_orbit

orbit = discover_monodromy_orbit(
    [x**2 - 1],
    (x,),
    [(1,)],
    max_loops=4,
    radius=2.0,
    random_seed=0,
    recognize=False,
)
```

For each loop, every currently known root is tracked. New endpoints that do not match an existing root are added to the discovered orbit and participate in later loops.

`MonodromyOrbitResult` reports:

- roots discovered;
- the original `variables` and `equations`;
- per-root `root_info` metadata aligned with `roots`;
- loops completed;
- number of paths tracked;
- new roots found by each loop;
- perturbation vectors used;
- whether all attempted paths and endpoint verifications succeeded.

Each `MonodromyRootInfo` records `digits`, `verification_digits`, an arbitrary-precision `relative_residual`, `verified`, `source_loop`, and `path_index`. `digits` is the numerical working precision; `verification_digits` is the residual target actually used to verify that retained root. Seed roots have `source_loop=None`; newly discovered roots record the first loop and path that supplied the retained representative. Loop endpoints are re-corrected against the closed base system before they are admitted to the orbit, so this metadata describes the retained base-system root rather than only the raw loop endpoint.

### Algebraic-system example

```python
import sympy as sp
from algroots import discover_monodromy_orbit

x = sp.symbols("x")
orbit = discover_monodromy_orbit(
    ((x**2 - 1) / (x - 3),),
    (x,),
    ((1,),),
    max_loops=4,
    radius=2.0,
    random_seed=0,
    recognize=False,
)

# The public roots are in x only; the denominator-saturation variable is internal.
assert len(orbit.roots) == 2
assert orbit.auxiliary_variables
assert len(orbit.tracking_variables) > len(orbit.variables)
```

The same projection/filtering rule applies to radicals. For example, the polynomial cover of `sqrt(x) - (x - 2)` contains an extraneous sheet with `x = 1`; `discover_monodromy_orbit` filters that point because it does not satisfy the original principal-square-root equation.

## Parallel path tracking

Paths starting from distinct roots on the same loop are independent. `track_loop(..., parallel=True)` and `discover_monodromy_orbit(..., parallel=True)` therefore use a process pool.

Processes are used rather than threads because arbitrary-precision numerical contexts should not be shared implicitly between independent path trackers.

## Stopping criteria

`discover_monodromy_orbit` supports three optional stopping mechanisms.

### Exact expected root count

If an independently justified **distinct** root count is supplied through `expected_root_count`, discovery stops when that many distinct verified **returned roots of the original system** have been found. For algebraic inputs, branch-invalid or pole roots in the augmented polynomial cover do not count toward this total. The result records `completeness_basis="exact_root_count"`.

### Second-order trace test

For the additive parameter family $F(x)-sr=0$, implicit differentiation gives

$$
Jx'=r,
$$

$$
Jx''=-H_F[x',x'].
$$

For a complete generic fiber, the coordinate trace is affine in the slicing parameter, so the summed second derivative vanishes. `second_order_trace_test` evaluates this residual numerically. A passing trace test is recorded as `completeness_basis="trace_test"`; it is numerical evidence, not an exact algebraic certificate.

### Capture-recapture statistical stopping

Successive monodromy loops can be viewed as mark/recapture experiments. `capture_recapture_estimate` uses the Chapman correction to the Lincoln-Petersen estimator and reports an approximate standard deviation and confidence interval. Statistical stopping is recorded as `completeness_basis="statistical"`.

The statistical model describes the explored monodromy population and can still miss disconnected orbits. It must not be interpreted as a deterministic all-roots proof.

`MonodromyOrbitResult` therefore intentionally has no universal `complete` Boolean. Instead inspect `stopping_reason` and `completeness_basis`.

## Residual scaling and precision escalation

Path correction and `PathResult.max_relative_residual` use the same per-equation scaled residual. For `SympyHomotopy`, each equation is scaled by the sum of the magnitudes of its expanded additive terms. This makes convergence invariant under multiplying an equation by a nonzero constant. Custom homotopies use a first-order scale derived from the Jacobian and parameter derivative unless they provide a compatible `residual_scales(values, t)` method.

When condition estimates trigger a precision increase, the tracker first re-corrects the current point at the new precision before taking another predictor step. This avoids carrying an old low-precision state forward unchanged.

## Exact recognition with `algrecognize`

Monodromy results now retain the per-root precision and verification metadata needed to choose a defensible recognition precision. Each retained endpoint is re-corrected against the base system and stored with its working digits and scaled residual.

`recognize_system_roots` accepts `MonodromyOrbitResult` directly. Recognition uses each root's own `verification_digits` when constructing the input to `algrecognize`, while `digits` controls exact-root matching precision. The reconstructed exact tuple is then substituted into every original equation stored on the orbit result.

```python
from algroots import recognize_system_roots

recognized = recognize_system_roots(
    orbit,
    max_degree=4,
    require_certified=True,
)
```

Exact recognition certifies individual reconstructed roots; it does **not** alter `stopping_reason` or `completeness_basis`. A partially discovered orbit can therefore contain exactly certified roots without becoming a certified complete root set.

## What is deliberately not implemented yet

The current monodromy layer still does **not** include:

- singular endpoint endgames;
- projective path tracking;
- automatic seed generation;
- automatic detection/merging of disconnected monodromy orbits;
- integration as `method="monodromy"` inside `polysolve`.

For the distinction between verified roots and a complete root set, see [Completeness versus Verification](completeness-versus-verification.md).
## Endpoint matching safety

Root matching uses a scale-aware tolerance. If an endpoint lies within tolerance of more than one supplied root, the match is treated as ambiguous and reported as unmatched (`None`) rather than choosing an arbitrary nearest label. Tighten tolerances or increase precision before interpreting such a loop as a permutation.


## Worked example: core seed → monodromy → exact recognition

The following example uses the core solver to obtain one numerically verified seed, lets monodromy discover the other root of the same fiber, and then recognizes every discovered root exactly.

```python
import sympy as sp

from algroots import (
    PathTrackerOptions,
    discover_monodromy_orbit,
    polysolve,
    recognize_system_roots,
)

x = sp.symbols("x")
core = polysolve((x**2 - 1,), (x,), digits=50, recognize=False)
# `core.roots[0]` is already a one-coordinate tuple, so pass it directly as a seed.
orbit = discover_monodromy_orbit(
    (x**2 - 1,),
    (x,),
    core.roots[:1],
    expected_root_count=2,
    max_loops=4,
    radius=2.0,
    random_seed=0,
    options=PathTrackerOptions(
        initial_digits=45,
        residual_digits=28,
        initial_step=0.03,
        max_step=0.07,
    ),
    recognize=False,
)
recognized = recognize_system_roots(
    orbit,
    max_degree=1,
    require_certified=True,
)

assert len(recognized) == 2
assert all(item.certified for item in recognized)
assert orbit.completeness_basis == "exact_root_count"
```

The exact certification applies to each reconstructed root. Here completeness comes separately from the independently supplied `expected_root_count=2`; recognition itself does not provide that count.


For the explicit total-degree backend, see [Total-Degree Homotopy Continuation](homotopy-continuation.md).
