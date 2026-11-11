# Coefficient fields and specialization

The native backend supports rational coefficients, prime fields, number fields, and rational function fields over QQ or a number field. The prime-field kernel is `f4_tiled_finish`. Algebraic modular components use a separate C++ F4 kernel with FLINT finite-extension arithmetic. Install the FLINT-enabled native wheel to use algebraic constants; rational parameter fields need only the compiled prime-field kernel.

## Compute over a parameter field

Parameters are coefficients, distinct from polynomial coordinates. A basis over `QQ(t)` describes a generic field problem; it does not describe every fiber at a numeric value of `t`.

<!-- algroots: execute -->
```python
import sympy as sp
from algroots import groebner_basis

x, y, t = sp.symbols("x y t")
field = sp.QQ.frac_field(t)
basis = groebner_basis([t*x - 1, y - x], [x, y], domain=field, backend="sympy")
assert basis.reduce(t*x - 1)[1] == 0
assert list(basis) == [x - 1/t, y - 1/t]
```

The default `backend="native"` constructs modular candidates through parameter evaluation and FLINT rational interpolation. The portable example above uses explicit SymPy selection. The adapter reconstructs rational functions across primes and checks fresh native images. The public API verifies over the original field, using independent reduced-basis comparison by default or generator and S-pair identities with `verification="certificate"`. Parameters remain symbolic in the returned basis. Multiple parameters use bounded total-degree interpolation; support and degree stability are candidate evidence, not certificates.

## Use algebraic constants and mixed fields

<!-- algroots: execute -->
```python
import sympy as sp
from algroots import groebner_backends, groebner_basis

x, y, t = sp.symbols("x y t")
alpha = sp.sqrt(2)
field = sp.QQ.algebraic_field(alpha).frac_field(t)
engine = groebner_backends()["native"]
if engine.available and "rational function fields over number fields" in engine.domains:
    basis = groebner_basis([(t + alpha)*x - 1, y - x], [x, y],
                          domain=field, backend="native")
    assert basis.reduce((t + alpha)*x - 1)[1] == 0
```

Number-field coefficients use a fixed algebraic power basis. At each prime the minimal polynomial is factored, repeated-factor primes are rejected, and every irreducible component is computed. Linear factors use the original prime-field engine; higher-degree factors use native FLINT arithmetic. Polynomial CRT combines all components before rational reconstruction. The method does not require inert primes, and it does not choose a modular root as a replacement for the original real embedding. Selected conjugates and exact `CRootOf` constants are preserved.

For mixed fields, parameter values are evaluated inside each modular algebraic component, then CRT recovers the power-basis coordinates and rational interpolation recovers their parameter dependence. This supports constant number-field extensions `K(t_1, ..., t_r)`. Parameter-dependent algebraic extensions and their towers use `AlgebraicFunctionField`; see [Algebraic function fields](algebraic-function-fields.md) for construction, verified primitive maps, native images and branch specialization.

## Preserve exceptional fibers

<!-- algroots: execute -->
```python
import sympy as sp
from algroots import parametric_groebner_basis

x, t = sp.symbols("x t")
result = parametric_groebner_basis([t*x], [x], [t])
assert result.verify()
assert list(result.basis) == [x]
assert result.guard == t
assert list(result.specialize({t: 2})) == [x]
assert list(result.specialize({t: 0})) == []
assert result.as_formula().subs({t: 0, x: 7}) == sp.true
```

The generic basis `x` alone loses the whole fiber at `t=0`. Its coefficients have no denominators, so inspecting only the output would miss the exception. `parametric_groebner_basis` derives its guard from exact forward and reverse membership identities and standard S-pair reductions. S-pair product leading monomials must be strictly smaller than the pair LCM; arbitrary S-pair identities cannot certify a basis. `verify()` independently replays the identities and derives the guard again.

`generic_condition` gives the open set where the stored reduced basis specializes correctly. `exceptional_condition` retains defined input points outside that open set. `specialize` requires all parameter values to be exact algebraic constants; it reuses the proven basis on the open set and recomputes original generators on exceptional points. Input coefficient poles are excluded, not treated as ordinary fibers. For example, `x/t` is undefined at `t=0`, while `t*x` is defined there and becomes zero. These are conditions on canonical coefficient-field elements: a removable factor such as `t/t` represents `1`. The semialg preprocessing adapter also preserves denominator conditions visible in the original real predicates. Poles already erased by symbolic simplification cannot be recovered.

`as_formula()` returns the disjunction of guarded generic equalities and original exceptional equalities. Semialg's `solvability_conditions(..., groebner_backend="native")` uses this preprocessing for outer conjunctive equalities and runs complete real elimination on the whole formula, including exceptional branches and original inequalities. This is not a comprehensive Gröbner-basis computation or a root-count/sign-invariance certificate. The complete CAD calculation still handles real geometry and exceptional loci.

## Bounds and profiling

`NativeF4Limits` retains the original prime-field budgets and adds:

| Setting | Default | Meaning |
| --- | --- | --- |
| `max_field_degree` | 16 | Constant-field power-basis dimension |
| `max_parameter_degree` | 16 | Total numerator/denominator degree of each scalar rational coordinate function |
| `max_evaluation_points` | 256 | Distinct reconstruction points per modular attempt; separately bounds fresh validation attempts |
| `max_interpolation_terms` | 512 | Combined numerator/denominator interpolation columns |
| `max_field_matrix_cells` | 4194304 | Logical row × column bound for finite-extension matrices |

Finite-extension images support up to 64 polynomial coordinates. Power-basis coordinate denominators can have higher degree than a coefficient's displayed denominator over `K`. Raise degree and point bounds together when necessary. Bounds refuse incomplete work; an explicit native request never substitutes a SymPy candidate. The independent exact reference and coefficient bit growth are not covered by native operation counts. Use [supervised execution](supervised-execution.md) for an overall calculation deadline.

`proof_limits` in `parametric_groebner_basis` separately bounds tracked witness pairs, active basis size, returned witness nodes, and guard terms. Witness construction can be more expensive than candidate construction. Guards are sufficient conditions and need not be minimal. Failure to construct a bounded proof raises; no unverified guard is returned.

The expert `algroots.native_fields.native_field_basis` returns image statistics and a candidate; `certify=True` adds exact comparison. Its timers separate modular images, interpolation, reconstruction, fresh validation, and exact verification. Validation includes modular-image calls, so nested times must not be summed. `collect_batch_metrics=True` retains native per-image matrix dimensions, degrees, pairs, intermediate-basis growth, and separate symbolic/matrix/insertion timings. The native interface uses one F4 algorithm across its supported coefficient fields.

## Build the FLINT extension

The Python package uses python-flint for factorization and interpolation. Building the separate C++ coefficient kernel also requires FLINT development headers and libraries:

```bash
ALGROOTS_BUILD_NATIVE_F4=1 ALGROOTS_BUILD_NATIVE_EXTENSION=1 python -m pip install .
```

Native wheels use the Python 3.11 Stable ABI; see [native wheels and Python compatibility](native-wheels.md) for binary requirements and artifact validation. The build uses `pkg-config flint`. Alternatively set `ALGROOTS_FLINT_PREFIX` to an installation containing `include/flint` and the FLINT, GMP, and MPFR libraries in `lib`. Windows builds can use the vcpkg FLINT package. Native wheels must bundle these dependencies using delocate, auditwheel, or delvewheel; `tools/repair_extension_wheels.py` performs the platform-specific repair. A wheel's minimum OS version must match its bundled libraries. The existing prime-only wheel workflow remains available, and a separate extension-field CI workflow builds and tests repaired host wheels.

## Choose dense or sparse interpolation

Use `NativeF4Limits(parameter_interpolation="auto")` for ordinary calculations. A single parameter uses dense rational interpolation. For multiple parameters, `auto` selects sparse interpolation if the configured total-degree dense problem would exceed the interpolation-column or evaluation-point bound. Set `"dense"` or `"sparse"` to compare methods on the same input. A choice changes candidate construction; every public result still receives exact verification.

Dense interpolation uses all parameter monomials through the trial total degree. Sparse interpolation recovers only the numerator and denominator supports needed by the result. It samples lines through a common regular anchor, normalizes each line denominator at that anchor, and recovers homogeneous coefficient polynomials from geometric directions. FLINT nullspaces, polynomial factorization, and Vandermonde solves perform the modular reconstruction. Translation returns the polynomial pair to the original parameters. Extra points and fresh primes screen unsuitable lines and encoding collisions; the selected exact verifier rejects incorrect proposals.

Sparse interpolation is useful when the number of possible monomials is much larger than the actual coefficient support. Translation can increase support, and line sampling can cost more than a small dense fit. There is no automatic speed guarantee. `max_interpolation_terms` bounds line-fit columns, combined recovered numerator/denominator support, and intermediate translated supports, `max_sparse_monomials` (default 100000) bounds exponent decoding, and `max_evaluation_points` bounds actual native evaluations per prime. Sparse decoding starts at a larger prime when needed to reduce exponent collisions; `stats["first_image_prime"]` records the chosen value. The ordinary native prime bound still applies.

For example, `(a**6 + b**5 + c*d + e)/(1 + a*b)` has few terms despite having five parameters. With polynomial coordinate `x`, construct `field = QQ.frac_field(a, b, c, d, e)` and pass `[x - coefficient]` to `groebner_basis(..., domain=field, backend="native")`. The default sparse route handles this example within its normal bounds; a dense fit needs more points and columns. Parameters and polynomial coordinates must remain disjoint.

## Construct a generic RUR

A rational univariate representation describes a finite algebraic variety using one auxiliary variable and exact coordinate maps. Supply `domain` to construct it over a rational parameter field or a mixed constant-number/parameter field. This lets downstream symbolic algorithms work with one polynomial without repeatedly eliminating coordinates.

<!-- algroots: execute -->
```python
import sympy as sp
from algroots.rational_univariate import compute_rational_univariate_representation

x, y, a = sp.symbols("x y a")
source = [x*x - a, y - x]
representation = compute_rational_univariate_representation(
    source, (x, y), domain=sp.QQ.frac_field(a)
)
assert representation.dimension == 2
assert representation.solution_count == 2
maps = dict(zip((x, y),
                (p.as_expr() for p in representation.normalized_coordinate_polynomials())))
for equation in source:
    identity = sp.Poly(equation.subs(maps, simultaneous=True),
                       representation.parameter, domain=sp.QQ.frac_field(a))
    assert identity.rem(representation.defining_polynomial).is_zero
special = compute_rational_univariate_representation(
    [p.subs(a, 0) for p in source], (x, y)
)
assert special.dimension == 2
assert special.solution_count == 1
```

Select `groebner_backend="native"` and `groebner_limits=NativeF4Limits(...)` for verified native construction. Semialg's `compute_rur` accepts the same `domain` and backend choices, and its cache separates domains, backends, and limits. The auxiliary RUR variable is distinct from both coordinates and coefficient parameters, including when an automatic name is chosen.

Here the generic algebra has two geometric points, but the zero fiber has one point with multiplicity two. Even a Gröbner specialization guard equal to one does not prove that root counts or signs are constant. Generic RUR root enumeration is refused: specialize the original equations and rebuild before extracting exact or numerical roots. RUR construction requires characteristic zero. These restrictions also apply when a generic unit ideal describes no generic points but an exceptional fiber may be nonempty.

## Profile a complete algorithm

`algroots.profiling.GroebnerProfiler` records calls to the public verified basis API without changing its mathematical return type. It can surround quotient, RUR, or adapter calls that select that API. Direct SymPy calculations and cache hits do not create public basis records. Clear application caches when measuring construction costs.

<!-- algroots: execute -->
```python
import sympy as sp
from algroots import groebner_basis
from algroots.profiling import GroebnerProfiler

x, y = sp.symbols("x y")
with GroebnerProfiler() as profile:
    basis = groebner_basis([x*x-y, y*y-1], (x, y), backend="sympy")
assert basis.reduce(x**4-1)[1] == 0
assert profile.records[0]["basis_size"] == 2
assert profile.records[0]["completed"]
```

Each record separates conversion, candidate construction, exact verification, output construction, and total public-call time. Native records also contain reconstruction/interpolation statistics. `collect_batch_metrics=True` retains per-image rows, columns, degrees, pair counts, basis growth, recorded F4 stages, and the compiled-call elapsed time. The compiled-call timer includes its C API parsing/export boundary. Native image time also includes coefficient evaluation and Python conversion. Timers are nested: do not sum image time, its kernel time, and its symbolic/elimination subtimers as disjoint costs. Profiling scopes are local to the current context; supervised child processes require their own profiling scope.

For complete semialg measurements, run `scripts/benchmark_field_pipeline.py --output results.json` from the semialg source tree. It measures RUR construction plus coordinate normalization, recursive GTZ decomposition, and localized Artinian decomposition with repeated roots. The native path retains independent exact basis verification, and decomposition certificates are replayed separately. The script clears semialg caches for every repetition and saves raw samples and nested profiles. Missing optional native components are reported as unavailable; mathematical failures stop the benchmark.

Reusable exact basis identities and RUR point-count specialization guards are available separately; see [Sparse quotients and certificates](sparse-quotients-and-certificates.md). A Gröbner specialization guard alone does not preserve geometric root counts.
