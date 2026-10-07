# Choosing a Backend

Both `polysolve` and `algsolve` accept:

```python
method = "auto"
method = "shape"
method = "action"
method = "triangular"
method = "rur"
method = "homotopy"
```

For most applications, use `"auto"`.

## Summary

| Backend | Best use | Main strength | Main failure mode |
|---|---|---|---|
| `auto` | normal use | chooses among available strategies | inherits backend limits |
| `shape` | systems in usable shape position | reduces root extraction to one univariate polynomial | system is not in the required shape form |
| `action` | moderate zero-dimensional radical systems | simultaneous multivariate recovery from quotient algebra | no isolated simple separator eigensystem; quotient too large |
| `triangular` | fallback / triangularizable systems | does not require a simple action eigensystem | symbolic triangular solving can grow expensive |
| `rur` | exact rational/algebraic coefficient systems | exact univariate representation retained | exact quotient construction can be expensive |
| `homotopy` | regular square systems with finite nonsingular Bézout endpoints | avoids Gröbner preprocessing and paths are independent | path count can be large; singular/infinite endpoints currently fail |

## `auto`

```python
result = polysolve(system, variables, method="auto")
```

This is the recommended default. Affine presolve and automatic variable ordering precede one grevlex basis. A cheap exposed shape is preferred. Action is attempted when quotient dimension is at most `min(64, max_action_dimension, max_solutions)`; unless bounded repeated-factor hints favor RUR; larger quotients favor RUR. RUR reuses the same quotient and is also the action fallback. Only shape/triangular requests and the final triangular fallback convert to lex with FGLM. These conservative cost thresholds are deterministic heuristics, not universal performance optima. Use `variable_order="input"` and/or `presolve=False` for controlled comparisons.

Use an explicit backend primarily for:

- reproducible algorithm comparisons;
- diagnosis;
- benchmarking;
- applications that know the system structure in advance.

Do not assume that forcing a backend makes the result more accurate.

## `shape`

The shape-position strategy looks for a representation conceptually like

$$
p(t)=0,
$$

with the other coordinates recoverable from $t$.

Root extraction then becomes:

1. solve the exact univariate eliminant numerically with arbitrary precision;
2. evaluate the coordinate relations;
3. verify the resulting tuples against the original system.

### Strengths

- only a univariate root problem needs numerical root isolation;
- can be very efficient when the Gröbner basis already exposes shape position;
- coordinate association is straightforward.

### Failure modes

`ShapePositionError` can occur when the required structure is absent or cannot be used reliably.

A system being zero-dimensional does **not** imply that the current variable ordering will expose the desired shape form.

## `action`

For the quotient algebra

$$
A=K[x_1,\ldots,x_n]/I,
$$

choose a linear form

$$
L=c_1x_1+\cdots+c_nx_n.
$$

The backend constructs only the multiplication matrix $M_L$, not one matrix for every coordinate. If the separator has an isolated simple eigensystem, its eigenvectors encode evaluation on the quotient basis. Coordinate values are recovered from the normal forms

$$
\operatorname{NF}(x_i).
$$

### Strengths

- naturally handles coupled multivariate systems;
- exact quotient dimension provides structural root-count information;
- coordinate pairing comes from one common eigensystem;
- arbitrary-precision FLINT/Arb numerical linear algebra handles root extraction.

### Failure modes

`ActionMatrixError` can arise when:

- the separator has non-simple eigenvalues;
- a complete simple eigenbasis cannot be isolated;
- exact quotient reduction fails;
- the quotient basis is unusable.

`SystemSolveLimitError` is used when a configured action dimension or solution limit is exceeded.

The action backend is particularly sensitive to multiplicity and nearly colliding eigenvalues. Adaptive precision helps with insufficient arithmetic precision but does not remove mathematical ill-conditioning.

## `triangular`

The triangular strategy solves a triangularized representation recursively.

### Strengths

- useful as a fallback when the action matrix cannot provide a simple eigensystem;
- can handle structures that are awkward for separator-based recovery.

### Failure modes

`TriangularSolveError` indicates that triangular solving could not produce a reliable finite solution set. Symbolic growth can also make this strategy expensive.

## Algebraic input

`algsolve` first algebraizes rational functions and rational powers. Backend selection then applies to the augmented polynomial system.

The backend therefore does **not** decide whether a radical branch is valid. After projection to the original variables, `algroots` checks the original algebraic equations and retained domain constraints.

## Practical guidance

Use:

```python
method = "auto"
```

unless you have a specific reason not to.

For debugging a system supported by multiple backends, compare:

```python
shape = polysolve(system, variables, method="shape")
action = polysolve(system, variables, method="action")
```

The root sets should agree up to numerical matching when both methods succeed.

For numerical difficulties in the action backend, see [Precision and Conditioning](precision-and-conditioning.md).


## RUR

Use `method="rur"` when the system has rational or exact algebraic polynomial coefficients and an exact rational-univariate representation is desirable. The backend constructs the RUR exactly, evaluates all complex branches numerically at the requested precision, and verifies them with the same residual/refinement machinery used by the other core backends. The exact representation is retained as `result.rational_univariate_representation`.

RUR is currently an explicitly selected backend; `method="auto"` continues to use shape, action, and triangular routes. Exact algebraic coefficients are handled natively in a compositum number field; inexact or transcendental coefficients are outside the RUR backend's contract.

## Total-degree homotopy

Use `method="homotopy"` to force classical total-degree homotopy continuation. For equation degrees $d_1,\ldots,d_n$, the backend tracks $\prod_i d_i$ start solutions of $x_i^{d_i}-1=0$ through a complex gamma homotopy to the target system. The path count is checked against `max_homotopy_paths` before tracking begins.

The current backend is deliberately limited to **regular square systems for which every total-degree path reaches a finite nonsingular endpoint**. A failed/divergent path or a singular/coalescing endpoint raises `HomotopySolveError`; `algroots` does not currently use a singular endgame or projective continuation to classify those paths. This conservative policy avoids silently presenting an incomplete affine root set as complete.

Independent paths can be tracked in separate processes with `homotopy_parallel=True`; each worker compiles the symbolic homotopy once and then reuses it for multiple starts. `homotopy_seed` starts a deterministic sequence of exact gamma candidates, and `homotopy_gamma_attempts` controls how many are tried before the backend gives up on an otherwise regular target.

`method="auto"` does **not** select homotopy in this release.


For the explicit total-degree backend, see [Total-Degree Homotopy Continuation](homotopy-continuation.md).


## Bounded homotopy recovery

For rational finite square systems, `polysolve(..., method="homotopy",
homotopy_recovery=True)` adds bounded endgame, chart, proof and deflation
orchestration. It computes a quotient for proof accounting, so it does not have
the ordinary route's no-Gröbner property. Recovery certifies finite endpoints/counts,
not numerical paths or infinity. It is sequential and remains outside auto.
See [Recovery Workflow](recovery-workflow.md).
