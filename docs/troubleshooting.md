# Troubleshooting

This page maps common failures to the public exceptions and controls exposed by `algroots`.

## `NotZeroDimensionalError`

### Meaning

The polynomial ideal does not have a finite zero-dimensional solution set in the form required by the all-roots solver.

Typical example:

$$
x+y=0
$$

in two variables describes a curve, not finitely many points.

### What to do

- check whether an equation or constraint is missing;
- verify that all intended equations were supplied;
- if the solution really is a curve/surface, use a representation intended for positive-dimensional varieties instead.

Increasing numerical precision does not fix positive dimension.

## `ActionMatrixError`

### Meaning

The action backend could not obtain a usable quotient-algebra eigensystem or exact reduction.

Examples include:

- separator has non-simple eigenvalues;
- separator did not yield a complete simple eigenbasis;
- separator eigenvector has unusable constant coordinate;
- exact Gröbner reduction failed.

### What to do

Normally use:

```python
method="auto"
```

so another separator/backend can be tried.

If diagnosing the action backend, consider whether the system has repeated roots or severe eigenvalue clustering. Increasing the precision budget can help when the problem is insufficient arithmetic precision, but not when the separator is mathematically non-simple.

## Precision exhaustion / `NumericalRootError`

### Meaning

A numerical stage could not obtain or validate roots within its configured numerical budget.

The action path can increase working precision adaptively, bounded by:

```python
max_precision_digits
```

### What to do

- keep coefficients exact;
- increase `digits` when you need more accurate output;
- increase `max_precision_digits` when isolation/validation needs more working precision;
- inspect whether roots are nearly multiple;
- try `method="auto"` if a backend was forced.

## Unsupported transcendental expression / `PolynomialSystemInputError`

### Meaning

The high-level algebraic front end accepts algebraic operations such as rational functions and rational powers, but not generic transcendental functions such as:

```python
sp.sin(x)
sp.exp(x)
sp.log(x)
```

### What to do

Reformulate only if the problem genuinely has an algebraic equivalent. Do not replace a transcendental equation by an unrelated polynomial approximation if the goal is an all-roots algebraic guarantee.

`PolynomialSystemInputError` is also used for malformed variables/equations and invalid algebraic constructs such as an identically zero denominator.

## Too many auxiliary variables / `PolynomialSystemInputError`

### Meaning

Algebraization of nested/rational powers would exceed:

```python
max_auxiliary_variables
```

### What to do

- inspect the expression with `algebraize_system`;
- simplify algebraically without erasing required domain information;
- increase `max_auxiliary_variables` only if you are prepared for a potentially much harder Gröbner problem.

The default is 32.

## Action dimension or solution limit / `SystemSolveLimitError`

### Meaning

A configured structural safety limit was exceeded.

Relevant controls include:

```python
max_solutions=10_000
max_action_dimension=256
```

### What to do

Increase the limit deliberately after estimating the expected quotient/root count and memory cost.

A dense action matrix scales as $D\times D$, and dense eigensolving is roughly cubic in $D$.

## `ShapePositionError`

The requested shape backend could not obtain/use the required shape representation.

Use `method="auto"` unless shape position is specifically required for your application.

## `TriangularSolveError`

The triangular backend could not complete its recursive solve reliably.

Again, `method="auto"` is generally preferable unless diagnosing the backend.

## Exact recognition failure / `ExactCertificationError`

### Meaning

`recognize_system_roots(..., require_certified=True)` could not produce a fully certified exact result.

This can happen because:

- a scalar coordinate was not recognized within `max_degree`/`max_height`;
- a recognized polynomial did not yield a compatible exact root;
- reconstructed exact coordinates did not jointly satisfy the original equations.

### What to do

First distinguish numerical solving from exact recognition. A numerically verified root can be perfectly useful even if recognition fails.

If exact reconstruction is required:

- solve at higher `digits`;
- choose a sufficiently large `max_degree`;
- adjust recognition bounds deliberately;
- inspect `scalar_certified`, `jointly_certified`, and `equation_values`.

Do not assume that increasing `max_degree` indefinitely is always desirable; it increases the space of candidate algebraic relations.

## `ImportError` from exact recognition

The numerical solver and exact-recognition layer are separate. Recognition requires the `algrecognize` integration to be importable.

Install the package dependencies in the environment used to run recognition.

## Input validation `ValueError`

Numerical controls are validated. Examples include:

- `digits < 15`;
- invalid `verification_digits`;
- negative `guard_digits`;
- nonpositive `maxsteps`;
- nonpositive solution/action limits;
- `max_precision_digits < digits`;
- invalid method names.

These are configuration errors rather than solver failures.

## A useful diagnostic sequence

When a solve fails:

1. call `algebraize_system` for algebraic input and inspect the augmented system;
2. verify the problem is actually zero-dimensional;
3. retry with `method="auto"`;
4. inspect multiplicity/near-multiplicity;
5. increase the precision budget if the failure is numerical isolation;
6. only then increase structural limits such as `max_action_dimension`.

See [Guarantees and Result Semantics](guarantees-and-result-semantics.md) for interpreting successful results.
