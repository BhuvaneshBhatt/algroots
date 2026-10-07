# Guarantees and Result Semantics

This page distinguishes the several meanings of “all roots,” “verified,” and “certified” used around `algroots`. These concepts are related, but they are not interchangeable.

## Distinct roots

The numerical solvers return **distinct numerical root tuples**. If a system has the same geometric root with algebraic multiplicity greater than one, that point is not repeated in `result.roots`.

For example,

$$
x^2=0
$$

has one distinct root, $x=0$, but that root has multiplicity two.

Therefore,

```python
len(result.roots)
```

counts returned **distinct points**, not algebraic multiplicity.

## Quotient dimension

For a zero-dimensional polynomial ideal

$$
I\subset K[x_1,\ldots,x_n],
$$

the quotient algebra

$$
A=K[x_1,\ldots,x_n]/I
$$

has finite dimension

$$
D=\dim_K A.
$$

When `algroots` constructs this quotient, the value is exposed as:

```python
result.quotient_dimension
```

The quotient dimension counts solutions over the algebraic closure **with algebraic multiplicity**. For a radical zero-dimensional ideal with $D$ simple points, the quotient dimension also equals the number of distinct roots. For a non-radical ideal it can be larger.

Consequently, `quotient_dimension == len(result.roots)` is a useful completeness invariant for simple-root systems, but it is not a universal identity for systems with multiplicity.

## Completeness

`algroots` currently does **not** expose a public Boolean field named `complete`.

Instead, completeness is enforced internally where the backend has an exact structural count that is applicable. In the simple-root action-matrix setting, the exact quotient dimension supplies the expected number of quotient characters/eigenvectors. A candidate set with excellent residuals is not accepted merely because the candidates it contains are accurate.

This distinction is important:

- **verification** asks whether each returned point satisfies the equations;
- **completeness** asks whether all roots that should be represented have been recovered.

`result.completeness` now records status, evidence basis, expected distinct count, returned count, quotient dimension and notes. Exact-backend numerical coordinates receive `conditional` status; inconsistent ideals receive `certified` empty-set evidence. Ordinary homotopy accounting receives `numerical` status. Opt-in bounded recovery receives `certified` finite-root accounting only when distinct exact endpoint proofs match the exact quotient count. Coordinate recognition is separate and does not automatically upgrade completeness. `total_multiplicity`, `geometric_solution_count`, `is_radical` and `has_multiple_roots` describe exact finite polynomial quotients. For algebraic branch/pole projections these fields remain unknown unless the projection is the polynomial identity; the polynomial cover dimension remains available.

See [Completeness versus Verification](completeness-versus-verification.md).

## Numerical verification

`PolynomialSystemRoots.max_relative_residual` summarizes the numerical residual of the returned polynomial roots. Per-root information is available through `result.diagnostics`.

A small residual establishes that a candidate is numerically consistent with the equations at the requested verification scale. It does **not** by itself prove:

- that no roots were missed;
- that a root is simple;
- that two nearby candidates represent distinct exact roots;
- that a numerical coordinate has a particular exact algebraic form.

For algebraic input, `algsolve` additionally checks candidates against the **original algebraic equations and retained domain restrictions**, not merely the polynomialized system.

## `scalar_certified`

After

```python
recognized = recognize_system_roots(...)
```

each `RecognizedSystemRoot` has:

```python
root.scalar_certified
```

This means that every scalar coordinate recognition supplied the certification expected from `algrecognize`.

It is a statement about the coordinates individually. It does not by itself prove that those exact coordinates jointly satisfy the original multivariate equations.

## `jointly_certified`

```python
root.jointly_certified
```

means the reconstructed exact coordinate tuple has been substituted into the original equations and every resulting exact residual has been certified as zero.

This is deliberately stronger than independently recognizing each coordinate.

For example, recognizing two nearby numerical values independently as algebraic numbers does not establish that they satisfy an exact relation such as $x-y=0$.

## `certified`

```python
root.certified
```

is the strongest convenience property on `RecognizedSystemRoot`. It requires both:

1. scalar certification of every recognized coordinate; and
2. exact joint certification against the original equations.

Thus:

$$
\texttt{certified}
=
\texttt{scalar\_certified}
\land
\texttt{jointly\_certified}.
$$

Passing `require_certified=True` to `recognize_system_roots` requires this stronger result and raises `ExactCertificationError` otherwise.

## Numerical verification versus exact certification

These are intentionally separate layers:

| Property | Numerical solve | Exact recognition/certification |
|---|---|---|
| Candidate coordinates | arbitrary-precision numerical | exact algebraic objects when recognized |
| Equation check | numerical residual | exact substitution |
| Branch/domain check | original algebraic expression evaluated numerically | original equations checked again after reconstruction |
| Completeness structure | exact Gröbner/quotient information where applicable | does not discover missing numerical roots |
| Multiplicity | exact per-root counts where established, explicit unknowns otherwise; global quotient metadata | exact individual multiplicity through isolated-root certificates |

## Root ordering

Roots use lexicographic coordinate order in the original supplied variable order, comparing real then imaginary parts. `result.ordering.status == "certified"` establishes that order exactly; numerical fallback uses arbitrary-precision coordinates and can change near ties. Use `root_order="required"` to require exact ordering or receive an explicit refusal. Metadata follows the same permutation.

`result.roots` always stores distinct points. `root_mode="with_multiplicity"` selects repeated iteration and `.output_roots`, bounded by `max_returned_roots`. Per-root counts require exact evidence; unknown counts cause repeated output to refuse. See [multiplicity semantics](multiplicity-and-singular-roots.md).

## What the result types mean

`PolynomialSystemRoots` describes the numerical solution of a polynomial system and records the exact structural data used by the solver.

`AlgebraicSystemRoots` adds the polynomialized system, augmented variables, auxiliary variables, and nonzero constraints used for algebraic input.

`RecognizedSystemRoot` describes one numerical root after optional exact coordinate reconstruction and joint certification.

For singular and repeated roots, read [Multiplicity and Singular Roots](multiplicity-and-singular-roots.md).


Exact singular-root certificates prove the target's existence, isolation, rank
and individual multiplicity. They do not establish numerical path tracking or
all-path completeness. See [Singular Certification and Charts](singular-certification-and-charts.md).

For attribute-by-attribute usage and optional values, see [Reading Results](reading-results.md).
