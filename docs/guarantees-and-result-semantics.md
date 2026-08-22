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

A future explicit completeness-status object could expose more of this reasoning, especially for multiplicities and fallback paths. Users should not invent a `complete` attribute or infer completeness solely from a small `max_relative_residual`.

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
| Multiplicity | not first-class | not currently reported |

## Root ordering

Root ordering is not a stable mathematical API. Do not attach semantic meaning to the order of tuples in `result.roots`. Compare root sets by numerical matching.

## What the result types mean

`PolynomialSystemRoots` describes the numerical solution of a polynomial system and records the exact structural data used by the solver.

`AlgebraicSystemRoots` adds the polynomialized system, augmented variables, auxiliary variables, and nonzero constraints used for algebraic input.

`RecognizedSystemRoot` describes one numerical root after optional exact coordinate reconstruction and joint certification.

For singular and repeated roots, read [Multiplicity and Singular Roots](multiplicity-and-singular-roots.md).
