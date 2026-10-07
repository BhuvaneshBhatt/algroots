# Performance Model

The cost of solving a system is governed by several different quantities. The number of input equations alone is usually a poor predictor.

## 1. Gröbner-basis complexity

The structural front end computes an exact Gröbner basis. This can dominate difficult problems before numerical root extraction begins.

Cost depends strongly on:

- number of variables;
- polynomial degrees;
- monomial structure and sparsity;
- coefficient domain;
- coefficient bit size;
- variable ordering;
- geometry of the ideal.

Intermediate expression growth can be far larger than the final Gröbner basis.

Arbitrary-precision numerical linear algebra does not remove this exact preprocessing cost.

## 2. Quotient dimension

For a zero-dimensional ideal,

$$
D=\dim_K K[x_1,\ldots,x_n]/I.
$$

This is a central size parameter.

The standard-monomial basis contains $D$ elements. The action backend constructs a dense separator multiplication matrix

$$
M_L\in\mathbb{C}^{D\times D}.
$$

`max_action_dimension` prevents unexpectedly large dense action problems.

## 3. Exact normal-form work

The optimized action backend constructs only $M_L$ for

$$
L=\sum_i c_i x_i
$$

and computes one normal form for each coordinate $x_i$.

This reduces the number of expensive exact reductions from the older conceptual strategy of constructing every $M_{x_i}$.

Roughly, the reduction count changes from an $nD$ pattern toward a $D+n$ pattern, where $n$ is the number of variables.

The reductions themselves are not equal-cost, so this is a structural complexity guide rather than an exact timing formula.

## 4. Dense eigensolve

Once $M_L$ is numerical, the action backend uses arbitrary-precision dense eigensystem computation.

Dense eigenvalue algorithms have cubic scaling in matrix dimension in the usual arithmetic model:

$$
O(D^3)
$$

arithmetic operations, with a substantial additional cost from arbitrary-precision arithmetic.

This means doubling $D$ can be far more expensive than doubling the requested decimal precision.

## 5. Precision

If roots/eigenvalues are well separated, modest guard precision may be enough.

Nearly colliding roots can require adaptive retries at increasing precision. The cost then depends on both $D$ and the number of bits carried by every matrix operation.

`max_precision_digits` bounds this process.

## 6. Number of variables

After the quotient basis is known, increasing the number of variables no longer requires one dense multiplication matrix per variable. Coordinate recovery uses the normal forms

$$
\operatorname{NF}(x_i).
$$

There is still additional exact normal-form and numerical evaluation work per coordinate, but the expensive dense eigensolve is shared.

## 7. Coefficient size and domain

Large integers, large rational denominators, or algebraic coefficients can make exact Gröbner reduction and coefficient conversion more expensive.

Two systems with the same degrees and quotient dimension can therefore have very different runtimes.

## 8. Shape-position backend

When a usable shape representation exists, the numerical root problem is a univariate polynomial root problem rather than a dense $D\times D$ eigensystem.

This can be much cheaper. It is one reason `method="auto"` is generally preferable to forcing the action backend.

## 9. Algebraization overhead

Rational powers can introduce auxiliary variables. More auxiliary variables can increase Gröbner complexity dramatically even when the original expression looks compact.

`max_auxiliary_variables` is therefore both a safety bound and a performance bound.

## 10. Validation and refinement

Polynomial residual evaluators and the Jacobian are compiled once, so repeated candidate verification avoids repeated SymPy expression traversal.

Newton refinement costs repeated numerical evaluations and linear solves. It is generally much cheaper than rebuilding symbolic structures for each root, but singular or ill-conditioned Jacobians can require more work.

## Practical expectations

The easiest problems tend to have:

- modest polynomial degree;
- sparse structure;
- small exact coefficients;
- small quotient dimension;
- simple, well-separated roots.

The hardest problems tend to combine:

- expensive Gröbner bases;
- large $D$;
- dense action matrices;
- nearly multiple roots;
- large coefficient heights;
- many algebraization auxiliaries.

For backend selection, see [Choosing a Backend](choosing-a-backend.md).


## New front-end and storage heuristics

Presolve uses exact constant-unit pivots, including polynomial relations such as x-y**2. Degree, term-count and relative-growth bounds are checked before expansion. It cannot divide by a variable-dependent coefficient.
After substitution, automatic ordering sorts remaining variables by equation
occurrence count, maximum degree, then supplied position. The public coordinate
order is unchanged. This is a simple reproducible heuristic; adversarial systems
can prefer the input order.

Exact coordinate multiplication matrices retain sparse storage below 10% measured
density. Separators of dimension at least 12 with sparse storage try incremental
packed Krylov elimination. Only full cyclic closure may replace the characteristic
polynomial; noncyclic closure falls back exactly, preserving nonreduced multiplicity.
The dense numerical action eigensolver and potentially expensive trace-pairing
rank calculation remain limits. No general speedup claim is made for the thresholds.

`benchmarks/quotient_profile.py` separates quotient construction, coordinate
actions, trace vectors, pairings and rank, and includes a cumulative-time profile.
Repeated monomial matrix powers dominated the measured multivariate cases.
A bounded eight-action predecessor cache now reuses exact products; standard
basis coordinate vectors avoid unnecessary reductions. Rational domain rank
outperformed expression rank in the measured rational cases, while algebraic
conversion lost in the algebraic case, so only rational rank uses that route.
Exact dense oracles, cache bounds and rank backend guards test correctness and
resource behavior without asserting wall-clock speed.

Root-output preparation has its own `root_output` cost phase and is included in
reported total time. Exact local multiplicity proofs may add work for nonreduced
quotients. `multiplicity=False` skips that automatic work; strict multiplicity or
ordering requests can require additional proof work or refuse at their budgets.
