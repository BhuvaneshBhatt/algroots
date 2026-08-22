# Multiplicity and Singular Roots

Multiplicity is one of the most important distinctions in polynomial root solving.

## One repeated root is not two nearby roots

Consider:

$$
f(x)=x^2.
$$

It has one geometric root,

$$
x=0,
$$

with algebraic multiplicity two.

Now compare:

$$
g_\varepsilon(x)=(x-\varepsilon)(x+\varepsilon)
=x^2-\varepsilon^2.
$$

For every nonzero exact $\varepsilon$, this has two distinct simple roots:

$$
x=-\varepsilon,\qquad x=\varepsilon.
$$

As $\varepsilon\to0$, the two roots become arbitrarily close, but the algebraic situations remain different. High precision can resolve sufficiently separated simple roots; it cannot turn a genuinely multiple root into a simple one.

## Quotient-algebra view

For a zero-dimensional ideal $I$, the quotient dimension

$$
D=\dim_K K[x_1,\ldots,x_n]/I
$$

counts roots with algebraic multiplicity.

For

$$
I=(x^2),
$$

the quotient has basis $\{1,x\}$, so $D=2$, although there is only one distinct geometric point.

For

$$
I=(x^2-\varepsilon^2),
$$

with $\varepsilon\ne0$, the quotient dimension is also two, but now there are two distinct points.

This is why quotient dimension alone does not tell us the number of distinct roots in a non-radical system.

## Why multiplicity matters to the action backend

The action backend forms multiplication by a separating linear form $L$ in the quotient algebra. For a radical ideal with distinct points, a generic separator produces a simple eigensystem whose eigenvectors encode evaluation at those points.

For a non-radical ideal, multiplication matrices can contain repeated eigenvalues or nontrivial Jordan structure. The current action path deliberately expects an isolated simple eigensystem. It does not pretend that a defective or repeated eigensystem is a collection of well-separated simple roots.

The automatic solver can try another backend, including triangular solving, but **multiplicity reporting is not currently a first-class result feature**.

## Singular roots

For a square system

$$
F(x)=0,
$$

a root $z$ is singular when the Jacobian

$$
J_F(z)
$$

does not have full rank.

Multiple roots are commonly singular. Newton refinement is much less favorable near singular roots because the local linear solve becomes ill-conditioned or singular and the usual quadratic convergence theory for simple roots no longer applies.

`algroots` therefore treats numerical refinement as a candidate-improvement mechanism, not as evidence that a root is simple.

## Nearly multiple roots

Nearly colliding simple roots can be numerically difficult even though the ideal is radical.

For an exact rational $\varepsilon$:

```python
import sympy as sp
from algroots import polysolve

x = sp.symbols("x")
eps = sp.Rational(1, 10**30)

result = polysolve(
    [(x - eps) * (x + eps)],
    (x,),
    digits=60,
    max_precision_digits=240,
)
```

The two roots are exactly distinct, but resolving them may require substantially more working precision than a well-separated pair.

See [Precision and Conditioning](precision-and-conditioning.md).

## Current semantics

At present:

- `result.roots` contains distinct returned numerical points;
- `result.quotient_dimension` records quotient dimension when the backend constructs it;
- multiplicities are not returned alongside roots;
- a repeated root should not be interpreted as several entries merely because the quotient dimension is larger;
- failure to obtain a simple action eigensystem can trigger a separator retry or backend fallback.

Future multiplicity support would ideally expose local multiplicities explicitly rather than overloading the numerical root list.
