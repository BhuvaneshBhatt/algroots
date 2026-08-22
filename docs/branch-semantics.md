# Branch Semantics

Algebraizing radicals and rational powers is not merely a matter of raising both sides to powers. Polynomialization can introduce roots that do not satisfy the original expression.

`algroots` therefore treats the polynomialized system as a **candidate generator**, then checks candidates against the original algebraic equations and retained domain constraints.

## Square roots

Consider:

$$
\sqrt{x}=x-2.
$$

Squaring gives:

$$
x=(x-2)^2,
$$

or

$$
x^2-5x+4=0.
$$

The polynomial roots are $x=1$ and $x=4$.

But:

$$
\sqrt{1}=1\ne-1=1-2,
$$

so $x=1$ is extraneous. Only $x=4$ satisfies the original principal-square-root equation.

```python
import sympy as sp
from algroots import algsolve

x = sp.symbols("x")

result = algsolve(
    [sp.sqrt(x) - (x - 2)],
    (x,),
)
```

`algroots` returns only the branch-valid root.

## Rational powers

For a rational exponent $p/q$, algebraization introduces a polynomial relation involving an auxiliary variable. That relation can forget which principal branch the original expression represented.

For example,

$$
x^{2/3}=4
$$

has polynomial consequences that can suggest both $8$ and $-8$. Under SymPy's principal complex-power semantics, however, the original expression does not treat those candidates identically.

The final original-expression check is therefore essential.

## Negative powers

Negative rational powers also impose a domain restriction.

For:

$$
x^{-1/2}=2,
$$

the base cannot be zero. The algebraized polynomial relation is constructed so that the inverse relation is represented without silently allowing the forbidden base, and final domain checks remain in place.

## Rational-function poles

Consider an unevaluated form of:

$$
\frac{x^2-1}{x-1}=0.
$$

The numerator vanishes at $x=1$, but the original rational expression is undefined there.

`algroots` retains denominator constraints and uses saturation to remove denominator-zero components from the polynomial ideal before numerical solving. It then checks the domain again after projection.

When preserving a removable singularity matters, construct the expression so SymPy has not already canceled it:

```python
expr = sp.Mul(
    x**2 - 1,
    sp.Pow(x - 1, -1, evaluate=False),
    evaluate=False,
)
```

If SymPy simplifies the expression to $x+1$ before `algroots` receives it, the original hole at $x=1$ is no longer recoverable from the expression tree.

## Nested radicals

Nested radicals are algebraized recursively. Each variable-dependent rational power can introduce an auxiliary variable and a polynomial relation.

The important invariant is:

> A solution of the augmented polynomial system is only a candidate for the original algebraic system.

Projection and original-expression validation decide whether the candidate survives.

## Exact algebraic constants

An exact algebraic constant such as:

```python
sp.sqrt(2)
```

that does not depend on the unknown variables remains an exact coefficient. It does not need to become an auxiliary variable merely because its printed form contains a radical.

## Complex branches

Rational powers are interpreted using the principal complex branch used by the numerical expression evaluator. Therefore algebraic identities that are valid after integer powering are not automatically valid as principal rational-power identities over the complex plane.

This is one reason branch filtering is performed on the **original expression**, not on a manually simplified surrogate.

## What is not supported

Generic transcendental branch problems involving functions such as `log`, `exp`, `sin`, or `cos` are outside the algebraic-system model.

See [Supported Problems](supported-problems.md) and [End-to-End Worked Example](end-to-end-example.md).
