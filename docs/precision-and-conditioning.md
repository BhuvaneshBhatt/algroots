# Precision and Conditioning

`algroots` uses arbitrary-precision arithmetic, but **precision and conditioning are different concepts**.

More digits reduce arithmetic error. They do not make an intrinsically ill-conditioned root problem well-conditioned.

## A well-conditioned example

Consider:

$$
x^2-1=0.
$$

The roots $-1$ and $1$ are simple and well separated.

```python
import sympy as sp
from algroots import polysolve

x = sp.symbols("x")

easy = polysolve(
    [x**2 - 1],
    (x,),
    digits=40,
)
```

The two roots are separated by $2$, so modest working precision is ample.

Inspect:

```python
easy.precision_digits
easy.working_digits
easy.max_relative_residual
easy.diagnostics
```

`precision_digits` records the requested output precision. `working_digits` records the working precision associated with the successful solve.

## A nearly multiple example

Now choose an exact rational:

$$
\varepsilon=10^{-30}
$$

and solve

$$
(x-\varepsilon)(x+\varepsilon)=0.
$$

The roots are simple but separated by only

$$
2\times10^{-30}.
$$

```python
eps = sp.Rational(1, 10**30)

hard = polysolve(
    [(x - eps) * (x + eps)],
    (x,),
    digits=60,
    max_precision_digits=240,
)
```

This is not the same problem as $x^2=0$. There are genuinely two roots, but numerical root isolation has to distinguish two extremely close values.

Depending on the backend and starting precision, `algroots` may need a larger `working_digits` value before the root structure and residual checks are reliable.

```python
print(hard.working_digits)
print(hard.max_relative_residual)
```

The precise number of retries is deliberately not part of the public API; it can depend on FLINT behavior and system representation. The important contract is that retries are bounded by `max_precision_digits`.

## Adaptive precision

For the action backend, insufficient eigenvalue isolation or failed original-system residual validation can cause a retry at higher precision.

Conceptually:

```text
initial working precision
        ↓
attempt isolated simple eigensystem
        ↓ failure
double precision
        ↓
retry
        ↓
validate original equations
        ↓ failure
double precision again
        ↓
...
```

until success or the precision budget is exhausted.

The exact Gröbner basis, staircase, and quotient dimension do not need to be recomputed merely because numerical precision is increased.

## `digits`, `guard_digits`, and `verification_digits`

`digits` is the requested numerical output precision.

`guard_digits` adds internal working precision to reduce loss of accuracy during numerical operations.

`verification_digits` controls the residual-verification target. If omitted, the solver chooses its normal default relative to `digits`.

`max_precision_digits` limits adaptive Arb working precision, including its internal univariate isolation ceiling. Algebraic coefficient conversion uses additional SymPy evaluation guard digits; the Arb ceiling does not limit that symbolic conversion. Each Arb call receives the current working precision as its bit ceiling; retries may increase it up to the requested decimal limit. If the ceiling cannot isolate the roots, the solver raises an error instead of exceeding it. The option must be `None` or a Python integer at least `digits`.

Univariate extraction translates the polynomial exactly to its root centroid before converting coefficients to Arb balls, then reconstructs the original coordinates. This reduces cancellation for clusters around a nonzero center without changing the roots or increasing the precision ceiling.

Numerical conversions preserve nonzero real and imaginary components regardless of their absolute size. Root comparison uses coordinate-relative distances, without an absolute unit-scale floor. This preserves small roots near zero while still recognizing repeated numerical candidates. Numerical midpoints of real roots may retain small imaginary noise; use recognized exact coordinates or exact certification to determine realness. Numerical rounding can still prevent resolving extremely close roots relative to their coordinate magnitudes; a root-count mismatch raises an error rather than claiming completeness. Increase `digits` in that case.

`polysolve`, quotient construction and numerical extraction integer budgets reject Boolean values, floating-point values (including NaN and infinity), and values outside their documented range. `certify` accepts only Boolean values or the strings `"auto"` and `"required"`.

## Residuals are scale aware

For a polynomial

$$
f(z)=\sum_\alpha c_\alpha z^\alpha,
$$

the solver uses a relative residual of the form

$$
\rho(f,z)
=
\frac{|f(z)|}
{\max\left(1,\sum_\alpha |c_\alpha z^\alpha|\right)}.
$$

This is more meaningful than using $|f(z)|$ alone when coefficients or monomials vary greatly in scale.

## Newton refinement

Marginal candidates can be refined using a precompiled numerical Jacobian:

$$
J_F(x_k)\Delta x_k=-F(x_k),
$$

followed by

$$
x_{k+1}=x_k+\Delta x_k.
$$

A step that worsens the residual can be damped, and refinement is not accepted if it produces a worse candidate.

Near a singular root, the Jacobian can itself be ill-conditioned. More precision may help distinguish numerical noise from true structure, but it cannot restore the simple-root Newton convergence theory.

## Practical advice

If a difficult system fails:

1. keep the input exact;
2. use `method="auto"` unless diagnosing a particular backend;
3. increase `digits` if the requested roots themselves need more precision;
4. increase `max_precision_digits` if the action backend needs more room to isolate roots;
5. inspect `working_digits`, `max_relative_residual`, and diagnostics;
6. consider whether the system has repeated or nearly repeated roots.

See [Multiplicity and Singular Roots](multiplicity-and-singular-roots.md) and [Troubleshooting](troubleshooting.md).
