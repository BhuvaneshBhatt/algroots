# Supported Problems

`algroots` targets systems with a **finite complex solution set** after algebraization.

| Input class | Support | Notes |
|---|---|---|
| Polynomial equalities | Yes | Exact algebraic coefficients are supported. |
| Rational functions | Yes | Denominators are retained as nonzero constraints and saturated. |
| Rational powers | Yes | Includes square roots, cube roots, negative rational powers, and nested radicals. |
| Exact algebraic constants | Yes | Examples: `sqrt(2)`, algebraic `RootOf` values. |
| Complex roots | Yes | Roots are computed over the complex numbers. |
| Positive-dimensional systems | No | There is no finite all-roots list. |
| Inequalities | No | The package solves equalities, not semialgebraic feasibility regions. |
| Generic transcendental functions | No | `sin`, `exp`, `log`, etc. are outside the algebraic quotient-ring model. |

## Zero-dimensionality

For a polynomial ideal

$$I = \langle f_1,\ldots,f_m\rangle,$$

`algroots` requires the quotient

$$K[x_1,\ldots,x_n]/I$$

to be finite-dimensional. The exact Gröbner basis is used to decide this and to determine the quotient dimension.

## Principal branches

Rational powers are interpreted with SymPy's principal complex-power semantics. Polynomialization may create extra algebraic branches. Every projected candidate is therefore checked numerically against the **original** algebraic equations before it is returned.

For example,

$$\sqrt{x}=x-2$$

polynomializes to an equation with candidates `1` and `4`, but only `4` satisfies the original equation.

Similarly,

$$x^{2/3}=4$$

has multiple roots after eliminating the rational power, but the principal-power filter determines which values satisfy the actual input expression.

## Rational-function domains

Clearing denominators alone is not sufficient. If `D` is the product of retained denominators, `algroots` adds a saturation equation

$$sD-1=0$$

so denominator-zero components are removed before the zero-dimensionality test. Candidates are also checked against the retained nonzero constraints after solving.

## Explicit total-degree homotopy

`polysolve(..., method="homotopy")` supports square exact polynomial systems within the current regular finite-endpoint continuation scope. It is not an automatic fallback and does not yet provide projective or singular-endgame handling.


For the explicit total-degree backend, see [Total-Degree Homotopy Continuation](homotopy-continuation.md).
