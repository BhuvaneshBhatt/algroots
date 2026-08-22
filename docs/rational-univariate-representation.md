# Rational Univariate Representation

A rational univariate representation (RUR) is an **exact representation of a finite polynomial variety**, not merely a numerical root-finding technique. `algroots` exposes the exact representation directly and also offers `method="rur"` as a numerical extraction backend.

## Exact representation

For a zero-dimensional system over an exact rational or algebraic number field, the RUR chooses a separating parameter $t$ and constructs an exact defining polynomial

$$
q(t)=0
$$

together with rational coordinate maps. After normalizing the common denominator modulo $q$, each coordinate can be represented by an exact polynomial in $t$ modulo the defining polynomial.

<!-- algroots: execute -->
```python
import sympy as sp
from algroots import compute_rational_univariate_representation

x, y = sp.symbols("x y")
representation = compute_rational_univariate_representation(
    (x**2 - 2, y - x),
    (x, y),
)

assert representation.solution_count == 2
```

The constructor operates natively over `QQ` for rational coefficients and over one exact compositum algebraic-number field for exact algebraic coefficients such as `sqrt(2)`, `sqrt(3)`, or `CRootOf` constants.

## Exact root extraction

`solve_rur_representation(representation, real=...)` enumerates exact algebraic roots of the RUR defining polynomial and evaluates the exact coordinate maps. This is appropriate when exact SymPy algebraic-number objects are the desired result, but exact root enumeration can be substantially more expensive than numerical extraction for higher degrees.

`solve_rur_points()` returns `RationalUnivariatePoint` objects that retain the parameter root and evaluate coordinates lazily from the exact representation.

## Numerical `method="rur"`

`polysolve(..., method="rur")` deliberately does **not** call exact RUR root enumeration and then convert the resulting `RootOf` objects back to floating-point numbers. Instead it:

1. constructs the exact RUR;
2. solves the exact defining univariate polynomial numerically at the requested working precision;
3. evaluates the exact coordinate maps numerically;
4. Newton-refines against the original multivariate equations;
5. applies the same scale-normalized residual verification used by other backends.

<!-- algroots: execute -->
```python
import sympy as sp
from algroots import polysolve

x = sp.symbols("x")
result = polysolve(
    (x**5 - sp.sqrt(2),),
    (x,),
    method="rur",
    digits=35,
    recognize=False,
)

assert len(result.roots) == 5
assert result.rational_univariate_representation is not None
```

Thus these are three distinct operations:

| Operation | Output | Purpose |
|---|---|---|
| `compute_rational_univariate_representation` | exact `RationalUnivariateRepresentation` | structural representation of the finite variety |
| `solve_rur_representation` | exact algebraic coordinate tuples | exact root extraction from an existing RUR |
| `polysolve(..., method="rur")` | verified arbitrary-precision numerical roots plus retained exact RUR | efficient numerical all-roots backend |

The numerical backend retains the exact representation in `result.rational_univariate_representation` rather than discarding it.

## Multiplicity and geometric roots

The quotient-algebra dimension counts multiplicity, while the squarefree RUR defining polynomial represents distinct geometric parameter roots. `algroots` keeps these notions separate. First-class multiplicity reporting is still outside the current numerical result API.

For the quotient-algebra relationship to action matrices and border bases, see [Algorithms](algorithms.md) and [Exact Border Bases](exact-border-bases.md).
