# Rational Univariate Representation

A rational univariate representation (RUR) is an **exact representation of a finite polynomial variety**, not merely a numerical root-finding technique. `algroots` exposes the exact representation directly and also offers `method="rur"` as a numerical extraction backend.

## Exact representation

For a zero-dimensional system over an exact rational or algebraic number field, the RUR chooses a separating parameter $t$ and constructs an exact defining polynomial

$$ q(t)=0 $$

together with rational coordinate maps. After normalizing the common denominator modulo $q$, each coordinate can be represented by an exact polynomial in $t$ modulo the defining polynomial.

<!-- algroots: execute -->
```python
import sympy as sp
from algroots.rational_univariate import compute_rational_univariate_representation

x, y = sp.symbols("x y")
representation = compute_rational_univariate_representation(
    (x**2 - 2, y - x),
    (x, y),
)

assert representation.solution_count == 2
```

The constructor operates natively over `QQ` for rational coefficients and over one exact compositum algebraic-number field for exact algebraic coefficients such as `sqrt(2)`, `sqrt(3)`, or `CRootOf` constants. A coordinate symbol may also occur as the bound variable in a `CRootOf` defining polynomial; that occurrence remains part of the coefficient, not an additional unknown. Reusable RUR certificates support these number fields as well as parameter fields.

## Exact root extraction

`solve_rur_representation(representation, real=...)` enumerates exact algebraic roots of the RUR defining polynomial and evaluates the exact coordinate maps. This is appropriate when exact SymPy algebraic-number objects are the desired result, but exact root enumeration can be substantially more expensive than numerical extraction for higher degrees.

`solve_rur_points()` returns `RationalUnivariatePoint` objects that retain the parameter root and evaluate coordinates lazily from the exact representation.

Both exact extraction functions default to `real=True`, which requires every original coordinate to be real. Over a complex coefficient field, a real parameter root can encode complex coordinates, and a complex parameter can encode a real point. Filtering therefore uses the coordinate maps; `real=False` returns all geometric points. For example, the system `x - I = 0, y + I = 0` has one complex point and no real points.

## Numerical `method="rur"`

`polysolve(..., method="rur")` does **not** call exact RUR root enumeration and then convert the resulting `RootOf` objects back to floating-point numbers. Instead it:

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

The numerical backend retains the exact representation in `result.rational_univariate_representation` instead of discarding it.

## Multiplicity and geometric roots

The quotient-algebra dimension counts multiplicity, while the squarefree RUR defining polynomial represents distinct geometric parameter roots. `algroots` keeps these notions separate. `total_multiplicity` and `geometric_solution_count` report global counts. Individual multiplicities are available through exact isolated-root certificates; the numerical root list contains distinct points.

For the quotient-algebra relationship to action matrices and border bases, see [Algorithms](algorithms.md) and [Exact Border Bases](exact-border-bases.md).

## Symbolic coefficient parameters

Pass `domain=QQ.frac_field(a)` or `domain=QQ.algebraic_field(sqrt(2)).frac_field(a)` to compute a generic RUR over an exact characteristic-zero field. The auxiliary RUR variable is separate from the coefficient parameters. `groebner_backend` and `groebner_limits` choose verified basis construction in the original domain. Exact coordinate normalization uses the same bounded-unit fraction representation as quotient arithmetic, including over number fields extended by parameters.

The representation's quotient dimension records algebra length, while `solution_count` records distinct geometric points over the coefficient field's algebraic closure. Repeated roots can change both counts after specialization. Neither number gives the number of real points on every parameter fiber. Root extraction requires constant exact coefficients: specialize the original equations and rebuild the RUR. See [Coefficient fields and specialization](coefficient-fields.md#construct-a-generic-rur) for a complete example and exceptional-fiber conditions.

See [Sparse quotients and reusable certificates](sparse-quotients-and-certificates.md) for sparse exact arithmetic, proof replay, and RUR specialization guards, and [Algebraic function fields](algebraic-function-fields.md) for parameter-dependent coefficient towers.
