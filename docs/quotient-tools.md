# Exact quotient tools

`QuotientAlgebra` is the shared finite algebra behind numerical action solving, RUR and border bases. It accepts rational or exact algebraic coefficients, or an explicit rational-function coefficient field over those constants, checks zero-dimensionality, and constructs a standard-monomial basis. Its dimension counts multiplicity; its exact trace-pairing rank gives the geometric point count.

<!-- algroots: execute -->
```python
import sympy as sp
from algroots.quotient import QuotientAlgebra

x = sp.Symbol("x")
quotient = QuotientAlgebra.from_polynomials((x**3,), (x,))
assert quotient.dimension == 3 and quotient.geometric_solution_count == 1
assert quotient.normal_form(x**5 + x) == x
assert quotient.coordinate_vector(x) == sp.Matrix([0, 1, 0])
```

`from_groebner_basis` reuses an existing exact basis. `variable_multiplication_matrices` represent coordinate multiplication; `multiplication_matrix(element)` handles arbitrary quotient elements. `trace_vector` and `trace_pairing` are exact and stream basis actions without retaining a cubic collection. The expert `basis_multiplication_matrices` property explicitly materializes that collection; request it only when its memory cost is acceptable.

## Cache and cost controls

`normal_form_cache_size=128` is the default; zero disables caching. Admission bounds expression complexity and rational coefficient bit size. LRU eviction bounds entries. `normal_form_cache_info` returns a copy of hit/miss/eviction, capacity/entry and reduction-time statistics. `clear_normal_form_cache()` resets entries and counters. The cache belongs to one quotient and does not cross ideals.

`operation_diagnostics` returns observed timings/counters for coordinate basis shortcuts, variable actions, streamed reuse, trace calculations and rank. Standard basis coordinates avoid reductions. Temporary streamed action reuse holds at most eight matrices. Exact rational-domain rank is selected for rational coefficients; algebraic coefficients retain the profiled expression-rank route. These choices do not change the exact rank contract.

## Representations and scope

[RUR](rational-univariate-representation.md) produces exact algebraic points; [border bases](exact-border-bases.md) currently have a rational-coefficient contract. Neither is an approximate representation for noisy inputs. Large exact Gröbner bases and dense trace-pairing rank/storage remain expensive.

See [quotient example](../examples/04_quotient_and_cache.py), [border example](../examples/10_border_bases.py) and [Performance Model](performance-model.md).

## Finite algebras over rational-function fields

<!-- algroots: execute -->
```python
import sympy as sp
from algroots.quotient import QuotientAlgebra

a, z = sp.symbols("a z")
T = sp.Symbol("T", real=True)
Kz = sp.QQ.frac_field(z)
A = QuotientAlgebra.from_polynomials([a**3 + z*a + 1], [a], domain=Kz)
assert A.minimal_polynomial(a, T) == sp.Poly(T**3 + z*T + 1, T, domain=Kz)
assert A.trace(a + 1) == 3 and A.norm(a + 1) == z
inverse = A.inverse(a + 1)
assert A.normal_form((a + 1)*inverse - 1) == 0
assert A.specialization_guard(inverse).subs(z, 0) is sp.false
```

The same interface accepts `QQ.algebraic_field(sqrt(2)).frac_field(z)`. `multiplication_matrix(a, representation="domain")` keeps matrix arithmetic in the coefficient domain. This need not be a native FLINT matrix for `K(z)`. Minimal polynomials accept short cyclic closure, while characteristic polynomials and norms use the full quotient dimension, including nilpotents. Finite-algebra construction does not assert irreducibility: `is_field` performs that certification and `inverse` rejects nonunits.

`specialize({z: 0})` reconstructs a new quotient from its defining equations. For `[z*a - 1]` it returns the zero algebra; for `[z*a]` the specialized fiber is positive-dimensional and raises. For `[a**2-z]` it recomputes multiplicity at zero. `specialization_guard` checks only denominators of generic representatives and is not a promise that ranks, dimensions or real branches survive specialization.
