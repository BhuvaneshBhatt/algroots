# Algebraic function fields

Use an algebraic function field when a coefficient is an algebraic function of parameters, such as a formal root `a` with `a**2=t`, or when a later coefficient depends on that root. Treating `a` as an independent parameter changes the mathematics. Treating its defining relation as an ordinary coordinate equation changes the coefficient field and the quotient dimension. `AlgebraicFunctionField` keeps coefficient algebra separate from the polynomial coordinates.

## Construct a field and a tower

<!-- algroots: execute -->
```python
import sympy as sp
from algroots import AlgebraicFunctionField, QuotientAlgebra, groebner_basis

t, a, b, x, y = sp.symbols("t a b x y")
K = AlgebraicFunctionField(sp.Poly(a**2 - t, a, domain=sp.QQ.frac_field(t)))
L = K.extend(sp.Poly(b**2 - a - 1, b, domain=K))
assert L.rank == 4
assert L.convert(a**2 - t) == L.zero
assert L.convert(b**2 - a - 1) == L.zero
assert all(step.verify() for step in L.compressions)

basis = groebner_basis([x**2 - b, y - x], (x, y), domain=L, backend="sympy")
algebra = QuotientAlgebra.from_polynomials([x**2 - b, y - x], (x, y), domain=L)
assert algebra.dimension == 2
assert algebra.normal_form(x**2 - b) == 0
```

The base may be QQ, a number field, or a rational function field over either. A modulus must be univariate and irreducible over its declared base. `extend` checks irreducibility over the existing field, including cases that split after an earlier extension. Reducible defining polynomials are rejected; they describe algebras with several components, not fields.

A tower is compressed into one power basis. A squarefree Trager norm of full product degree is factored exactly over the base. Irreducibility of that norm certifies the extension. A linear gcd recovers the old and new generators in the primitive basis. Each `TowerCompression` replays the relations and the inverse primitive map. Original generator symbols remain valid inputs; the returned basis may display a private primitive symbol. `.images` records its expressions for the original generators. Native values from an unrelated field require an explicit embedding and are rejected.

`extend(..., max_degree=64, max_attempts=32)` bounds product degree and primitive-element search. `GroebnerLimitError` reports exhausted budgets. These bounds do not limit all coefficient-growth costs of exact norm factorization; use `run_supervised` with `ExecutionLimits` for a wall-clock limit on a complete operation.

## Native F4 and exact factorization

Use `groebner_basis(..., domain=L, backend="native")` or the quotient/RUR `groebner_backend="native"` option when native kernels are installed. The algorithm remains `f4_tiled_finish`. Tower degree, parameter interpolation, and native matrix budgets use `NativeF4Limits`; the same limits apply to the fixed native kernel.

At each parameter point, the adapter specializes the primitive modulus, rejects poles and ramification, factors it with FLINT, and computes in every coprime irreducible component. CRT recovers the whole power basis. Rational interpolation and lifting reconstruct coefficients in the original field. Number-field constants are merged into the primitive model with verified maps. Fresh images screen candidates; public exact verification still establishes correctness over the original field. Remote services are not involved.

For univariate factorization in a custom algebraic function field, call `L.factor_univariate(poly)`. It uses squarefree Trager norms, base-field factorization and exact gcd recovery, then checks reconstruction. Its output follows `Poly.factor_list()`. Use this method for this domain: SymPy's generic `Poly.factor_list()` does not recognize all splits in a custom finite extension. Quotient field tests and finite primary decomposition route through the correct factorizer. The exact QQ ground and modular components use FLINT where available.

## Choose a specialization branch

<!-- algroots: execute -->
```python
from algroots import certify_groebner_basis, certify_rational_univariate_representation

embedding = L.specialize({t: 4}, roots={a: 2, b: sp.sqrt(3)})
assert embedding.verify()
assert embedding(a) == 2
assert embedding(b**2) == 3

proof = certify_groebner_basis([x**2 - b, y - x], (x, y), domain=L, backend="sympy")
specialized = proof.specialize(embedding)
assert specialized.domain == sp.QQ.algebraic_field(sp.sqrt(3))

rur = certify_rational_univariate_representation([x**2 - b], (x,), domain=L)
assert rur.specialize_field(embedding).solution_count == 2
```

All parameters and original tower roots must be assigned exact algebraic constants. Relations are rechecked in the resulting number field. These choices specify an embedding; a formal root has no implicit real branch. Evaluation first recovers the original triangular power basis. At an apparent pole, an exact local colon-ideal witness can recover a regular expression on the selected branch. A genuine pole or an element with no well-defined value in that local model raises `ValueError`. A basis certificate specializes every identity and rejects poles in its evidence. A RUR certificate can rebuild the original specialized equations when its generic representation degenerates. Numeric all-roots routines require specializing first.

Semialg's `algebraic_function_fields.to_coefficient_domain(field)` converts its `RationalFunctionField` and `MonogenicFunctionField` towers into this domain, checking each extension. Pass the result to the algroots basis, quotient, and RUR APIs. Alggeom exposes the same class through `alggeom.fields.AlgebraicFunctionField`; its polynomial rings retain the declared coefficients and reject coordinate/field-generator name collisions.

## Limits

This interface supplies finite separable characteristic-zero field towers, exact coefficient arithmetic, native modular candidates, quotient/RUR operations, and replayable maps. It does not construct a comprehensive decomposition of parameter space, choose analytic continuation branches, or certify real sign invariance. Exceptional primitive models are handled through the original tower basis and exact local regularity witnesses. At a singular point, expressions with different limits on analytic branches do not have a unique value from generator values alone. Sparse quotient storage cannot prevent fill or staircase growth. Factorization and primitive-element search can dominate runtime on high-degree towers.

## Enumerate branches and repair exceptional primitive models

<!-- algroots: execute -->
```python
branches = L.specializations({t: 4})
assert len(branches) == 4
assert len(L.specializations({t: 4}, real=True)) == 2
exceptional = L.specialize(
    {t: sp.Rational(3, 4)},
    roots={a: sp.sqrt(3)/2, b: (1+sp.sqrt(3))/2},
)
assert exceptional(b) == (1+sp.sqrt(3))/2
positive = K.specialize({t: 1}, roots={a: 1})
assert positive(K.convert(1/(a+1))) == sp.Rational(1, 2)
u = sp.Symbol("u")
extended = L.with_parameters(u)
assert extended.parameters == (t, u)
assert extended.convert(b**2-a-1) == extended.zero
```

`specializations` enumerates distinct exact compatible point embeddings over the declared constant-field embedding. Repeated roots contribute one point; `max_branches` bounds the number of results. `real=True` keeps real generator choices. Enumeration does not choose an analytic continuation path or select a principal root implicitly.

`original_expression` returns coordinates in the original tower basis. If a reduced expression N/D appears singular, local evaluation computes a witness h*N = g*D modulo the tower relations with h nonzero at the selected point. This proves that g/h is a valid regular expression. It distinguishes a removable compressed-basis pole from a genuine pole on another branch. `NativeF4Limits` bounds local witness basis size, pairs and proof nodes; exhaustion raises `GroebnerLimitError` without guessing a value.

`with_parameters` adjoins independent parameters while preserving the original tower generators and compression maps. Passing `L.frac_field(u)` to a public basis, quotient or RUR constructor normalizes to the same algebraic function-field presentation. This supports localization steps that introduce parameters after algebraic extensions.

Specialization is a partial map on coefficients regular at the selected point, not a homomorphism defined on every element of a rational function field. Prefix coefficients of later defining relations use the same local regularity test. Branches where a defining coefficient has a genuine pole are excluded from enumeration; if every branch is undefined, specialization raises an error.

Bound symbols inside exact algebraic constants belong to those constants. Formal field coordinates remain independent of those names during conversion, tower compression, factorization and portable snapshot reconstruction.
