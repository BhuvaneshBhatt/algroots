# Sparse quotients and reusable certificates

A zero-dimensional ideal turns a polynomial problem into a finite vector space. Its multiplication operators let you compute element minimal polynomials, inverses, traces, norms, and rational univariate representations (RURs). When most products have only a few nonzero coordinates, sparse arithmetic reduces both work and memory. Reusable certificates help when the same ideal or a parameter family is checked repeatedly.

## Sparse quotient arithmetic

<!-- algroots: execute -->
```python
import sympy as sp
from algroots import QuotientAlgebra

x, z = sp.symbols("x z")
algebra = QuotientAlgebra.from_polynomials([x**64 - 2], (x,))
action = algebra.multiplication_matrix(x, representation="sparse")
assert len(action.todok()) == 64
assert algebra.minimal_polynomial(x, z) == sp.Poly(z**64 - 2, z, domain=sp.QQ)
assert algebra.normal_form(algebra.inverse(x + 1) * (x + 1) - 1) == 0
```

Variable actions are assembled from nonzero normal-form terms. Products that remain in the staircase need no polynomial reduction. Sparse Krylov iteration keeps nonzero action entries, pivot rows, and dependence witnesses; it does not expand the action to a dense list. The first dependence gives an element's minimal polynomial, including nilpotent multiplicities. Only full cyclic closure can replace a characteristic polynomial. Noncyclic separators retain the exact characteristic-polynomial route.

`multiplication_matrix(element, representation="domain")` returns an exact SymPy `DomainMatrix`. `representation="sparse"` returns a SymPy `SparseMatrix`; the default `"sympy"` chooses an expression representation. Unit inversion solves one right-hand side with sparse pivoted elimination, avoiding construction of a full inverse. A singular multiplication operator raises `QuotientAlgebraError`, even if its particular right-hand side is consistent. Arithmetic uses the declared coefficient field, including finite and algebraic function fields.

Sparse operators can fill during multiplication or elimination. Trace pairings and Frobenius operators retain sparse support in the coefficient domain. Sparse characteristic polynomials and large sparse norms use cyclic invariant quotients, including noncyclic and nonreduced algebras. RUR coordinate construction uses sparse Horner recurrences. Fill can still reach dimension-squared storage, and exact rank or polynomial factorization can dominate runtime. Numerical eigensolvers remain dense. Sparse storage does not remove the exponential growth of a quotient staircase.

## Verify a basis once, reuse its identities

<!-- algroots: execute -->
```python
import pickle
from algroots import certify_groebner_basis, groebner_basis

x, y = sp.symbols("x y")
source = (x*y - 1, y**2 - 1)
proof = certify_groebner_basis(source, (x, y))
assert proof.verify()
restored = pickle.loads(pickle.dumps(proof))
basis = groebner_basis(source, (x, y), certificate=restored)
assert basis == proof.basis
assert QuotientAlgebra.from_certificate(restored).dimension == 2
```

A `GroebnerCertificate` records both ideal inclusions and standard representations of every S-pair. Replay checks monicity, reducedness, all polynomial identities, and the strict monomial bound required by Buchberger's criterion. It performs exact arithmetic without a second Gröbner computation. Ordinary certificate uses replay the proof. For repeated use, `prepare()` constructs a detached immutable snapshot and checks it once; subsequent fibers use the specialization theorem and regularity conditions. Serialized prepared objects are checked again when loaded. The supplied source sequence, variable order, coefficient field, and term order must match. `candidate=` lets `certify_groebner_basis` check a supplied SymPy basis through identities.

`groebner_basis(..., backend="native", verification="certificate")` verifies a newly computed candidate through these identities. The default verification remains independent reference-basis comparison. `proof_limits=NativeF4Limits(...)` bounds certificate basis count, pair count, and witness nodes independently of the candidate backend. Producing membership witnesses can be expensive, so certificate verification is not guaranteed faster on a first call. Budget exhaustion raises `GroebnerLimitError`; no unchecked result is returned.

Certificates preserve exact fields and orders when pickled.

## Specialize an RUR without losing exceptional fibers

<!-- algroots: execute -->
```python
from algroots import certify_rational_univariate_representation

t = sp.Symbol("t")
family = certify_rational_univariate_representation(
    [x**2 - t, y - x], (x, y), parameters=(t,)
)
assert family.verify()
assert family.guard == t
assert family.specialize({t: 2}).solution_count == 2
assert family.specialize({t: 0}).solution_count == 1
assert family.specialize({t: 0}).dimension == 2
```

`RationalUnivariateCertificate` checks the basis certificate and exact maps in both directions between the RUR roots and the geometric points of the quotient. Nilpotence identities with exponent bounded by the quotient dimension make this work for nonradical ideals. The certificate checks the actual coordinate denominator, the inverse separator identity, the staircase, and the declared algebra dimension and geometric point count.

For rational parameter fields, the guard includes poles in the identities, the RUR discriminant and the resultant with its coordinate denominator. Thus it detects point collisions even when the Gröbner staircase stays fixed. At a regular exact algebraic parameter value, `specialize` reuses the representation. At an exceptional value, it rebuilds the original equations using the selected backend. A pole in the input coefficients is an error. A fiber that is positive-dimensional cannot be represented by a finite RUR and raises the usual quotient error. The guard is sufficient, not a minimal description of the exceptional locus, and makes no claim about real signs or the number of real roots.

`candidate=` and `basis_certificate=` allow checking an existing RUR with reusable basis evidence. For [algebraic function fields](algebraic-function-fields.md), `specialize_field(embedding)` uses explicit selected algebraic roots and recomputes a fiber if the generic RUR loses validity.

## Capability matrix

| Operation | Exact fields | Reuse boundary |
|---|---|---|
| Sparse actions, minimal polynomial, inverse | QQ, number fields, supported finite fields, rational function fields, certified algebraic function fields | Sparse fill may become dense |
| Basis certificate replay | These exact fields | Same ordered generators, field, variables, and order |
| Generic RUR certificate | Supported characteristic-zero quotient fields | Globally zero-dimensional ideal |
| Parameter RUR specialization guard | Rational functions over QQ or a number field | All parameter values required; exceptional fibers rebuilt |
| Tower basis specialization | Certified algebraic function fields | Every proof coefficient must be defined at the chosen embedding |
| Tower RUR specialization | Certified algebraic function fields | Exact root choices required; bad fibers rebuilt |
| Real sign invariance or comprehensive parameter stratification | Not supplied by these certificates | Use semialg's real elimination tools |

## Prepare once for many fibers

<!-- algroots: execute -->
```python
family = family.prepare()
assert family.specialize({t: 3}).solution_count == 2
assert family.specialize({t: 0}).solution_count == 1
basis_family = certify_groebner_basis([t*x, y-x], (x, y), domain=sp.QQ.frac_field(t)).prepare()
assert set(basis_family.specialize({t: 1})) == {x, y}
assert set(basis_family.specialize({t: 0})) == {x-y}
```

`VerifiedGroebnerFamily` and `VerifiedRationalUnivariateFamily` copy mathematical state before verification. Changes to the original certificate or its basis cannot change that snapshot. Basis materialization and regular specializations avoid replaying the generic identities. Bad proof points rebuild the original system. Preparing and loading a snapshot still cost a full verification; use it when reuse justifies that setup cost. Numerical eigensolvers, real sign invariance and comprehensive parameter decomposition are separate operations.
