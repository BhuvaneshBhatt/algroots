# Limitations

This page distinguishes the guarantees of the all-roots solvers from the specialized algebraic interfaces. A numerical approximation, a certified exact result, and an incomplete search carry different evidence.

## Polynomial and algebraic root solving

`polysolve` and `algsolve` require a globally zero-dimensional system to enumerate all complex roots. Positive-dimensional zero sets cannot be represented as a finite root list. The solvers accept exact rational or supported exact algebraic-number coefficients, not noisy floating-point data or symbolic coefficient families. General transcendental equations, inequalities, and quantified real formulas are outside their contract. Curve normalization and function fields are available through other modules and do not remove these solver restrictions.

The dense action-matrix backend requires a usable isolated simple separator eigensystem. Nonreduced quotient algebras are handled through exact alternatives such as RUR where applicable; precision alone cannot resolve a multiple root as distinct simple roots. Exact root certificates can establish local multiplicity, but the package does not provide general local dual-basis computations. See [Multiplicity and Singular Roots](multiplicity-and-singular-roots.md).

## Exact coefficient domains and specializations

`QuotientAlgebra` supports finite quotient algebras over rational and exact algebraic-number coefficient fields, explicit rational-function coefficient fields, and validated finite fields. Parameters in a rational-function coefficient field are separate from quotient variables. A specialization rebuilds the quotient from its defining equations; denominator guards do not imply that fiber dimension or multiplicities remain constant. Specializations producing positive-dimensional quotients are not accepted by the finite quotient API.

The numerical root solvers do not acquire symbolic-parameter support when the quotient API is given a rational-function domain. Finite-field algebra is available through [Finite Algebras](finite-algebras.md), while ordinary complex numerical root extraction does not operate over finite fields. In positive characteristic, finite-algebra nilradicals and primary decomposition require a **finite perfect coefficient field**, not an arbitrary imperfect field such as `GF(2)(t)`.

## Functional decomposition and function fields

[Rational and algebraic functional decomposition](functional-decomposition.md) supports exact candidate construction and verification. Automatic searches are subject to explicit degree, factor, and candidate budgets. A completed exhaustive search can support a negative conclusion; an interrupted or unresolved search cannot. Recursive chain APIs construct a verified decomposition tree from supported inputs, potentially with open leaves. They do **not** enumerate every inequivalent recursive chain.

Multivariate rational decomposition supports a univariate outer function and multivariate rational inner function; general coordinatewise or vector-valued inner decompositions are outside scope. Constructive Lüroth generators cover supported rational subfields of transcendence degree at most one; higher-degree results may certify only a lower bound, without constructing a transcendence basis. General function-field computation over arbitrary positive characteristic is not provided.

## Curves

[Plane-curve geometry](curves.md) provides projective normalization and geometric genus for supported geometrically integral plane curves in characteristic zero, together with rational parametrizations and appropriate point obstructions. Genus zero does not by itself imply a rational parametrization over the original coefficient field. General rational-point decision over arbitrary number fields may require a supplied point or explicit constant-field extension.

## Finite-algebra decomposition

`nilradical_basis`, `reduced_algebra`, and `primary_decomposition` operate on **finite-dimensional quotient algebras** over supported characteristic-zero or finite coefficient fields. The primary components preserve nilpotent structure, including nonmonogenic cases. These functions do not perform arbitrary primary decomposition of positive-dimensional polynomial ideals, nor compute full local dual bases. See [Finite Algebras](finite-algebras.md).

## Computational complexity and memory

The Gröbner basis is the structural front end for many solvers and can experience rapid monomial or coefficient growth. Numerical action matrices have quadratic storage costs in quotient dimension; exact trace pairings retain sparse support but can fill to quadratic size, and root isolation may require high Arb precision. `max_action_dimension`, cache limits, degree/term budgets, and solver precision ceilings bound work but can cause explicit refusal on difficult instances. A rejected computation is not evidence that a root or decomposition does not exist.

Presolve applies exact substitution only when guarded degree, term, and growth estimates permit it. A refused substitution leaves the original constraint for the solver; the guard does not change its mathematical meaning. Adaptive rational-box root certificates can be refused when candidate precision or isolation work is insufficient. Automatic boxes require rational coefficients and globally finite ideals, although exact-point certificates also cover supported algebraic coefficients.

Calibration and cost estimates guide backend selection, not mathematical claims or universal runtime bounds. See [Performance Model](performance-model.md) and [Precision and Conditioning](precision-and-conditioning.md).

## Numerical continuation and homotopy

Ordinary `polysolve(..., method="homotopy")` targets regular square systems whose total-degree paths reach finite nonsingular endpoints. It is not selected by automatic dispatch; multiplicities or paths to infinity can cause failure even when the finite affine roots exist. The continuation and monodromy interfaces provide numerical path/orbit evidence, not certified interval path tubes or projective completeness.

Opt-in bounded homotopy recovery uses exact quotient counts and distinct certified endpoints for **finite-root accounting**, subject to rational-coefficient, quotient-dimension, path, retry, deflation, and chart budgets. Recovery retains proved endpoints and retries unresolved paths, but a certified endpoint is not a certificate for the entire numerical path or for solutions at infinity. Numerically near-infinity outcomes are diagnostic, not proofs of divergence or projective completeness.

See [Homotopy Continuation](homotopy-continuation.md), [Recovery Workflow](recovery-workflow.md), and [Singular Certification and Charts](singular-certification-and-charts.md).

## Where to go next

For examples of supported and unsupported operations, use the [Examples Gallery](examples-gallery.md), [End-to-End Worked Example](end-to-end-example.md), [Feature Matrix](feature-support-matrix.md), and [Troubleshooting](troubleshooting.md).

## Gröbner construction

Gröbner-backed construction defaults to native throughout algroots. A compiled kernel is required when construction reaches that path; a pure Python installation needs explicit SymPy selection. Finite extension fields GF(p^k) use the FLINT extension kernel. Positive-characteristic parameter fields currently require explicit SymPy selection. Accelerated public backends use independent exact reference computation by default. This can dominate total time. `verification="certificate"` selects exact generator and S-pair identities instead, and reusable certificates avoid a new basis computation. Native limits do not bound reference verification time; certificate production has separate proof limits. Native modular adapters support QQ, prime fields, number fields, and rational function fields over QQ or number fields. Algebraic components require the FLINT native extension. Parameter-dependent algebraic function-field extensions and towers are supported through certified primitive models. Block orders are unsupported by the native adapter. See [Gröbner bases](groebner-bases.md) for the full capability matrix and supported limits.
