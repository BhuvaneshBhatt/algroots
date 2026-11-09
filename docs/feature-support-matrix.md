# Feature and support matrix

A supported route specifies the input class and result contract, not a guarantee of feasible runtime for every instance. The all-roots solvers require exact coefficients and finitely many complex solutions; quotient-algebra, decomposition, finite-field, and plane-curve interfaces have their own contracts.

| Feature | Status | Route | Boundary |
|---|---|---|---|
| Exact polynomial systems | Supported | `polysolve` | Rational or exact algebraic coefficients |
| Rational functions, principal rational powers, nested radicals | Supported | `algsolve` | Original branches and retained pole constraints are checked |
| Complex numerical roots | Supported | Both solvers | Distinct metadata records; optional repeated views; canonical order with explicit evidence |
| Automatic portfolio | Supported | `method="auto"` | Guarded presolve, grevlex, shared quotient, calibrated action/RUR and exact fallbacks |
| Shape / triangular extraction | Supported | `method="shape"`, `method="triangular"` | Lex conversion on demand; shape can be refused |
| Action matrices | Supported | `method="action"` | Bounded dimension and a usable simple separator eigensystem |
| Numerical RUR | Supported | `method="rur"` | Exact representation retained; rational/algebraic fields |
| Exact RUR and algebraic points | Supported | `compute_rational_univariate_representation`, `solve_rur_representation` | Globally finite ideals |
| Quotient tools and bounded cache | Supported | `QuotientAlgebra` | Exact normal forms, actions, traces and counts |
| Finite algebras over `K(z)` | Supported, explicit domain | `QuotientAlgebra.from_polynomials(..., domain=K.frac_field(z))` | Exact element minimal/characteristic polynomials, trace, norm, unit inverse and rebuilt specialization; normalized curve genus lives in the separate curve layer |
| Sparse quotient linear algebra | Supported | `QuotientAlgebra` | Sparse actions, characteristic/minimal polynomials, traces, Frobenius and RUR recurrences; fill can become dense |
| Reusable basis and RUR certificates | Supported | `certify_groebner_basis`, `certify_rational_univariate_representation` | Exact replay or detached prepared snapshots; denominator/discriminant guards and exceptional recomputation |
| Algebraic function fields and towers | Supported, explicit domain | `AlgebraicFunctionField` | Irreducible characteristic-zero extensions, checked primitive maps, native modular images, branch enumeration and local specialization |
| Fixed-base generated subfields | Supported | `QuotientAlgebra.subfield` | Certified ambient field; exact membership, primitive element and replayable embeddings |
| Fields generated over constants | Supported, declared exact constants | `coefficient_field` | One-variable algebraic ambient presentations; rational ambient fields of transcendence degree at most one have constructive Lüroth maps |
| Multivariate Lüroth generators | Supported over rational/algebraic-number constants | `function_fields.luroth_generator` | GRS02 Algorithm 3.1, exact maps/membership and replayable decisions; higher-degree outcomes certify a lower bound of two |
| Normalized plane-curve genus | Supported, exact characteristic zero | `PlaneCurve.normalization()` | Projective/infinite/infinitely-near singularities; geometrically integral curves |
| Genus-zero birational parametrization | Supported over `QQ`; witness/extension over number fields | `PlaneCurve.rational_parametrization()` and coefficient-field methods | Conic obstructions and explicit splitting fields; general number-field point decisions require a witness |
| Rational functional decomposition | Supported, bounded automatic search | `functional_decomposition.rational_decompose` | Two-factor search completeness is explicit; one verified recursive chain can also be constructed |
| Multivariate rational functional decomposition | Supported, bounded automatic search | `functional_decomposition.multivariate_rational_decompose` or tuple/list dispatch | GRS02 Algorithm 2.1; univariate outer, inner total degree >1, exact two-factor classes over declared constants; no coordinatewise or vector-valued decomposition |
| Algebraic coefficient extensions and factors | Supported, bounded automatic discovery | `functional_decomposition.discover_algebraic_factors` | Exact monic factor orbits over the declared constants and input coefficients; binomial factor-algebra dimensions |
| Implicit algebraic functional decomposition | Automatic search or supplied candidates | `functional_decomposition.algebraic_decompose` | Minimal coefficient-field pairs; first-answer/exhaustive modes, exact resultant powers, explicit incomplete cases and verified recursive chains |
| Finite prime and extension fields | Supported | `finite_fields.finite_field`, `field_elements`, `field_order`, `factor_univariate` | Prime characteristic and exact irreducible extension modulus; univariate factorization |
| Nilradical and reduced finite algebra | Supported | `finite_decomposition.nilradical_basis`, `reduced_algebra` | Finite quotient; characteristic zero or validated finite coefficient field |
| Finite-algebra primary decomposition | Supported, exact | `finite_decomposition.primary_decomposition` | Finite quotient over characteristic-zero or validated finite field; not general positive-dimensional ideal decomposition |
| Finite-field Frobenius | Supported | `finite_decomposition.frobenius_matrix` | Validated finite coefficient field; q-power map on the quotient |
| Rational recursive decomposition chains | Supported, bounded | `functional_decomposition.rational_decomposition_chain` | Constructs and verifies one chain; not enumeration of all equivalent chains |
| Implicit algebraic decomposition chains | Supported, bounded | `functional_decomposition.algebraic_decomposition_chain` | Supplied initial splits are supported; open leaves and exhausted budgets are reported |
| Border bases | Supported | `compute_border_basis`, `compute_border_basis_linear` | Exact rational coefficients; no noisy AVI/SVD |
| Arb extraction and numerical validation | Supported | Core numerical layer | Residuals alone do not prove completeness |
| Best-effort recognition | Supported, default | Solvers / `recognize_system_roots` | Recognition does not find missing roots or certify paths |
| Global multiplicity/radical metadata | Supported | Polynomial result fields | Not generally transferred through nontrivial algebraic projections |
| Individual multiplicity and singular-root proof | Supported | Isolated-root certificates | Exact finite ideal; exact target point |
| Automatic adaptive boxes | Supported, opt-in | `certify="auto"` / `"required"` | Rational coefficients; explicit dimension/refinement/attempt limits |
| Exact local deflation | Supported, bounded | `deflate_isolated_root` | Local regularization; not a computation of all local dual bases |
| Ordinary total-degree homotopy | Explicit numerical route | `method="homotopy"` | Square systems, finite regular required endpoints; not selected by auto |
| Bounded certified endpoint recovery | Opt-in | `homotopy_recovery=True` | Rational finite square systems, sequential; proof is finite-root accounting |
| Numerical Cauchy endgame | Experimental numerical route | `cauchy_endgame` | Cycle/level budgets; optional rational endpoint proof |
| Automatic projective charts | Experimental numerical route | `track_projective_path` | No interval path tubes or certified infinity |
| Seeded monodromy | Experimental | `discover_monodromy_orbit` | Supplied seeds; no `polysolve(method="monodromy")` |
| Trace / capture-recapture stopping | Numerical / statistical | Monodromy controls | Not exact completeness evidence |
| Certified numerical paths / projective completeness | Unsupported | — | Endpoint proofs do not supply these |
| Positive-dimensional all-roots enumeration | Unsupported | `polysolve`, `algsolve` | Curve genus and parametrization are separate operations; no finite all-roots representation |
| Noisy coefficients, inequalities, generic transcendental equations | Unsupported | — | Outside the exact equality problem contract |
| Full local dual bases and arbitrary positive-dimensional primary decomposition | Unsupported | — | Finite-algebra primary decomposition is a separate supported operation |

See [Finite Algebras](finite-algebras.md), [Reading Results](reading-results.md), [Certification Workflow](certification-workflow.md), [Recovery Workflow](recovery-workflow.md) and [Limitations](limitations.md).

Gröbner construction defaults to native throughout the solver, quotient, RUR, border-basis and certificate APIs. Gröbner bases over QQ, prime and finite extension fields, algebraic number fields, and rational function fields over QQ or a number field, and certified algebraic function-field towers have explicit verified native backends. Algebraic components require the FLINT native extension. Verified specialization guards retain exceptional parameter fibers. See the [backend capability matrix](groebner-bases.md#choose-a-backend).

Generic RUR construction also accepts an explicit characteristic-zero rational function field or algebraic function-field tower. It preserves algebra length and generic geometric point counts. Exact or numeric root extraction requires specialization of the original equations; no root-count or sign-invariance claim follows from a generic Gröbner basis. See [Coefficient fields and specialization](coefficient-fields.md).
