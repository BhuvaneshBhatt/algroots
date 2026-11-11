# Changelog

## 0.4.1

Use stable CPython ABI in order to extend Python version compatibility to 3.11+.
Also fix four serious issues:
- "Real solutions" could include complex points.
- Bound symbols inside CRootOf constants caused valid systems to be rejected.
- RootOf numerical conversion and homotopy compilation could fail.
- Number-field RUR certificates could crash during construction.
- Precision scopes interfering across threads.
- Lost precision in midpoint conversion.
- A homotopy parameter name was colliding with an input variable.
- Failed field snapshots involving algebraic roots.

## 0.4.0

### Gröbner bases and solver integration

Add `groebner_basis`, which implements the F4 Gröbner-basis algorithm, returning a reduced monic SymPy `GroebnerBasis` over an exact coefficient field. Native FLINT-based computation is the default throughout Gröbner construction, solvers, quotient algebras, rational-univariate representations, certificate constructors and supporting algebraic operations, but SymPy available through explicit selection as the portable fallback. Use `backend="native"` for direct basis construction and `groebner_backend="native"` for solver and quotient interfaces.

### Native arithmetic over exact coefficient fields

Add native computation over prime fields, FLINT arithmetic over finite extension fields, and modular reconstruction over rational coefficients. Native construction also supports number fields and rational-function fields over rational or algebraic-number constants.

CRT reconstruction and parameter interpolation recover coefficients in the original field. Accelerated results do independent exact verification by default, with identity certificates available as an alternative.

### Algebraic function fields and towers

Add `AlgebraicFunctionField` for algebraic coefficents whose defining equations depend on parameters. Support extensions, primitive-element compression, exact arithmetic and univariate factorisation.

Original generator symbols remain usable through verified embedding maps. Specialisation accepts explicit algebraic root choices or enumerates compatible branches, keeping formal algebraic coefficients separate from selected numerical values.

Fix bound-symbol handling in algebraic constants during conversion, serialization, tower compression and factorization, including `CRootOf` definitions whose symbol overlap field coordinates or coefficient parameters.

### Sparse quotient algebra and RUR construction

Expand sparse exact linear algebra for quotient multiplication operators, trace pairings, Frobenius maps and rational-univariate construction.

Characteristic and minimal polynomials, norms and inverses preserve the coefficient domain and account for noncyclic and nonreduced algebras. Sparse recurrences reduce unnecessary dense matrix construction while supporting computations whose intermediate matrices become dense.

### Verification certificates

Add `certify_groebner_basis` and `certify_rational_univariate_representation`. Certificates verify ideal-membership identities, S-pair representations and coordinate maps required by RUR.

Support serialization and replay without recomputing a reference Gröbner basis. `prepare()` creates a checked, detached snapshot for repeated use, validating the ordered generators, variables, coefficient field and monomial order.

### Guarded parameter specialization

Add generic basis and RUR construction over supported parameter fields, with specialisation guards for coefficient poles, denominator failures and point collisions. Regular fibers reuse verified representations, while exceptional fibers are reconstructed from the original equations. Algebraic-tower specialisation checks the selected embedding and supports recovery from removable poles introduced by a primitive presentation through exact local regularity witnesses. These guards do not establish real sign invariance or a comprehensive decomposition of parameter space.

### Solver settings and supervised execution

Add reusable `PrecisionOptions`, `SolverLimits` and `SolverOptions` objects for precision, resource, recovery and execution settings.

Add `ExecutionLimits` and `run_supervised` for worker calculation deadlines covering computation, verification and result transfer, with optional memory limits on Linux. Improve validation and error reporting for unsupported domains, unavailable kernels, malformed inputs and exhausted budgets.

### Numerical precision and homotopy

Coordinate algroots' mpmath and FLINT precision scopes across threads, restore the preceding precision after nested calls and exceptions.

Fix ACB midpoint conversion losing requested digits when the caller's default mpmath precision is lower. Reject nonfinite midpoints with a clear error.

Prevent internal homotopy parameters from colliding with user coordinates. Reuse symbolic compilation independently of numerical precision to reduce repeated homotopy setup costs.

### Correctness and internal design

Improve coefficient-field normalisation in functional decomposition and constructive Lüroth calculations, and exact conversion of numerical endgame coordinates to rational values.

### Documentation, examples and testing

Update README, generated API references, capability matrix and limitations. Expand guides for Gröbner operations, coefficient fields, algebraic towers, sparse quotient arithmetic, reusable certificates, precision management and supervised execution.

Organize executable examples by task, update their catalog and documentation links.

Expand exact mathematical checks, adversarial tests, serialization tests and native/portable backend coverage.

## 0.3.0

### Certified rational functional decomposition

`rational_decompose` searches for two-component decompositions of univariate rational functions over a specified rational or algebraic-number constant field. It enumerates eligible products of factors of a near-separated polynomial, reconstructs the inner and outer functions, and verifies the composition identity. Results distinguish complete searches, certified indecomposability, successful decompositions, and searches stopped by candidate or degree limits. Original denominator exclusions are retained when present in the input.

### Multivariate functional decomposition

`multivariate_rational_decompose` implements the decomposition method of Gutiérrez, Rubio and Sevilla, with a univariate rational outer component and multivariate rational inner component. Candidate enumeration is lazy and degree-aware; composition identities and search-status claims are replayable. Passing a tuple or list of variables to `rational_decompose` selects this interface.

### Implicit algebraic decomposition

`algebraic_decompose` reconstructs components from supplied algebraic factors or discovers coefficient extensions and factors automatically. Discovery uses universal monic-factor algebras and Galois-orbit coverage, avoiding construction of a full splitting field in the supported cases. Certificates verify coefficient normalization and the implicit composition through resultant identities, including necessary powers.

### Recursive decomposition chains

Added APIs for rational and implicit algebraic decomposition chains, including an optional supplied initial split. Results retain composed identities, replayable split trees, and reasons for unresolved leaves. Chain completeness describes the selected decomposition tree, not enumeration of every possible tree.

### Constructive subfields and function fields

Finite quotient fields now support generated subfields, exact membership and nonmembership, primitive-element presentations, and verified embeddings. `coefficient_field` builds fields generated by specified coefficients over the declared constants, with saturated image presentations and memebership tests for supported algebraic extensions. Constructive Lüroth generators and inverse maps are available for supported univariate and multivariate rational inputs of transcendence degree at most one.

### Normalized plane curves

`algroots.curves` computes geometric genus through projective normalisation, including singularities at infinity and infinitely near points. For supported genus-zero curves, it constructs and verifies birational parametrisations, records exceptional affine charts, and distinguishes rationality over the original constant field from rationality after an explicit extension. Conic descent can identify rational-point obstructions over `QQ`.

### Rational-function coefficient domains

Finite quotient algebras accept explicit rational-function fields over rational or algebraic-number constants. Multiplication matrices, minimal and characteristic polynomials, traces, norms, inverses, and field checks preserve the coefficient domain. `specialize` reconstructs a quotient from its defining equations at parameter values.

### Finite-field quotient tools

Added finite coefficient-field constructors and univariate factorization, together with nilradical bases, reduced quotients, primary decompositions, and Frobenius operations for supported finite algebras. Nonreduced and nonmonogenic examples are covered.

### Documentation and tests

Added guides and executable examples for functional decomposition, generated function fields, plane curves, and rational-function quotient domains. Expanded regression tests.

## 0.2.0

- Faster exact solving with guarded presolve, improved variable ordering and backend selection, and Gröbner order conversion only when needed.
- Shared quotient algebra for RURs, multiplication matrices and border bases, with selective caching, lower-memory trace calculations and cost diagnostics.
- Explicit completeness evidence, radical/nonreduced metadata and multiplicities, exact root certificates and adaptive certification boxes.
- Bounded singular-root recovery through endgames, deflation and automatic projective chart switching, with clear certification limits.
- More robust small and clustered roots, bounded Arb precision retries and consistent input checks.
- Expanded tests, benchmarks, documentation; added 15 examples.
