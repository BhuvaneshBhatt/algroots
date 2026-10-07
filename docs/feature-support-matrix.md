# Feature and support matrix

Support means there is an intentional public route; it does not promise that
every instance is computationally tractable. All polynomial inputs require exact
coefficients and a globally finite complex solution set unless explicitly noted.

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
| Border bases | Supported | `compute_border_basis`, `compute_border_basis_linear` | Exact rational coefficients; no noisy AVI/SVD |
| Arb extraction and numerical validation | Supported | Core numerical layer | Residuals alone do not prove completeness |
| Best-effort recognition | Supported, default | Solvers / `recognize_system_roots` | Recognition does not find missing roots or certify paths |
| Global multiplicity/radical metadata | Supported | Polynomial result fields | Not generally transferred through nontrivial algebraic projections |
| Individual multiplicity and singular-root proof | Supported | Isolated-root certificates | Exact finite ideal; exact target point |
| Automatic adaptive boxes | Supported, opt-in | `certify="auto"` / `"required"` | Rational coefficients; explicit dimension/refinement/attempt limits |
| Exact local deflation | Supported, bounded | `deflate_isolated_root` | Local regularization; no full primary decomposition |
| Ordinary total-degree homotopy | Explicit numerical route | `method="homotopy"` | Square systems, finite regular required endpoints; not selected by auto |
| Bounded certified endpoint recovery | Opt-in | `homotopy_recovery=True` | Rational finite square systems, sequential; proof is finite-root accounting |
| Numerical Cauchy endgame | Experimental numerical route | `cauchy_endgame` | Cycle/level budgets; optional rational endpoint proof |
| Automatic projective charts | Experimental numerical route | `track_projective_path` | No interval path tubes or certified infinity |
| Seeded monodromy | Experimental | `discover_monodromy_orbit` | Supplied seeds; no `polysolve(method="monodromy")` |
| Trace / capture-recapture stopping | Numerical / statistical | Monodromy controls | Not exact completeness evidence |
| Certified numerical paths / projective completeness | Unsupported | — | Endpoint proofs do not supply these |
| Positive/mixed-dimensional varieties | Unsupported | — | No finite all-roots representation |
| Noisy coefficients, inequalities, generic transcendental equations | Unsupported | — | Outside the exact equality problem contract |
| Full local dual bases / primary decomposition | Unsupported | — | Global and individual multiplicities do not provide full local structure |

See [Reading Results](reading-results.md), [Certification Workflow](certification-workflow.md),
[Recovery Workflow](recovery-workflow.md) and [Limitations](limitations.md).
