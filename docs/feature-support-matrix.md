# Feature and Support Matrix

This matrix is a compact statement of the current public support boundary. “Supported” means the package has an intentional public route for that input or operation; it does not imply that every mathematically valid instance is computationally easy.

| Problem or feature | Status | Primary route | Important notes |
|---|---|---|---|
| Exact polynomial systems | Supported | `polysolve` | Requires a zero-dimensional complex solution set. |
| Rational-function equations | Supported | `algsolve` | Denominator nonzero constraints are retained and saturated. |
| Principal rational powers and radicals | Supported | `algsolve` | Polynomial candidates are filtered against original principal-branch semantics. |
| Exact algebraic coefficients | Supported | Both solvers | Includes exact algebraic constants representable by SymPy. |
| Complex roots | Supported | Both solvers | Core solving is over the complex numbers. |
| Shape-position backend | Supported | `method="shape"` | Useful when a separating parameter gives a univariate eliminant. |
| Action-matrix backend | Supported | `method="action"` | Dense quotient-algebra backend; bounded by `max_action_dimension`. |
| Triangular fallback | Supported | `method="triangular"` | Important for some non-radical or non-simple action cases. |
| Rational-univariate backend | Supported | `method="rur"` | Exact RUR construction for rational or exact algebraic coefficients using native number-field arithmetic, followed by verified numerical evaluation. |
| Total-degree homotopy | Supported (explicit, limited scope) | `method="homotopy"` | Regular square polynomial systems whose Bézout paths all terminate at finite nonsingular endpoints; not part of `auto`. |
| Automatic backend routing | Supported | `method="auto"` | Selects among implemented core backends. |
| Rational univariate representation | Supported | `compute_rational_univariate_representation` | Exact zero-dimensional representation over `QQ` or an exact algebraic number field, with distinct-root extraction. |
| Exact border basis | Supported | `compute_border_basis` | Gröbner-derived and Macaulay linear-algebra construction; not approximate AVI/SVD. |
| Arbitrary-precision numerical extraction | Supported | Core numerical layer | Working precision may increase when conditioning requires it. |
| Numerical residual verification | Supported | Core validation layer | Establishes numerical consistency of returned candidates, not completeness by itself. |
| Default best-effort exact recognition | Supported | public solvers / `recognize_system_roots` | Solvers attempt certified recognition through `algrecognize` with degree bound 8 by default; `recognize=False` disables it, while `recognize_system_roots` exposes custom bounds. |
| Multiplicity reporting | Partial | Core solvers | Distinct roots are returned; multiplicity is not a first-class result field. |
| Singular/multiple-root solving | Partial | Core solvers | Some cases use fallback routes; not every singular system is equally robust. |
| Positive-dimensional varieties | Not supported | — | There is no finite all-roots list. |
| Inequalities / semialgebraic regions | Not supported | — | Equalities only. |
| Generic transcendental equations | Not supported | — | `sin`, `exp`, `log`, etc. are outside the algebraic model. |
| Predictor/corrector path tracking | Experimental | `track_path` | Intended for nonsingular paths; adaptive step size and precision are implemented. |
| Closed-loop monodromy permutations | Experimental | `monodromy_permutation` | Requires supplied roots/seeds; ambiguous endpoint matches are left unmatched. |
| Monodromy orbit discovery | Experimental | `discover_monodromy_orbit` | Accepts the same supported polynomial/algebraic expression subset as `algsolve`; algebraic tracking uses an augmented polynomial cover and filters returned roots against original branch/domain semantics. |
| Exact-count monodromy stopping | Experimental | `expected_root_count` | Exact only when the supplied count is independently exact. |
| Trace-test stopping | Experimental | `trace_test=True` | Numerical evidence, not an exact completeness certificate. |
| Capture-recapture stopping | Experimental | `statistical_stop=True` | Statistical evidence, not an exact completeness certificate. |
| Singular endpoint endgames | Not implemented | — | Future continuation work. |
| Projective path tracking | Not implemented | — | Future continuation work. |
| Automatic monodromy seed generation | Not implemented | — | Core solver and experimental monodromy remain separate subsystems. |
| `polysolve(method="monodromy")` | Not implemented | — | Monodromy is not currently an alternate core backend. |

For the underlying mathematical conditions, see [Supported Problems](supported-problems.md), [Guarantees and Result Semantics](guarantees-and-result-semantics.md), and [Limitations](limitations.md).


For the explicit total-degree backend, see [Total-Degree Homotopy Continuation](homotopy-continuation.md).
