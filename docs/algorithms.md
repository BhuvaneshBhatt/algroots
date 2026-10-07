# Algorithms

## Pipeline

```text
original algebraic equations
        ↓
exact algebraization and domain constraints
        ↓
exact polynomial system
        ↓
guarded constant-unit polynomial presolve + deterministic variable ordering + grevlex Gröbner basis
        ↓
zero-dimensionality + quotient structure (`QuotientAlgebra`)
        ↓
auto: cheap exposed shape → bounded action → shared-quotient RUR
        ↓ if exact RUR declines
FGLM to lexicographic → triangular fallback

explicit shape/triangular: FGLM to lex on demand

explicit numerical route: total-degree homotopy
        ↓
Arb arbitrary-precision numerical roots
        ↓
compiled residual verification + Newton refinement
        ↓
original algebraic branch/domain filtering
        ↓
default best-effort algrecognize exact reconstruction
```

## Shape-position backend

When the lexicographic basis has the form

$$p(t)=0,\qquad x_i=q_i(t),$$

only the eliminant `p` needs a numerical all-roots solve. `algroots` uses FLINT/Arb `acb_poly.roots()` when available and evaluates each exact coordinate polynomial at the corresponding parameter root.

## Action-matrix backend

For a zero-dimensional ideal, let

$$A=K[x_1,\ldots,x_n]/I$$

and let $B=(b_1,\ldots,b_D)$ be the standard-monomial basis. The quotient dimension

$$D=\dim_K A$$

is exact.

Instead of constructing every coordinate multiplication matrix $M_{x_i}$, `algroots` chooses a deterministic linear form

$$L=c_1x_1+\cdots+c_nx_n$$

and constructs only $M_L$. The columns are the exact normal forms

$$\operatorname{NF}(Lb_j).$$

This requires roughly $D$ Gröbner reductions instead of $nD$. Each coordinate is reduced once:

$$x_i \equiv q_i(B) \pmod I.$$

The transpose of $M_L$ is diagonalized with `acb_mat.eig()`. For a simple isolated separator eigenvalue, its eigenvector is proportional to the evaluation vector

$$v(z)=(b_1(z),\ldots,b_D(z))^T.$$

After normalizing the constant basis component to one, each coordinate is recovered by evaluating its normal form on the same vector. This preserves coordinate pairing without constructing separate $M_{x_i}$ matrices.

### Terminology and historical references

This backend belongs to the classical **multiplication-matrix / action-matrix / endomorphism-matrix eigenmethod** family. Multiplication by a polynomial in the finite-dimensional quotient algebra defines a linear endomorphism, and the matrix representing that map is variously called a multiplication matrix, action matrix, endomorphism matrix, or (in related literature) a Stetter matrix. `algroots` uses the modern term *action matrix* in its public API, while constructing the endomorphism associated with a separating linear form rather than a separate coordinate matrix for every variable.

Useful references:

- W. Auzinger and H. Stetter, “An elimination algorithm for the computation of all zeros of a system of multivariate polynomial equations,” *International Series of Numerical Mathematics* **86** (1988), 11–31.
- H. M. Möller, “Systems of algebraic equations solved by means of endomorphisms,” in *Applied Algebra, Algebraic Algorithms, and Error-Correcting Codes*, Lecture Notes in Computer Science **673** (1993), Springer-Verlag, 43–56.
- K. Yokoyama, M. Noro, and T. Takeshima, “Solutions of systems of algebraic equations and linear maps on residue class rings,” *Journal of Symbolic Computation* **14** (1992), 399–417.
- R. M. Corless, “Editor’s corner: Gröbner bases and matrix eigenproblems,” *SIGSAM Bulletin: Communications in Computer Algebra* **30**(4) (1996), 26–32.
- D. Cox, “Introduction to Gröbner bases,” *Proceedings of Symposia in Applied Mathematics* **53** (1998), 1–24; see especially the discussion of solving zero-dimensional systems via quotient algebras.
- D. Lichtblau, “Solving finite algebraic systems using numeric Gröbner bases and eigenvalues,” in *SCI2000, Proceedings of the World Conference on Systemics, Cybernetics, and Informatics*, vol. 10 (2000), 555–560.

These references describe the same broad algebraic mechanism; they do not imply that `algroots` reproduces any one historical implementation. In particular, the current action backend uses exact quotient structure together with arbitrary-precision numerical eigensolving and recovers coordinates from normal forms using a single separator eigensystem.

## Triangular fallback

When neither shape position nor a certified simple action eigensystem is available, the triangular backend recursively solves suitable Gröbner-basis equations and back-substitutes through branches. It is slower but useful for some non-radical or structurally awkward systems.

## Exact and numerical boundary

The following remain exact:

- algebraization and saturation;
- Gröbner basis and leading ideal;
- standard monomials and quotient dimension;
- separator/coordinate normal forms.

The following are arbitrary-precision numerical operations:

- polynomial roots or action eigensystems;
- coordinate evaluation;
- residual evaluation;
- Newton refinement;
- original-expression branch filtering.

This division preserves structural completeness information while avoiding unnecessary exact algebraic-number expression growth during root extraction.

## Rational univariate representation (RUR)

`algroots` also provides an exact rational-univariate representation for zero-dimensional polynomial systems with rational or exact algebraic coefficients. Algebraic coefficients are embedded in a single compositum number field over `QQ`, and the quotient-algebra computation remains exact. Starting from a finite quotient algebra, it chooses a separating linear form, computes its squarefree defining polynomial, and expresses each original coordinate as a rational function of the separating parameter. This is closely related to the quotient-algebra/action-matrix machinery, but it retains an exact univariate representation instead of numerically diagonalizing an action matrix.

The implementation distinguishes the quotient-algebra dimension (which counts multiplicity) from the number of distinct geometric roots represented by the squarefree defining polynomial.

For numerical `method="rur"`, `algroots` does not construct exact parameter `RootOf` objects merely to convert them back to approximations. It numerically solves the exact defining univariate polynomial, evaluates the exact coordinate maps at those parameter roots, and then applies the ordinary multivariate refinement/verification pipeline. Quotient multiplication is assembled from coordinate multiplication matrices, reducing Gröbner reductions from roughly quadratic in the quotient dimension to roughly `n * dimension`; multiplication by other quotient-basis elements is obtained exactly by matrix products.

A standard reference is Fabrice Rouillier, “Solving Zero-Dimensional Systems Through the Rational Univariate Representation,” *Applicable Algebra in Engineering, Communication and Computing* 9 (1999), 433–461. See [Rational Univariate Representation](rational-univariate-representation.md) for the exact representation, exact root extraction, and numerical backend distinction.

## Exact border bases

The exact border-basis implementation represents the same finite quotient algebra by an order ideal and rewrite rules for its border monomials. It provides exact multiplication matrices and verifies their pairwise commutation. The current implementation is symbolic/exact; it is not an approximate AVI/SVD border-basis algorithm for noisy coefficients.

Border bases are related to the action-matrix roadmap because they provide another basis and reduction mechanism for the quotient algebra from which multiplication operators can be constructed. See [Exact Border Bases](exact-border-bases.md) for the public objects and exact commutation diagnostisc.


### Rational-univariate root backend

For rational- or exact algebraic-coefficient zero-dimensional polynomial systems, `method="rur"` constructs an exact rational univariate representation over `QQ` or a compositum algebraic number field. The defining univariate polynomial is solved over all complex algebraic roots, the coordinate parametrizations are evaluated at those roots, and the resulting numerical tuples are passed through the same refinement and residual verification pipeline as the other `polysolve` backends. RUR consumes the existing grevlex quotient. Automatic dispatch includes the same RUR route, including nonreduced ideals and cases exceeding the dense action budget.

## Total-degree homotopy backend

For a square polynomial system $F=(f_1,\ldots,f_n)$ with total degrees $d_i$, the explicit homotopy backend uses the start system

$$
G_i(x)=x_i^{d_i}-1,
$$

whose $\prod_i d_i$ solutions are known products of roots of unity. It tracks the gamma homotopy

$$
H_i(x,t)=(1-t)\,\gamma\,G_i(x)+t f_i(x),
$$

using the same adaptive mixed-precision predictor/corrector engine used by monodromy. Gamma values are exact deterministic complex rationals generated from `homotopy_seed`; several candidates can be retried because the gamma trick guarantees regularity only generically. Start points are generated lazily, and process-parallel tracking compiles the symbolic homotopy once per worker rather than once per path.

The ordinary homotopy backend does not compute a Gröbner basis before path tracking. Opt-in bounded recovery computes one for exact finite-root proof accounting. Every endpoint is re-refined and residual-verified against the original target equations, and a scale-normalized Jacobian check rejects singular or insufficiently resolved endpoints. All required Bézout paths must succeed; otherwise `HomotopySolveError` is raised. This is intentionally stricter than mature projective/endgame solvers but prevents incomplete results from being mislabeled as an all-roots solve.

`method="auto"` remains `shape → action → triangular`; total-degree homotopy must be requested explicitly. See [Total-Degree Homotopy Continuation](homotopy-continuation.md) for path counts, gamma retries, regularity diagnostics, and current limitations.


The current automatic cost policy retains small action solving and can favor RUR for bounded repeated-factor hints. Such hints do not prove nonradicality. Cancellation-aware presolve probes exact sparse substitutions under final and intermediate resource bounds. Quotient operations use standard-basis shortcuts, eight-action streamed reuse and rational exact-domain rank. See [Performance Model](performance-model.md).
