# Numerical Reliability

## Soundness and completeness are separate

`algroots` checks two different properties.

**Soundness:** every returned tuple satisfies the original equations to the requested verification threshold.

**Completeness:** where the exact quotient structure supplies an expected count, the numerical backend must retain the expected number of distinct roots. A list of highly accurate roots is not considered complete merely because every listed root has a small residual.

## Adaptive precision

The action backend starts from the requested working precision. If FLINT cannot rigorously isolate a simple eigensystem, or if the resulting coordinates fail original-system residual validation, the solver retries at higher precision, doubling the working precision until success or `max_precision_digits` is reached.

The exact quotient data are reused during these retries.

## FLINT/Arb

The normal numerical backend uses python-flint complex-ball arithmetic:

- `acb_mat.eig()` for action matrices;
- `acb_poly.roots()` for shape-position eliminants.

The action backend relies on FLINT's isolated simple eigensystem rather than on a manually chosen floating-point eigenvalue-distance threshold.

## Residual evaluation

Polynomial equations are compiled once into sparse coefficient/exponent tuples. Repeated verification therefore avoids repeated SymPy `Poly` construction and symbolic substitution.

A scaled relative residual is used:

$$
\rho(f,z)=\frac{|f(z)|}{\max(1,\sum_\alpha |c_\alpha z^\alpha|)}.
$$

## Newton refinement

The polynomial Jacobian is differentiated and compiled once. Marginal candidates are refined using reusable arbitrary-precision Newton iterations. For overdetermined systems a least-squares solve is used. A Newton step that worsens the residual is damped, and a refinement is never accepted if it is worse than the input candidate.

## Conditioning

Arbitrary precision does not make an ill-conditioned problem well-conditioned. Increasing precision can reveal separated roots that were hidden by insufficient arithmetic precision, but nearly singular roots, multiple roots, and poorly conditioned quotient algebras can still require substantially more work or fall back to a different backend.

## Related guides

- [Guarantees and Result Semantics](guarantees-and-result-semantics.md) defines verification, quotient dimension, and exact certification.
- [Precision and Conditioning](precision-and-conditioning.md) gives a worked comparison of separated and nearly colliding roots.
- [Completeness versus Verification](completeness-versus-verification.md) explains why residuals cannot prove that no roots were missed.
- [Multiplicity and Singular Roots](multiplicity-and-singular-roots.md) covers the important non-simple-root caveat.
