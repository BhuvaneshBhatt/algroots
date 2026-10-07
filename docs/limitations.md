# Limitations

`algroots` deliberately focuses on finite algebraic solution sets. The items below are current boundaries of the implementation rather than silent approximations.

## Positive-dimensional systems

`algroots` is an all-roots solver for finite solution sets. Curves, surfaces, and other positive-dimensional varieties require a different representation and are rejected rather than converted to an incomplete finite list.

## Multiple and non-radical roots

The dense action backend expects an isolated simple separator eigensystem. Nonreduced systems can use RUR in automatic dispatch. Global multiplicity/radical metadata and exact individual multiplicities through root certificates are available. Full local dual bases and primary decomposition remain unsupported.

See [Multiplicity and Singular Roots](multiplicity-and-singular-roots.md).

## Gröbner-basis complexity

The exact Gröbner basis remains the structural front end. Some systems experience severe coefficient or monomial growth before numerical root extraction begins.

See [Performance Model](performance-model.md).

## Large quotient dimensions

The action matrix is dense and $D\times D$. `max_action_dimension` prevents unexpectedly large dense eigensystems, but the package is not currently designed as a large-scale sparse homotopy solver.

## Algebraic branch semantics

Principal rational powers are checked against the original input expression. Domain information already erased by SymPy simplification before `algroots` receives an expression cannot be reconstructed.

See [Branch Semantics](branch-semantics.md).

## Transcendental equations

Generic `sin`, `cos`, `exp`, `log`, and similar transcendental functions are outside the quotient-algebra model.

## Continuation and monodromy

The continuation/monodromy subsystem is experimental. High-level orbit discovery accepts the same supported algebraic expression subset as `algsolve`, but tracks algebraic systems through an augmented polynomial cover and still assumes nonsingular paths and supplied seed roots. Cauchy endgames with exact endpoint certificates, targeted deflation and automatic projective chart switching are available as separate expert APIs. They remain separate from monodromy. Bounded opt-in high-level homotopy recovery now orchestrates endgames, charts and certified deflation for rational finite systems. Automatic seed generation and a `method="monodromy"` route through `polysolve` remain unsupported.

Trace and capture-recapture stopping are numerical/statistical evidence. They are not exact completeness certificates. See [Continuation and Monodromy](continuation-and-monodromy.md) and [Completeness versus Verification](completeness-versus-verification.md).

## Where to go next

For practical examples, use the [Examples Gallery](examples-gallery.md) and [End-to-End Worked Example](end-to-end-example.md). For failures and configuration limits, see [Troubleshooting](troubleshooting.md).

## Total-degree homotopy

The explicit `method="homotopy"` backend currently targets regular square polynomial systems whose full total-degree path set reaches finite nonsingular endpoints. Without `homotopy_recovery=True`, it does not invoke certified endgame, deflation or projective chart APIs. Consequently, systems with multiplicities or paths to infinity can raise `HomotopySolveError` even when their finite affine roots are otherwise well defined. The backend is intentionally not selected by `method="auto"` yet.


For the explicit total-degree backend, see [Total-Degree Homotopy Continuation](homotopy-continuation.md).

Exact isolated-root and rational-box certificates, determinantal deflation and numerical automatic chart switching are now available. Certification requires a globally finite exact ideal; rational-box proofs currently require rational coefficients. Interval-certified numerical paths and projective completeness certificates are not provided. See [Singular Certification and Charts](singular-certification-and-charts.md).

In 0.5.0, RUR and shape solving share Arb extraction; guarded constant-unit polynomial substitution and opt-in automatic per-root certificates are available. Trace calculations avoid retaining all basis actions. Exact Gröbner cost, quadratic trace-pairing storage, rational-only automatic boxes, and numerical-only path tracking remain limitations.

Automatic box certification in 0.6.0 adapts proposed radii using neighbour
distances and exact separator counts, with finite attempt and refinement budgets.
It can still refuse proofs on ill-conditioned or insufficiently accurate candidates.
Bounded homotopy recovery certifies finite root accounting through original-system
quotient counts and distinct endpoint certificates; numerical paths and infinity
remain uncertified. Algebraic-field box proofs, positive-dimensional components
and unbounded all-path continuation remain outside scope.


The 0.7.0 calibration corpus is broader but still small. Diagnostic retries remain
sequential and bounded; they do not certify paths. Cancellation-aware presolve
can still decline due to intermediate work/support bounds. Exact rational rank
and bounded action reuse improve measured cases, but dense pairings remain
quadratic and Gröbner coefficient growth remains a principal bottleneck.
