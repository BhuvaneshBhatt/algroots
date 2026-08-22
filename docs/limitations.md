# Limitations

`algroots` deliberately focuses on finite algebraic solution sets. The items below are current boundaries of the implementation rather than silent approximations.

## Positive-dimensional systems

`algroots` is an all-roots solver for finite solution sets. Curves, surfaces, and other positive-dimensional varieties require a different representation and are rejected rather than converted to an incomplete finite list.

## Multiple and non-radical roots

The dense action backend expects an isolated simple separator eigensystem. Multiple roots can require the triangular fallback, and multiplicity reporting is not yet a first-class result feature.

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

The continuation/monodromy subsystem is experimental. High-level orbit discovery accepts the same supported algebraic expression subset as `algsolve`, but tracks algebraic systems through an augmented polynomial cover and still assumes nonsingular paths and supplied seed roots. Singular endpoint endgames, projective path tracking, automatic seed generation, and a `method="monodromy"` route through `polysolve` are not implemented.

Trace and capture-recapture stopping are numerical/statistical evidence. They are not exact completeness certificates. See [Continuation and Monodromy](continuation-and-monodromy.md) and [Completeness versus Verification](completeness-versus-verification.md).

## Where to go next

For practical examples, use the [Examples Gallery](examples-gallery.md) and [End-to-End Worked Example](end-to-end-example.md). For failures and configuration limits, see [Troubleshooting](troubleshooting.md).

## Total-degree homotopy

The explicit `method="homotopy"` backend currently targets regular square polynomial systems whose full total-degree path set reaches finite nonsingular endpoints. It has no singular endpoint endgame, deflation, or projective path tracking. Consequently, systems with multiplicities or paths to infinity can raise `HomotopySolveError` even when their finite affine roots are otherwise well defined. The backend is intentionally not selected by `method="auto"` yet.


For the explicit total-degree backend, see [Total-Degree Homotopy Continuation](homotopy-continuation.md).
