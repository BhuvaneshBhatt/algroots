# algroots documentation

`algroots` combines exact polynomial and algebraic-system root solving with structural algebraic computation: quotient algebras, functional decomposition, generated subfields, finite fields, and plane-curve geometry. The root solvers enumerate only finite complex solution sets; separate algebraic interfaces handle selected curve and function-field questions. Start with a task below to choose the appropriate guarantee level.

## Start here

1. [Quick Start](quick-start.md)
2. [Supported Problems](supported-problems.md)
3. [Feature and Support Matrix](feature-support-matrix.md)
4. [Guarantees and Result Semantics](guarantees-and-result-semantics.md)
5. [Choosing a Backend](choosing-a-backend.md)
6. [Algorithms](algorithms.md)
7. [Numerical Reliability](numerical-reliability.md)
8. [API Reference](api.md)

## Structural algebra and geometry

- **[Finite Algebras](finite-algebras.md)** — construct finite coefficient fields, detect nilpotents, form reduced quotients, and split finite algebras into primary factors.
- **[Quotient Tools](quotient-tools.md)** — finite-dimensional exact coordinate rings, traces, norms, normal forms, and specialization.
- **[Functional Decomposition](functional-decomposition.md)** — rational and implicit algebraic composition, bounded discovery, and verified recursive chains.
- **[Generated Subfields and Function Fields](function-fields.md)** — generators, exact membership, embeddings, and Lüroth maps.
- **[Plane-Curve Geometry](curves.md)** — normalized geometric genus, birational maps, and rationality over a specified coefficient field.

## Correctness and numerical behavior

- [Multiplicity and Singular Roots](multiplicity-and-singular-roots.md)
- [Singular Certification, Deflation and Chart Switching](singular-certification-and-charts.md)
- [Precision and Conditioning](precision-and-conditioning.md)
- [Branch Semantics](branch-semantics.md)
- [Completeness versus Verification](completeness-versus-verification.md)
- [Performance Model](performance-model.md)
- [Total-Degree Homotopy Continuation](homotopy-continuation.md)
- [Continuation and Monodromy](continuation-and-monodromy.md)
- [Rational Univariate Representation](rational-univariate-representation.md)
- [Exact Border Bases](exact-border-bases.md)
- [Exact Recognition](exact-recognition.md) — reconstruct and exactly certify algebraic coordinates with `algrecognize`.
- [Troubleshooting](troubleshooting.md)

## Tutorials and examples

- [Examples Gallery](examples-gallery.md)
- [End-to-End Worked Example](end-to-end-example.md)
- [Limitations](limitations.md)

The core design principle is:

> Exact algebra determines the finite solution structure; arbitrary-precision numerical algebra extracts roots efficiently; default best-effort exact recognition reconstructs and certifies algebraic coordinates afterward.

## Practical workflows and validation

- [Reading Results](reading-results.md)
- [Certification Workflow](certification-workflow.md)
- [Recovery Workflow](recovery-workflow.md)
- [Testing and Validation](testing-and-validation.md)
- [Standalone runnable examples](../examples/README.md)
