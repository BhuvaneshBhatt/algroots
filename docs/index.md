# algroots documentation

`algroots` is an all-roots solver for supported **exact, zero-dimensional algebraic equation systems**. The core solver combines exact finite-dimensional structure with arbitrary-precision numerical extraction and verification; the continuation/monodromy subsystem has a separate experimental guarantee level. It combines exact algebraic preprocessing with arbitrary-precision numerical root extraction and default best-effort exact algebraic recognition.

## Start here

1. [Quick Start](quick-start.md)
2. [Supported Problems](supported-problems.md)
3. [Feature and Support Matrix](feature-support-matrix.md)
4. [Guarantees and Result Semantics](guarantees-and-result-semantics.md)
5. [Choosing a Backend](choosing-a-backend.md)
6. [Algorithms](algorithms.md)
7. [Numerical Reliability](numerical-reliability.md)
8. [API Reference](api.md)

## Correctness and numerical behavior

- [Multiplicity and Singular Roots](multiplicity-and-singular-roots.md)
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
