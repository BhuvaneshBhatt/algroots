# Gröbner kernel design

The [Gröbner guide](groebner-bases.md) covers mathematical results, coefficient fields, budgets and verification. The public default is native `f4_tiled_finish`; `backend="sympy"` provides the portable path.

## Native computation

The prime-field engine interns exact exponent vectors within each call. Packed comparison keys apply only when all coordinates fit; overflow and dimensions above eight use the retained exponent vectors. Monomial IDs never cross computation boundaries.

Critical pairs are processed in increasing LCM degree. Product and prospective chain criteria prune pairs only when their replacement obligations justify the same ideal. Symbolic preprocessing computes full support closure before row assembly. Sparse pivots and fixed 64-column tiles support sparse and dense work rows without changing their arithmetic or order. Every emitted row is reduced exactly against the current basis before insertion; an echelon row alone does not establish that invariant. Final reduction retains a minimal leading ideal and checks reduced tails.

Rational computation combines independent prime images with CRT and rational reconstruction. Number fields and parameter-dependent algebraic towers use FLINT coefficient arithmetic and checked reconstruction in the original field. Validated finite fields GF(p^d) run directly in the same FLINT coefficient kernel, preserving their supplied irreducible modulus. Finite-extension arithmetic has its own coefficient-specific row operations; it is not a user-selectable basis algorithm. Every modular coefficient and pivot is computed in its current field.

## Verification and expert interfaces

`groebner_basis` returns a verified SymPy `GroebnerBasis`. Independent exact reference comparison is the default verification; `verification="certificate"` checks both ideal inclusions and S-pair identities. Reusable prepared certificates retain their exact-system and specialization checks.

`algroots.groebner.compute_groebner_basis` accepts a `PolyRing`. `native_prime_basis` returns its reduced field basis and operation statistics. `native_modular_basis` returns reconstruction metadata and an explicit certification status; `certify=False` is an unverified candidate interface. These interfaces offer operation budgets without algorithm or storage-policy selectors.

## Building and measuring

Build native wheels with `ALGROOTS_BUILD_NATIVE_F4=1 ALGROOTS_BUILD_FF_SPARSE=1 ALGROOTS_BUILD_NATIVE_EXTENSION=1` and a compatible `ALGROOTS_FLINT_PREFIX`. The build and repair workflows bundle the required runtime libraries. Pure wheels support explicit SymPy computation.

`collect_batch_metrics=True` records pair counts, LCM degrees, matrix rows and columns, basis growth, and symbolic, elimination and insertion times. The fixed-kernel [benchmark driver](../benchmarks/bench_cross_cas.py) compares identical inputs with independent engines. Kernel-only time excludes reference or certificate verification; public end-to-end time includes it.

The native CI profile runs the complete algroots suite. The pure Python profile runs explicit SymPy construction and proof contracts, including rejection of missing native defaults. It does not claim to run workflows that omit backend selection and require compiled kernels. `python tools/run_test_profile.py --native` and `python tools/run_test_profile.py` reproduce these scopes.
