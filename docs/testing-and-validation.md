# Testing and validation

Tests should check mathematical/evidence contracts, rather than reproduce implementation choices or rely on root ordering, backend timing or a printed representation. A proof-budget failure must never be accepted as certification.

## Run the checks

From an installed development tree:

```bash
python -m pytest
python -m pytest tests/test_runnable_examples.py tests/test_documentation_examples.py tests/test_documentation_consistency.py
python examples/run_all.py
python -m ruff check .
python -m ruff format --check .
```

The examples are standalone public-API programs. Tests run each in a fresh process from a temporary working directory, assert success and enforce a timeout. Their mathematical assertions are executed; output snapshots are avoided. The catalog/runner tests catch missing examples and invalid selections.

Marked Markdown blocks execute in their document's tutorial namespace. Selected new reference examples additionally execute independently so copy/paste use is checked. Unmarked Python fences are syntax-checked; deliberate failures must be handled inside runnable examples. API signature/default checks and Markdown navigation tests prevent stale parameters and broken local links.

## Current mathematical coverage

- Differential oracles compare root sets across backends and cached/uncached exact operations.
- Metamorphic tests cover permutations, rational scaling/recombination, symbol changes and presolve reconstruction.
- Dense trace/rank and exact RUR/border identities validate optimized quotient operations.
- Certificate replay/tampering tests distinguish exact membership, isolation, rank and multiplicity.
- Clustered roots, nonreduced ideals and algebraic fields exercise precision and coefficient-domain boundaries.
- Structural resource guards test work/storage/degree/term limits without wall-clock assertions.
- Recovery tests preserve proofs, reuse extraction, retry only unresolved paths and refuse incomplete results.

The release test report records exact counts, dependency versions and limitations. Coverage percentage measures executed code, not mathematical soundness.

## Structural-algebra regression coverage

- `tests/test_finite_fields.py` checks finite-field cardinalities and element arithmetic, univariate factorization, nilradical dimensions, finite-algebra primary factors, and rejection of imperfect coefficient fields.
- `tests/test_decomposition_chains.py` checks exact recursive composition identities and preservation of unresolved leaves when searches reach their limits.
- `tests/test_functional_decomposition.py` and `tests/test_multivariate_decomposition.py` check compositional reconstruction, polynomial and rational domains, candidate bounds, and certificate replay.
- `tests/test_coefficient_fields.py`, `tests/test_quotient_function_fields.py`, and `tests/test_luroth.py` exercise generated fields, exact membership, quotient specialization, and rational subfield maps.
- `tests/test_curves.py` checks normalization, genus, parametrization, and ground-field rationality obstructions on supported curves.

The examples demonstrate public use; the tests probe exact invariants and failure behavior. Neither source is a substitute for testing the built wheel on the supported Python and dependency matrix.

## Highest-value next additions

| Priority | Addition | Why it helps |
|---|---|---|
| 1 | Run the configured wheel/sdist example CI and add minimum/latest dependency combinations | Detect missing package data/dependencies and source-path assumptions |
| 2 | Add minimum/latest SymPy/python-flint jobs to the existing Python/platform matrix | Exercise domain and numerical-backend compatibility |
| 3 | Bounded generated finite systems with independent exact counts | Broaden differential coverage beyond hand-picked examples |
| 4 | Proof-kernel mutation testing and additional serialized/tampered inputs once a format exists | Check that evidence depends on all claimed invariants |
| 5 | Larger reproducible benchmark families with recorded structural metrics | Calibrate dispatch without noisy timing gates |

The existing CI already configures Python 3.11–3.14, Linux/macOS/Windows and wheel/sdist smoke checks. The configured wheel and source-distribution jobs should run the full example catalog and check that tutorial files are packaged. Verify their current CI status before relying on a release. Noisy coefficients, positive-dimensional solving and certified numerical paths need new functionality before tests can claim their support.

## Benchmarks

The [performance model](performance-model.md) describes the calibration drivers. Run `benchmarks/expanded_calibration.py` and `benchmarks/quotient_profile.py` with their `--output` controls; the latter also takes `--profile`. They retain exact checks and explicit failures. Compare counterbalanced medians and original inputs; do not infer universal speed guarantees from a small local corpus.
