# Contributing to algroots

Contributions are welcome: bug reports, difficult algebraic systems, numerical-conditioning regressions, documentation corrections, tests that expose backend disagreement, code improvements or new capabilities.

## Development setup

Use Python 3.11 or newer and install the package with its test and development dependencies:

```bash
python -m pip install -e ".[test,dev]"
```

Run the full test suite with:

```bash
python -m pytest
```

Run the release-quality static checks with:

```bash
python -m ruff check .
python -m ruff format --check .
```

Coverage can be inspected with:

```bash
python -m pytest --cov=algroots --cov-branch --cov-report=term-missing
```

## Tests

Every bug fix should include a focused regression test. New public behavior should include tests for normal inputs, boundary cases, invalid inputs, and interactions with relevant alternative backends. For mathematically equivalent representations, prefer differential or metamorphic tests that compare the resulting root sets, not relying only on one handwritten expected ordering.

Tests should compare root sets with a precision-appropriate tolerance unless ordering is the behavior under test. Ordering tests must inspect `result.ordering`: certified canonical order follows the supplied variable order; numerical fallback can change at near-ties.

## Documentation

Public API changes should be reflected in `docs/api.md`. New capabilities / limitations should update `docs/feature-support-matrix.md` / `docs/limitations.md`. Executable examples in the README and selected documentation pages are exercised by the test suite.

## Pull requests

Before submitting a pull request, run the full tests, Ruff linting/format checks, the runnable examples, and relevant coverage diagnostics. Keep changes focused and explain any new mathematical assumptions or guarantee boundaries explicitly.

## Benchmarks

Performance checks are intentionally opt-in so routine test CI is not tied to noisy wall-clock thresholds:

```bash
python -m pip install -e ".[benchmark]"
python -m pytest benchmarks --benchmark-only
```

Use `pytest-benchmark` JSON output or saved baselines when evaluating structural changes to numerical hot paths.

## Mutation testing

A scheduled, non-blocking GitHub Actions job mutates the correctness-sensitive continuation, monodromy, stopping, and recognition modules. To reproduce it locally on a fork-capable platform:

```bash
python -m pip install -e ".[test,mutation]"
mutmut run "algroots.recognition*"
```

Mutation results are diagnostic rather than a release gate; surviving mutants should be reviewed for meaningful assertion gaps before adding tests mechanically.

The [testing guide](docs/testing-and-validation.md) describes evidence contracts, executable examples and remaining validation priorities. New public workflows should update the standalone examples catalog, support matrix and workflow guides.
