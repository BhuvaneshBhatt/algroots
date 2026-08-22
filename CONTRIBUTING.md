# Contributing to algroots

Contributions welcome, especially reproducible bugs, difficult algebraic systems, numerical-conditioning regressions, documentation corrections, and tests that expose backend disagreement.

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

Every bugfix should include a focused regression test. New public behavior should include tests for normal inputs, boundary cases, invalid inputs, and interactions with relevant alternative backends. For mathematically equivalent representations, prefer differential or metamorphic tests that compare the resulting root sets rather than relying only on one hand-written expected ordering.

The order in which roots are returned is not guaranteed. Tests should compare root sets with a precision-appropriate tolerance unless exact ordering is itself the behavior under test.

## Documentation

Public API changes must be reflected in `docs/api.md`. New capabilities or limitations should update `docs/feature-support-matrix.md` and, where appropriate, `docs/limitations.md`. Executable examples in the README and selected documentation pages are exercised by the test suite.

## Pull requests

Before submitting a pull request, run the full tests, do Ruff linting/ formatting checks, and the coverage command above. Keep changes focused and explain any new mathematical assumptions or guarantee boundaries explicitly.

## Benchmarks

Performance checks are intentionally opt-in so routine test CI is not tied to noisy wall-clock thresholds:

```bash
python -m pip install -e ".[benchmark]"
python -m pytest benchmarks --benchmark-only
```

Use `pytest-benchmark` JSON output or saved baselines when evaluating structural changes to numerical hot paths.
