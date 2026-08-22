# Benchmarks

The benchmark suite is opt-in so normal unit-test CI does not become timing-sensitive.
Install it with `python -m pip install -e ".[benchmark]"`, then run:

```bash
python -m pytest benchmarks --benchmark-only
```

The suite covers representative shape-position solving, action-matrix solving, and continuation path tracking. It deliberately contains no absolute wall-clock pass/fail thresholds: hardware and CI noise make those brittle. `pytest-benchmark` can save and compare JSON baselines when evaluating a structural change.
