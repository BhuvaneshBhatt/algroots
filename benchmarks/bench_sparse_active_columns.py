"""Standalone wide-sparse-matrix comparison; native extension optional.

No pass/fail time thresholds: every result is compared with the Python
reducer and its replayable elimination operations.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import statistics
import sys
from pathlib import Path
from time import perf_counter

from algroots.ff_sparse_matrix import compiled_available, reduce_rows


def benchmark(*, seed=42, nrows=220, ncols=150000, pool=360, width=6, prime=1009, repeats=5):
    rng = random.Random(seed)
    columns = rng.sample(range(ncols), pool)
    rows = [{k: rng.randrange(1, prime) for k in rng.sample(columns, width)} for _ in range(nrows)]
    modes = [("python", "adaptive")]
    if compiled_available():
        modes.extend([("compiled", "dense"), ("compiled", "sparse"), ("compiled", "adaptive")])
    results = []
    expected = reduce_rows(rows, ncols, prime, backend="python")
    for backend, mode in modes:
        timings = []
        for _ in range(repeats):
            start = perf_counter()
            output = reduce_rows(rows, ncols, prime, backend=backend, active_columns=mode)
            timings.append(perf_counter() - start)
            if output != expected:
                raise AssertionError(f"row reductions differ in {backend}/{mode}")
        results.append(
            {
                "backend": backend,
                "mode": mode,
                "samples_seconds": timings,
                "median_seconds": statistics.median(timings),
            }
        )
    return {
        "matrix": {
            "nrows": nrows,
            "ncols": ncols,
            "pool": pool,
            "nnz_per_row": width,
            "prime": prime,
            "seed": seed,
        },
        "python": sys.version,
        "platform": platform.platform(),
        "compiled_available": compiled_available(),
        "results": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="bench_sparse_active_columns.json")
    args = parser.parse_args()
    result = benchmark()
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    for run in result["results"]:
        print(run["backend"], run["mode"], round(run["median_seconds"] * 1000, 3), "ms")
