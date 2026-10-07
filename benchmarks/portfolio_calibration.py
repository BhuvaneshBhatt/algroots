"""Small counterbalanced calibration corpus, not a universal performance claim.

Run with PYTHONPATH=src python benchmarks/portfolio_calibration.py --output FILE.
"""

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import sympy as sp

from algroots import polysolve
from algroots.cost_policy import graph_cost, ordered_variables


def calibrate():
    x, y = sp.symbols("x y")
    backends = []
    for degree in (2, 3, 4, 5, 6):
        equations = (x**degree - 2, y**degree - 3)
        samples = {name: [] for name in ("action", "rur")}
        for repetition in range(3):
            for name in ("action", "rur") if repetition % 2 == 0 else ("rur", "action"):
                started = time.perf_counter()
                result = polysolve(
                    equations, (x, y), method=name, presolve=False, recognize=False, digits=30
                )
                assert len(result.roots) == degree**2
                samples[name].append(time.perf_counter() - started)
        backends.append(
            {
                "dimension": degree**2,
                "median_seconds": {
                    name: statistics.median(values) for name, values in samples.items()
                },
            }
        )
    variables = sp.symbols("v0:5")
    rng = random.Random(7)
    corpus = []
    for _ in range(200):
        rows = []
        for i, v in enumerate(variables):
            neighbours = rng.sample([w for w in variables if w != v], rng.randint(1, 2))
            rows.append(v**2 - sum(neighbours) - (i + 1))
        baseline = ordered_variables(rows, variables, policy="frequency")
        selected = ordered_variables(rows, variables)
        if selected != baseline:
            corpus.append(tuple(rows))
        if len(corpus) == 3:
            break
    # Include a tie case: retain the existing policy when predicted cost is equal.
    corpus.append(
        tuple(v**2 - variables[(i + 1) % len(variables)] - 1 for i, v in enumerate(variables))
    )
    ordering = []
    for rows in corpus:
        entries = {}
        for name in ("frequency", "auto"):
            order = ordered_variables(rows, variables, policy=name)
            samples = []
            for _ in range(3):
                sp.core.cache.clear_cache()
                started = time.perf_counter()
                basis = sp.groebner(rows, *order, order="grevlex", domain=sp.QQ)
                samples.append(time.perf_counter() - started)
                assert basis.is_zero_dimensional
            entries[name] = {
                "order": list(map(str, order)),
                "graph_cost": graph_cost(rows, order),
                "basis_terms": sum(len(p.terms()) for p in basis.polys),
                "median_seconds": statistics.median(samples),
            }
        ordering.append({"equations": list(map(str, rows)), "policies": entries})
    return {
        "policy": "calibrated-v1",
        "scope": "small rational finite systems; no universal guarantee",
        "backends": backends,
        "ordering": ordering,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    results = calibrate()
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
