"""Counterbalanced exact finite-system portfolio corpus.

Run with PYTHONPATH=src python benchmarks/expanded_calibration.py --output FILE.
Failed explicit backends are retained in the report, never counted as fast solves.
"""

import argparse
import json
import statistics
from pathlib import Path
from time import perf_counter

import sympy as sp

from algroots import PolynomialSystemError, polysolve
from algroots.quotient import QuotientAlgebra


def corpus():
    x, y, z = sp.symbols("x y z")
    return [
        ("separable-4", "separable", (x**2 - 2, y**2 - 3), (x, y)),
        ("separable-16", "separable", (x**4 - 2, y**4 - 3), (x, y)),
        ("separable-36", "separable", (x**6 - 2, y**6 - 3), (x, y)),
        ("univariate-8", "univariate", (x**8 - x - 1,), (x,)),
        ("univariate-16", "univariate", (x**16 - x - 1,), (x,)),
        ("shape-chain", "shape", (x - y**2, y - z**2, z**3 - 2), (x, y, z)),
        ("shape-permuted", "shape", (x - y**2, y - z**2, z**3 - 2), (z, y, x)),
        ("sparse-cycle", "sparse", (x**2 - y, y**2 - z, z**2 - x - 1), (x, y, z)),
        ("dense-coupled", "coupled", (x**2 + y**2 - 3, x * y - 1), (x, y)),
        ("algebraic-separable", "algebraic", (x**2 - sp.sqrt(2), y**2 - 3), (x, y)),
        ("algebraic-shape", "algebraic", (x - y**2, y**3 - sp.sqrt(2)), (x, y)),
        ("nonreduced-separable", "nonreduced", (x**3, y**2), (x, y)),
        ("nonreduced-coupled", "nonreduced", ((x - y) ** 2, y**2 - 1), (x, y)),
        ("nonreduced-hidden", "nonreduced", (x**2 - y, y**2), (x, y)),
        ("clustered", "clustered", ((x - 1) * (x - 1 - sp.Rational(1, 10**12)), y - x), (x, y)),
        ("large-coefficients", "coefficient_growth", (x**3 - 10**30 * y, y**2 - 2), (x, y)),
        (
            "rational-denominators",
            "coefficient_growth",
            (x**3 - y / sp.Integer(1000003), y**2 - sp.Rational(2, 1000033)),
            (x, y),
        ),
    ]


def calibrate(repetitions=3):
    report = []
    for name, family, equations, variables in corpus():
        quotient = QuotientAlgebra.from_polynomials(equations, variables)
        expected = quotient.geometric_solution_count
        samples = {backend: [] for backend in ("action", "rur", "auto")}
        failures = {backend: [] for backend in samples}
        roots = {}
        selected_methods = {backend: set() for backend in samples}
        phases = {backend: [] for backend in samples}
        for repetition in range(repetitions):
            sequence = ("action", "rur", "auto")
            for backend in sequence[repetition % 3 :] + sequence[: repetition % 3]:
                sp.core.cache.clear_cache()
                started = perf_counter()
                try:
                    result = polysolve(
                        equations,
                        variables,
                        method=backend,
                        presolve=False,
                        variable_order="input",
                        recognize=False,
                        digits=35,
                    )
                except PolynomialSystemError as exc:
                    failures[backend].append(type(exc).__name__ + ": " + str(exc))
                    continue
                assert len(result.roots) == expected, (name, backend)
                samples[backend].append(perf_counter() - started)
                phases[backend].append(dict(result.cost_diagnostics.phase_seconds))
                roots[backend] = result.roots
                selected_methods[backend].add(result.method)
        for backend, points in roots.items():
            if backend == "rur":
                continue
            # Every numeric point must match the independent explicit RUR solve.
            for root in points:
                assert (
                    min(
                        max(abs(complex(a - b)) for a, b in zip(root, other, strict=True))
                        for other in roots["rur"]
                    )
                    < 1e-20
                ), name
        report.append(
            dict(
                name=name,
                family=family,
                equations=list(map(str, equations)),
                variables=list(map(str, variables)),
                dimension=quotient.dimension,
                geometric_count=expected,
                domain=str(quotient.domain),
                median_seconds={k: statistics.median(v) if v else None for k, v in samples.items()},
                failures=failures,
                phase_samples=phases,
                selected_methods={k: sorted(v) for k, v in selected_methods.items()},
            )
        )
        print(name, report[-1]["median_seconds"], flush=True)
    return dict(
        policy="calibrated-v2",
        repetitions=repetitions,
        scope="17 finite exact cases across nine families; local medians, no universal speed guarantee",
        cases=report,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("repetitions must be positive")
    args.output.write_text(json.dumps(calibrate(args.repetitions), indent=2) + "\n")
