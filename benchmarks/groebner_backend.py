"""Reproducible microbenchmarks of the optional FLINT Gröbner adapter.

Run with ``PYTHONPATH=src python benchmarks/groebner_backend.py --repeats 7``.
All reference/candidate bases are checked for equality *before* the timed
region. This script is standalone and does not depend on alggeom.
"""

import argparse
import json
import platform
import statistics
from pathlib import Path
from time import perf_counter

import sympy as sp
from flint import __version__ as flint_version
from sympy.polys.rings import PolyRing

from algroots.groebner import compute_groebner_basis


def cases():
    x, y, z, w = sp.symbols("x y z w")
    return (
        ("three_variable", (x, y, z), (x * y - z, x**2 - y, y**2 - z)),
        (
            "four_variable",
            (x, y, z, w),
            (x * y - z * w, x * z - y * w, x * w - y * z, x**2 + y**2 + z**2 + w**2 - 1),
        ),
        ("nontrivial_lex", (x, y, z), (x**3 + y**2 - z, y**3 + z**2 - x, x**2 + y * z - 1)),
        ("rational_coefficients", (x, y), (x**2 / 2 + y / 3, x * y / 5 - 1)),
        ("implicit_cubic", (x, y, z), (z**2 - x**3 - y**2, x * y - z, x**2 + y**2 - 1)),
    )


def run(repeats=7, orders=("lex", "grevlex")):
    if repeats < 2:
        raise ValueError("at least two timed repetitions are required")
    records = []
    for name, variables, expressions in cases():
        for order in orders:
            ring = PolyRing(variables, sp.QQ, order)
            reference = compute_groebner_basis(expressions, ring, backend="sympy")
            candidate = compute_groebner_basis(expressions, ring, backend="flint")
            if set(reference) != set(candidate):
                raise ArithmeticError(f"basis mismatch on {name} / {order}")
            samples = {}
            for backend in ("sympy", "flint"):
                # Warm the code paths without counting compilation/caching effects.
                compute_groebner_basis(expressions, ring, backend=backend)
                samples[backend] = []
                for _ in range(repeats):
                    start = perf_counter()
                    compute_groebner_basis(expressions, ring, backend=backend)
                    samples[backend].append((perf_counter() - start) * 1000)
            ms_sympy = statistics.median(samples["sympy"])
            ms_flint = statistics.median(samples["flint"])
            records.append(
                {
                    "case": name,
                    "order": order,
                    "variables": len(variables),
                    "basis_size": len(reference),
                    "sympy_ms": ms_sympy,
                    "flint_ms": ms_flint,
                    "speedup": ms_sympy / ms_flint,
                    "samples_ms": samples,
                }
            )
    return {
        "python": platform.python_version(),
        "sympy": sp.__version__,
        "python_flint": flint_version,
        "repeats": repeats,
        "auto_dispatch": False,
        "results": records,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = json.dumps(run(arguments.repeats), indent=2)
    if arguments.output:
        arguments.output.write_text(report + "\n", encoding="utf-8")
    else:
        print(report)
