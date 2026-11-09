"""Matched opt-in FLINT and exact-verifier benchmarks.

Optionally supply --baseline FILE pointing to a previous groebner.py module.
Times exclude cold import and include conversion for each candidate call.
Run with the alggeom and algroots source directories in PYTHONPATH.
"""

import argparse
import importlib.util
import json
import platform
import statistics
import sys
from pathlib import Path
from time import perf_counter

import sympy as sp
from sympy.polys.rings import PolyRing

from algroots.groebner import compute_groebner_basis


def cases():
    x, y, z, w = sp.symbols("x y z w")
    return (
        ("medium_lex", "lex", (x, y, z), (x**3 + y**2 - z, y**3 + z**2 - x, x**2 + y * z - 1)),
        (
            "medium_grevlex",
            "grevlex",
            (x, y, z),
            (x**3 + y**2 - z, y**3 + z**2 - x, x**2 + y * z - 1),
        ),
        ("toric_grevlex", "grevlex", (x, y, z), (x * y - z, x * x - y, y * y - z)),
        ("rational_lex", "lex", (x, y), (x * x / sp.Integer(2) + y / 3, x * y / 5 - 1)),
        (
            "implicit_4_grevlex",
            "grevlex",
            (x, y, z, w),
            (x * y - z * w, x * z - y * w, x * w - y * z, x * x + y * y + z * z + w * w - 1),
        ),
    )


def median_time(action, repeats):
    timings = []
    for _ in range(repeats):
        start = perf_counter()
        action()
        timings.append((perf_counter() - start) * 1000)
    return {"median_ms": statistics.median(timings), "samples_ms": timings}


def load_baseline(path):
    spec = importlib.util.spec_from_file_location("algroots_previous_groebner", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run(repeats, baseline_path):
    baseline = load_baseline(baseline_path) if baseline_path else None
    output = {
        "python": platform.python_version(),
        "sympy": sp.__version__,
        "repeats": repeats,
        "baseline_file": str(baseline_path) if baseline_path else None,
        "records": [],
    }
    try:
        import flint

        output["python_flint"] = flint.__version__
    except ImportError:
        output["python_flint"] = "missing"
    for name, order, variables, expressions in cases():
        ring = PolyRing(variables, sp.QQ, order)
        ref = compute_groebner_basis(expressions, ring)
        candidate = compute_groebner_basis(expressions, ring, backend="flint")
        assert candidate == ref, name

        def calc(expressions=expressions, ring=ring):
            return compute_groebner_basis(expressions, ring, backend="flint")

        record = {
            "name": name,
            "order": order,
            "basis_size": len(ref),
            "optimized": median_time(calc, repeats),
        }
        if baseline:
            assert set(baseline.compute_groebner_basis(expressions, ring, backend="flint")) == set(
                ref
            )
            record["baseline"] = median_time(
                lambda expressions=expressions, ring=ring: baseline.compute_groebner_basis(
                    expressions, ring, backend="flint"
                ),
                repeats,
            )
            record["speedup"] = record["baseline"]["median_ms"] / record["optimized"]["median_ms"]
        if order == "lex":
            try:
                fglm = compute_groebner_basis(
                    expressions, ring, backend="flint", lex_strategy="fglm"
                )
                assert fglm == ref
                record["fglm"] = median_time(
                    lambda expressions=expressions, ring=ring: compute_groebner_basis(
                        expressions, ring, backend="flint", lex_strategy="fglm"
                    ),
                    repeats,
                )
            except ValueError:
                record["fglm"] = "not zero-dimensional"
        output["records"].append(record)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.repeats < 2:
        parser.error("--repeats must be >= 2")
    results = run(arguments.repeats, arguments.baseline)
    text = json.dumps(results, indent=2) + "\n"
    if arguments.output:
        arguments.output.write_text(text)
    else:
        print(text)
