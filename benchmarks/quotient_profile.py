"""Profile exact quotient stages and compare exact rank implementations."""

import argparse
import cProfile
import hashlib
import io
import json
import pstats
from pathlib import Path
from time import perf_counter

import sympy as sp

from algroots.quotient import QuotientAlgebra


def measure(function):
    start = perf_counter()
    value = function()
    return value, perf_counter() - start


def profile():
    x, y = sp.symbols("x y")
    cases = [
        ("univariate-24", (x**24 - x - 1,), (x,)),
        ("separable-25", (x**5 - 2, y**5 - 3), (x, y)),
        ("nonreduced-24", (x**6, y**4), (x, y)),
        ("algebraic-12", (x**4 - sp.sqrt(2), y**3 - 3), (x, y)),
    ]
    reports = []
    profiler = cProfile.Profile()
    profiler.enable()
    for name, equations, variables in cases:
        sp.core.cache.clear_cache()
        q, groebner = measure(
            lambda e=equations, v=variables: QuotientAlgebra.from_polynomials(e, v)
        )
        _, actions = measure(lambda q=q: q.variable_multiplication_matrices)
        _, trace = measure(lambda q=q: q.trace_vector)
        pairing, pairing_time = measure(lambda q=q: q.trace_pairing)
        ranks = {}
        for method, call in [
            ("expression", pairing.rank),
            ("domain", lambda p=pairing, q=q: p.to_DM(domain=q.domain).rank()),
        ]:
            value, seconds = measure(call)
            ranks[method] = dict(value=value, seconds=seconds)
        assert ranks["expression"]["value"] == ranks["domain"]["value"]
        selected, selected_seconds = measure(lambda q=q: q.geometric_solution_count)
        assert selected == ranks["expression"]["value"]
        reports.append(
            dict(
                name=name,
                dimension=q.dimension,
                domain=str(q.domain),
                seconds=dict(groebner=groebner, actions=actions, trace=trace, pairing=pairing_time),
                rank=ranks,
                selected_rank_seconds=selected_seconds,
                geometric_count=selected,
                pairing_digest=hashlib.sha256(
                    sp.srepr(pairing.applyfunc(sp.simplify)).encode()
                ).hexdigest(),
                normal_form=q.normal_form_cache_info,
                operations=getattr(q, "operation_diagnostics", {}),
            )
        )
    profiler.disable()
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumtime").print_stats(30)
    return dict(
        scope="four exact finite quotient cases; local profiling only", cases=reports
    ), stream.getvalue()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    args = parser.parse_args()
    report, details = profile()
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    args.profile.write_text(details)
    print(json.dumps(report, indent=2))
