"""Measure exact coefficient fields, with native construction and verification separated."""

import argparse
import json
import platform
import statistics
from functools import partial
from pathlib import Path
from time import perf_counter

import flint
import sympy as sp
from sympy.polys.groebnertools import groebner
from sympy.polys.rings import PolyRing

from algroots import NativeF4Limits, groebner_basis, parametric_groebner_basis
from algroots.native_fields import native_field_basis


def measure(operation, repetitions):
    """Retain warm timing samples separately from the untimed first execution."""
    result = operation()
    samples = []
    for _ in range(repetitions):
        start = perf_counter()
        result = operation()
        samples.append(perf_counter() - start)
    return result, {"samples_seconds": samples, "median_seconds": statistics.median(samples)}


def main():
    """Write independently checked bases and full image profiles to JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("repetitions must be positive")
    x, y, t, u = sp.symbols("x y t u")
    number = sp.QQ.algebraic_field(sp.sqrt(2))
    biquadratic = sp.QQ.algebraic_field(sp.sqrt(2), sp.sqrt(3))
    cases = [
        ("rational_one_parameter", sp.QQ.frac_field(t), [x * y - t, y * y - x - 1]),
        ("rational_two_parameters", sp.QQ.frac_field(t, u), [x * y - t, y * y - u * x - 1]),
        ("rational_denominators", sp.QQ.frac_field(t), [(t + 1) * x + y - 1, x * x + y * y - 1]),
        ("quadratic_constants", number, [x * y - sp.sqrt(2), y * y - x - 1]),
        (
            "biquadratic_constants",
            biquadratic,
            [x * y - sp.sqrt(2), y * y - sp.sqrt(3) * x - 1],
        ),
        (
            "mixed_one_parameter",
            number.frac_field(t),
            [x * y - t, y * y - (t + sp.sqrt(2)) * x - 1],
        ),
        (
            "mixed_two_parameters",
            number.frac_field(t, u),
            [x * y - t, y * y - u * x - sp.sqrt(2)],
        ),
    ]
    limits = NativeF4Limits(collect_batch_metrics=True, max_reduction_steps=2000000)
    records = []
    for name, domain, source in cases:
        ring = PolyRing((x, y), domain, "grevlex")
        inputs = tuple(ring.from_expr(p) for p in source)
        expected, reference_time = measure(partial(groebner, list(inputs), ring), args.repetitions)
        reference_public, reference_public_time = measure(
            partial(groebner_basis, source, (x, y), domain=domain), args.repetitions
        )
        candidate, native_time = measure(
            partial(native_field_basis, inputs, ring, limits=limits), args.repetitions
        )
        if {frozenset(p.items()) for p in candidate.basis} != {
            frozenset(p.items()) for p in expected
        }:
            raise ArithmeticError(f"native candidate disagrees: {name}")
        verified, public_time = measure(
            partial(groebner_basis, source, (x, y), domain=domain, backend="native", limits=limits),
            args.repetitions,
        )
        if verified != reference_public:
            raise ArithmeticError(f"public bases disagree: {name}")
        records.append(
            dict(
                case=name,
                domain=str(domain),
                source=list(map(str, source)),
                basis=list(map(str, verified)),
                basis_size=len(expected),
                reference=reference_time,
                reference_public=reference_public_time,
                native_candidate=native_time,
                verified_public=public_time,
                primes=candidate.primes,
                stats=candidate.stats,
                status="exact_match",
            )
        )
        args.output.write_text(json.dumps(dict(records=records), indent=2) + "\n")
    source = [t * x, x - 1]
    guarded, guard_time = measure(
        partial(parametric_groebner_basis, source, (x,), (t,), backend="native", limits=limits),
        args.repetitions,
    )
    if not guarded.verify() or list(guarded.specialize({t: 0})) != [x - 1]:
        raise ArithmeticError("exceptional specialization failed")
    report = dict(
        environment=dict(
            python=platform.python_version(),
            platform=platform.platform(),
            sympy=sp.__version__,
            python_flint=flint.__version__,
        ),
        repetitions=args.repetitions,
        records=records,
        guarded=dict(source=list(map(str, source)), guard=str(guarded.guard), timing=guard_time),
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
