"""Measure candidate kernels and verified mathematical construction separately."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import time
from functools import partial
from pathlib import Path

import sympy as sp
from gb_case_catalog import build_case
from sympy.polys.rings import PolyRing

from algroots import QuotientAlgebra, groebner_basis, polysolve
from algroots.groebner import compute_groebner_basis


def measure(operation, repetitions):
    """Warm once and retain every timed sample with the final mathematical result."""
    result = operation()
    samples = []
    for _ in range(repetitions):
        started = time.perf_counter()
        result = operation()
        samples.append(time.perf_counter() - started)
    return result, {"samples_seconds": samples, "median_seconds": statistics.median(samples)}


def canonical(polynomials):
    return {frozenset(p.items()) for p in polynomials}


def basis_cases(names, prime, backends, repetitions):
    """Check identical exact inputs before comparing kernel and public API costs."""
    records = []
    domain = sp.QQ if prime == 0 else sp.GF(prime)
    for name in names:
        case = build_case(name)
        variables = sp.symbols(f"x:{case.nvars}")
        ring = PolyRing(variables, domain, "grevlex")
        inputs = tuple(ring.from_dict(p) for p in case.polynomials)
        expressions = tuple(p.as_expr() for p in inputs)
        reference, reference_time = measure(
            partial(compute_groebner_basis, inputs, ring), repetitions
        )
        for backend in backends:
            row = {"case": name, "prime": prime, "backend": backend, "reference": reference_time}
            try:
                candidate, row["candidate"] = measure(
                    partial(
                        compute_groebner_basis,
                        inputs,
                        ring,
                        backend=backend,
                        certify=False,
                    ),
                    repetitions,
                )
                if canonical(candidate) != canonical(reference):
                    raise ArithmeticError("candidate disagrees with the exact reference")
                basis, row["verified_basis"] = measure(
                    partial(groebner_basis, expressions, variables, domain=domain, backend=backend),
                    repetitions,
                )
                row.update(status="ok", basis_size=len(basis.polys), candidate_verified=True)
                if basis.is_zero_dimensional:
                    quotient, row["quotient"] = measure(
                        partial(
                            QuotientAlgebra.from_polynomials,
                            expressions,
                            variables,
                            domain=domain,
                            groebner_backend=backend,
                        ),
                        repetitions,
                    )
                    row["quotient_dimension"] = quotient.dimension
                    if any(quotient.normal_form(p) != 0 for p in expressions):
                        raise ArithmeticError("quotient failed generator membership")
                else:
                    row["quotient_status"] = "positive_dimensional"
            except (ArithmeticError, RuntimeError, ValueError, TypeError, ImportError) as error:
                row.update(status="failed", error_type=type(error).__name__, error=str(error))
            records.append(row)
    return records


def solver_cases(backends, repetitions):
    """Use finite small systems so root extraction does not hide basis costs."""
    x, y = sp.symbols("x y")
    cases = {
        "coupled": (x * x - y, y * y - 1),
        "triangular": (x - y * y, y**3 - 2),
        "multiple": (x * x, y - x),
    }
    rows = []
    for name, equations in cases.items():
        for backend in backends:
            result, timings = measure(
                partial(
                    polysolve,
                    equations,
                    (x, y),
                    groebner_backend=backend,
                    presolve=False,
                    recognize=False,
                    digits=35,
                ),
                repetitions,
            )
            if len(result.roots) != {"coupled": 4, "triangular": 3, "multiple": 1}[name]:
                raise ArithmeticError("solver returned the wrong geometric root count")
            for point in result.roots:
                substitution = dict(zip((x, y), point, strict=True))
                if any(abs(complex(sp.N(p.subs(substitution), 30))) > 1e-25 for p in equations):
                    raise ArithmeticError("solver failed independent residual checks")
            rows.append(
                dict(
                    case=name,
                    backend=backend,
                    roots=len(result.roots),
                    quotient_dimension=result.quotient_dimension,
                    timings=timings,
                )
            )
    return rows


def main():
    """Write raw timing samples, exact checks, and environment fingerprints."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["eco4", "noon3", "katsura3", "random_sparse_n3_g3_d2_s17", "cyclic5"],
    )
    parser.add_argument("--prime", type=int, default=0)
    parser.add_argument(
        "--backends", nargs="+", choices=["sympy", "native"], default=["native", "sympy"]
    )
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("repetitions must be positive")
    source = Path(__file__).resolve().parents[1] / "src/algroots"
    digest = hashlib.sha256()
    for path in sorted(source.glob("*")):
        if path.suffix in {".py", ".c", ".cpp", ".h", ".inc"}:
            digest.update(path.name.encode() + b"\0" + path.read_bytes())
    report = dict(
        environment=dict(
            python=platform.python_version(),
            sympy=sp.__version__,
            platform=platform.platform(),
            source_sha256=digest.hexdigest(),
        ),
        basis=basis_cases(args.cases, args.prime, args.backends, args.repetitions),
        solver=solver_cases(args.backends, args.repetitions) if args.prime == 0 else [],
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)
    if any(row["status"] != "ok" for row in report["basis"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
