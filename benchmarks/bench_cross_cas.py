"""Reproducible four-engine Groebner-basis benchmark on the algroots 59-case corpus.

Engines: algroots native F4, SymPy, Singular and Macaulay2.
All finite fields are prime and all ideals use the same input monomials,
coefficient reductions, variable order and global monomial order.

Examples:
  python benchmarks/bench_cross_cas.py --list-cases
  python benchmarks/bench_cross_cas.py --cases cyclic5 noon4 katsura5 \
      --primes 101 1009 --orders grevlex --repetitions 5 \
      --output /tmp/gb4.json --export-scripts /tmp/gb4_scripts
  python benchmarks/bench_cross_cas.py --suite all --primes 101 \
      --orders grevlex --repetitions 1 --wall-seconds 15 \
      --output /tmp/gb4_screen.json

Internal GB calculation times exclude startup, input construction and output
serialization. Worker wall times include all three. External CAS execution
is genuinely attempted, and missing binaries or timeouts are recorded, not
replaced with historical results. Independent SymPy checks are optional and
outside each native/CAS measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from gb_case_catalog import SUITES, GBCase, build_case, names_for_suite
from gb_resources import peak_rss_kib, resource_module
from groebner_benchmark import _validate_exact_lift, fingerprint, is_prime

SCRIPT = Path(__file__).resolve()

CAS_NAMES = ("algroots", "sympy", "singular", "macaulay2")
ORDER_TRANSLATIONS = {
    "grevlex": ("dp", "GRevLex"),
    "lex": ("lp", "Lex"),
    "grlex": ("Dp", "GLex"),
}


def input_polynomials(case: GBCase, prime: int) -> list[dict[tuple[int, ...], int]]:
    if prime == 0:
        return [dict(poly) for poly in case.polynomials]
    if not is_prime(prime):
        raise ValueError("Coefficient characteristic must be zero or a prime")
    _validate_exact_lift(case.name, prime)
    return [
        {mono: coefficient % prime for mono, coefficient in poly.items() if coefficient % prime}
        for poly in case.polynomials
    ]


def polynomial_expression(terms: dict[tuple[int, ...], int], *, variables: tuple[str, ...]) -> str:
    pieces = []
    for exponents, coefficient in sorted(terms.items(), reverse=True):
        if not coefficient:
            continue
        factors = [
            f"{name}^{exponent}" if exponent > 1 else name
            for name, exponent in zip(variables, exponents, strict=False)
            if exponent
        ]
        pieces.append(f"({coefficient})" + ("*" + "*".join(factors) if factors else ""))
    return " + ".join(pieces) if pieces else "0"


def external_scripts(case: GBCase, prime: int, order: str) -> dict[str, str]:
    sparse = input_polynomials(case, prime)
    names = tuple(f"x{index}" for index in range(case.nvars))
    equations = ",\n    ".join(polynomial_expression(p, variables=names) for p in sparse)
    sing_order, m2_order = ORDER_TRANSLATIONS[order]
    singular = (
        'system("--ticks-per-sec",1000);\n'
        f"ring R = {prime},({','.join(names)}),{sing_order};\n"
        f"ideal I = {equations};\n"
        "option(redSB);\n"
        "int startGB = rtimer;\n"
        "ideal B = std(I);\n"
        'print("GB_TIME_MS=" + string(rtimer-startGB));\n'
        'print("GB_SIZE=" + string(size(B)));\n'
        'print("GB_BASIS_BEGIN");\n'
        'for (int k=1; k<=size(B); k++) { print("GB_POLY=" + string(B[k])); }\n'
        'print("GB_BASIS_END");\n'
        "quit;\n"
    )
    coefficient_domain = "QQ" if prime == 0 else f"(ZZ/{prime})"
    m2 = (
        f"R = {coefficient_domain}[{','.join(names)}, MonomialOrder => {m2_order}];\n"
        f"I = ideal({equations});\n"
        "startGB = cpuTime();\n"
        "B = gens gb I;\n"
        'print ("GB_TIME_S=" | toString(cpuTime()-startGB));\n'
        'print ("GB_SIZE=" | toString numColumns B);\n'
        'print "GB_BASIS_BEGIN";\n'
        'if numColumns B > 0 then scan(0..(numColumns B - 1), j -> print ("GB_POLY=" | toString(B_(0,j))));\n'
        'print "GB_BASIS_END";\n'
        "exit 0\n"
    )
    return {"singular": singular, "macaulay2": m2}


def load_native():
    from gb_native import load_native as load_kernel

    return load_kernel()


def from_sympy(case: GBCase, sparse: list[dict], prime: int, order: str):
    import sympy as sp
    from sympy.polys.rings import PolyRing

    symbols = sp.symbols(f"x0:{case.nvars}")
    ring = PolyRing(symbols, sp.QQ if prime == 0 else sp.GF(prime), order)
    polys = [ring.from_dict(p) for p in sparse]
    return ring, [p for p in polys if p]


def sympy_basis(case: GBCase, sparse: list[dict], prime: int, order: str):
    from sympy.polys.groebnertools import groebner

    ring, polys = from_sympy(case, sparse, prime, order)
    return groebner(polys, ring) if polys else []


def canonical_sparse(basis, prime: int):
    if prime == 0:
        import sympy as sp

        return [
            {tuple(mon): str(sp.Rational(coef)) for mon, coef in poly.terms() if coef}
            for poly in basis
        ]
    return [
        {tuple(mon): int(coef) % prime for mon, coef in poly.terms() if int(coef) % prime}
        for poly in basis
    ]


def basis_fingerprint(polynomials, prime: int):
    if prime:
        return fingerprint(polynomials)
    canonical = sorted(
        tuple(
            sorted((tuple(exponents), str(coefficient)) for exponents, coefficient in poly.items())
        )
        for poly in polynomials
    )
    return hashlib.sha256(repr(canonical).encode("utf8")).hexdigest()


def verified_fingerprint(
    case: GBCase, sparse: list[dict], prime: int, order: str, observed: list[dict]
):
    """Independent exact normal forms via SymPy, excluded from all timed regions."""
    from sympy.polys.groebnertools import groebner

    ring, inputs = from_sympy(case, sparse, prime, order)
    candidate = [
        ring.from_dict(
            {m: __import__("sympy").Rational(c) if prime == 0 else c for m, c in poly.items()}
        )
        for poly in observed
        if poly
    ]
    # Computing GB(candidate) alone is not proof of equality with GB(inputs).
    # Both are separately normalized; exact equality is required.
    independent = canonical_sparse(groebner(inputs, ring) if inputs else [], prime)
    candidate_red = canonical_sparse(groebner(candidate, ring) if candidate else [], prime)
    return basis_fingerprint(independent, prime) == basis_fingerprint(
        candidate_red, prime
    ), basis_fingerprint(independent, prime)


def worker_local(args):
    resource = resource_module()
    resource.setrlimit(resource.RLIMIT_CPU, (args.cpu_seconds, args.cpu_seconds + 1))
    if args.memory_mib:
        resource.setrlimit(resource.RLIMIT_AS, (args.memory_mib * 1024**2,) * 2)
    case = build_case(args.case)
    sparse = input_polynomials(case, args.prime)
    start = time.perf_counter()
    if args.engine == "algroots" and args.prime == 0:
        ring, inputs = from_sympy(case, sparse, args.prime, args.order)
        from algroots.native_f4 import NativeF4Limits, native_modular_basis

        settings = NativeF4Limits(
            max_pairs=args.max_pairs,
            max_basis=args.max_basis,
            max_terms=args.max_terms,
            max_reduction_steps=args.max_reduction_steps,
            batch_size=args.batch_size,
            max_primes=args.max_primes,
        )
        init_time = time.perf_counter() - start
        start = time.perf_counter()
        result = native_modular_basis(
            inputs, ring, limits=settings, certify=(args.qq_mode == "native_certified")
        )
        native_seconds = time.perf_counter() - start
        basis = canonical_sparse(result.basis, 0)
        extra = {
            "basis_size": len(basis),
            "initialization_seconds": init_time,
            "algorithm": "algroots_native_F4_QQ",
            "certificate_produced": result.certified,
            "verification": result.verification,
            "primes": list(result.primes),
        }
    elif args.engine == "algroots":
        ext = load_native()
        init_time = time.perf_counter() - start
        start = time.perf_counter()
        basis, stats = ext.compute(
            sparse,
            args.prime,
            case.nvars,
            args.order,
            max_pairs=args.max_pairs,
            max_basis=args.max_basis,
            max_terms=args.max_terms,
            max_reduction_steps=args.max_reduction_steps,
            batch_size=args.batch_size,
        )
        native_seconds = time.perf_counter() - start
        extra = {
            "basis_size": len(basis),
            "compiled_extension": True,
            "algorithm": "f4_tiled_finish",
            "initialization_seconds": init_time,
            "pairs_processed": stats.get("pairs_processed"),
            "symbolic_preprocessing_ns": stats.get("f4_symbolic_preprocessing_ns"),
            "matrix_elimination_ns": stats.get("f4_matrix_elimination_ns"),
        }
    else:
        ring, inputs = from_sympy(case, sparse, args.prime, args.order)
        from sympy.polys.groebnertools import groebner

        init_time = time.perf_counter() - start
        start = time.perf_counter()
        computed = groebner(inputs, ring) if inputs else []
        native_seconds = time.perf_counter() - start
        basis = canonical_sparse(computed, args.prime)
        extra = {
            "basis_size": len(basis),
            "initialization_seconds": init_time,
            "algorithm": "sympy.groebnertools.groebner",
        }
    result = {
        "status": "ok",
        "internal_seconds": native_seconds,
        "fingerprint": basis_fingerprint(basis, args.prime),
        "rss_kib": peak_rss_kib(resource.getrusage(resource.RUSAGE_SELF)),
        **extra,
    }
    if args.verify:
        verify_start = time.perf_counter()
        result["independent_sympy_agreement"], result["reference_fingerprint"] = (
            verified_fingerprint(case, sparse, args.prime, args.order, basis)
        )
        result["verification_seconds_excluded"] = time.perf_counter() - verify_start
        if not result["independent_sympy_agreement"]:
            result["status"] = "mismatch"
    return result


def _monomial_sort_key(monomial: tuple[int, ...], order: str):
    """Return order-preserving key for lex/grlex/grevlex."""
    if order == "lex":
        return monomial
    if order == "grlex":
        return (sum(monomial), monomial)
    if order == "grevlex":
        return (sum(monomial), tuple(-degree for degree in reversed(monomial)))
    raise ValueError(f"Unsupported monomial order: {order}")


def parse_external_polynomial(expression: str, nvars: int, prime: int, order: str):
    """Exact sparse parser for Singular/Macaulay2's standard expanded output.

    No Gröbner basis is recomputed. Every coefficient and monomial is parsed,
    combined exactly, then made monic using the *requested* monomial order.
    Unsupported syntax fails closed instead of silently dropping terms.
    """
    from fractions import Fraction

    source = expression.replace(" ", "").replace("\t", "")
    if not source:
        raise ValueError("Empty polynomial expression")
    parts = re.findall(r"([+-]?)([^+-]+)", source)
    if not parts or "".join(sign + term for sign, term in parts) != source:
        raise ValueError(f"Unsupported polynomial expression: {expression[:200]}")
    acc = {}
    for sign, term in parts:
        coeff = Fraction(-1 if sign == "-" else 1)
        exponents = [0] * nvars
        for factor in term.split("*"):
            m = re.fullmatch(r"x(\d+)(?:\^(\d+))?", factor)
            if m:
                index = int(m.group(1))
                if index >= nvars:
                    raise ValueError(f"Variable index out of range: {factor}")
                exponents[index] += int(m.group(2) or 1)
            else:
                # The CASes output integer or rational literal coefficients.
                coeff *= Fraction(factor.strip("()"))
        monomial = tuple(exponents)
        acc[monomial] = acc.get(monomial, Fraction(0)) + coeff
    acc = {m: c for m, c in acc.items() if c}
    if not acc:
        return {}
    lead = max(acc, key=lambda m: _monomial_sort_key(m, order))
    if prime:

        def mod_q(value):
            return (value.numerator * pow(value.denominator, -1, prime)) % prime

        reduced = {m: mod_q(c) for m, c in acc.items()}
        reduced = {m: c for m, c in reduced.items() if c}
        if not reduced:
            return {}
        # Characteristic reduction can eliminate a nominal leading term.
        lead = max(reduced, key=lambda m: _monomial_sort_key(m, order))
        inv = pow(reduced[lead], -1, prime)
        return {m: c * inv % prime for m, c in reduced.items()}
    lc = acc[lead]
    return {m: str(c / lc) for m, c in acc.items()}


def worker_external(args):
    resource = resource_module()
    exe = {"singular": "Singular", "macaulay2": "M2"}[args.engine]
    executable = shutil.which(exe)
    if executable is None:
        return {"status": "unavailable", "reason": f"{exe} executable not on PATH"}
    case = build_case(args.case)
    scripts = external_scripts(case, args.prime, args.order)
    content = scripts[args.engine]
    extension = {"singular": ".sing", "macaulay2": ".m2"}[args.engine]
    with tempfile.TemporaryDirectory(prefix="gb-four-cas-") as directory:
        path = Path(directory) / ("input" + extension)
        path.write_text(content, encoding="utf8")
        argv = (
            [executable, "-q", str(path)]
            if args.engine == "singular"
            else [executable, "--script", str(path)]
        )
        t0 = time.perf_counter()

        def cas_limits():
            resource.setrlimit(resource.RLIMIT_CPU, (args.cpu_seconds, args.cpu_seconds + 1))
            if args.memory_mib:
                resource.setrlimit(resource.RLIMIT_AS, (args.memory_mib * 1024**2,) * 2)

        run = subprocess.run(
            argv,
            cwd=directory,
            capture_output=True,
            text=True,
            errors="replace",
            check=False,
            preexec_fn=cas_limits,
            env={
                **os.environ,
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            },
            timeout=args.wall_seconds,
        )
        outer_seconds = time.perf_counter() - t0
        child_peak_kib = peak_rss_kib(resource.getrusage(resource.RUSAGE_CHILDREN))
    out = run.stdout + "\n" + run.stderr
    if run.returncode != 0 or "GB_BASIS_END" not in out:
        return {
            "status": "error",
            "exit_code": run.returncode,
            "stderr_tail": out[-1000:],
            "worker_wall_seconds": outer_seconds,
        }
    tm = re.search(r"GB_TIME_(MS|S)=\s*([0-9.eE+\-]+)", out)
    size = re.search(r"GB_SIZE=\s*(\d+)", out)
    if tm is None or size is None:
        return {"status": "parse_error", "stdout_tail": out[-1000:]}
    reported_internal = float(tm.group(2)) / (1000.0 if tm.group(1) == "MS" else 1.0)
    if not math.isfinite(reported_internal) or reported_internal < 0:
        return {"status": "parse_error", "reason": "Invalid internal timer"}
    lines = out.split("GB_BASIS_BEGIN", 1)[1].split("GB_BASIS_END", 1)[0].splitlines()
    expressions = [line.split("GB_POLY=", 1)[1].strip() for line in lines if "GB_POLY=" in line]
    if len(expressions) != int(size.group(1)):
        return {
            "status": "parse_error",
            "reason": "Polynomial count does not match emitted basis",
            "declared_size": int(size.group(1)),
            "emitted_size": len(expressions),
        }
    result = {
        "status": "ok",
        "internal_seconds": reported_internal,
        "internal_clock": {
            "singular": "Singular rtimer (milliseconds)",
            "macaulay2": "Macaulay2 cpuTime (CPU seconds)",
        }[args.engine],
        "worker_wall_seconds": outer_seconds,
        "peak_rss_kib": child_peak_kib,
        "emitted_basis_size": len(expressions),
        "fingerprint": None,
        "executable": executable,
    }
    # Independently canonicalize the ACTUAL emitted output; no reference CAS
    # computation or SymPy parser sits in this path. This matters for large
    # emitted bases that would otherwise take seconds to parse per trial.
    verify_t0 = time.perf_counter()
    parsed = []
    try:
        for expression in expressions:
            poly = parse_external_polynomial(expression, case.nvars, args.prime, args.order)
            if poly:
                parsed.append(poly)
        result["fingerprint"] = basis_fingerprint(parsed, args.prime)
        result["parsed_basis"] = True
        result["normalized_basis_size"] = len(parsed)
    except Exception as error:
        result["status"] = "parse_error"
        result["error"] = f"{type(error).__name__}: {error}"
        result["parsed_basis"] = False
    result["parsing_seconds_excluded"] = time.perf_counter() - verify_t0
    if args.verify and result["status"] == "ok":
        verify_t0 = time.perf_counter()
        sparse = input_polynomials(case, args.prime)
        try:
            ok, reference = verified_fingerprint(case, sparse, args.prime, args.order, parsed)
            result["independent_sympy_agreement"] = ok
            result["reference_fingerprint"] = reference
            if not ok:
                result["status"] = "mismatch"
        except Exception as error:
            result["status"] = "verification_error"
            result["verification_error"] = f"{type(error).__name__}: {error}"
        result["verification_seconds_excluded"] = time.perf_counter() - verify_t0
    return result


def _exec_worker(args, case_name: str, prime: int, order: str, engine: str, algorithm: str | None):
    if (
        engine in ("singular", "macaulay2")
        and shutil.which({"singular": "Singular", "macaulay2": "M2"}[engine]) is None
    ):
        return {"status": "unavailable", "reason": f"{engine} executable not installed"}
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--worker",
        "--case",
        case_name,
        "--prime",
        str(prime),
        "--order",
        order,
        "--engine",
        engine,
        "--wall-seconds",
        str(args.wall_seconds),
        "--cpu-seconds",
        str(args.cpu_seconds),
        "--memory-mib",
        str(args.memory_mib),
        "--max-pairs",
        str(args.max_pairs),
        "--max-basis",
        str(args.max_basis),
        "--max-terms",
        str(args.max_terms),
        "--max-reduction-steps",
        str(args.max_reduction_steps),
        "--batch-size",
        str(args.batch_size),
        "--qq-mode",
        args.qq_mode,
        "--max-primes",
        str(args.max_primes),
    ]
    if args.verify:
        cmd.append("--verify")
    # All backends have the same clean-Python-process startup policy.
    begin = time.perf_counter()
    try:
        child = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=args.wall_seconds + 2,
            env={
                **os.environ,
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "PYTHONHASHSEED": "0",
            },
        )
        wall_seconds = time.perf_counter() - begin
        if child.returncode:
            if child.returncode == -24:
                return {
                    "status": "cpu_limit",
                    "exit_code": child.returncode,
                    "wall_seconds": wall_seconds,
                }
            return {
                "status": "error",
                "exit_code": child.returncode,
                "stderr_tail": child.stderr[-900:],
                "wall_seconds": wall_seconds,
            }
        res = json.loads(child.stdout.strip().splitlines()[-1])
        res["wall_seconds"] = wall_seconds
        return res
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "wall_seconds_limit": args.wall_seconds + 2}
    except (ValueError, OSError, json.JSONDecodeError) as error:
        return {"status": "error", "error": repr(error)}


def version_info():
    env = {
        "host": platform.platform(),
        "python": sys.version,
        "cpu_model": Path("/proc/cpuinfo")
        .read_text()
        .split("model name", 1)[-1]
        .split("\n", 1)[0]
        .lstrip(": \t")
        if Path("/proc/cpuinfo").exists()
        else "unknown",
        "binary_versions": {},
    }
    import sympy as sp

    env["sympy_version"] = sp.__version__
    source_file = SCRIPT.parents[1] / "src" / "algroots" / "_native_f4.cpp"
    if source_file.is_file():
        env["native_f4_source_sha256"] = hashlib.sha256(source_file.read_bytes()).hexdigest()
    shared = sorted(source_file.parent.glob("_native_f4*.so"))
    if len(shared) == 1:
        env["native_f4_binary_sha256"] = hashlib.sha256(shared[0].read_bytes()).hexdigest()
    try:
        import flint

        env["python_flint_version"] = getattr(flint, "__version__", "unknown")
    except ImportError:
        env["python_flint_version"] = "not installed"
    for exe, cmd in (
        ("Singular", ["Singular", "--version"]),
        ("M2", ["M2", "--version"]),
    ):
        path = shutil.which(exe)
        if not path:
            env["binary_versions"][exe] = {"status": "unavailable"}
            continue
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=False)
            env["binary_versions"][exe] = {
                "executable": path,
                "version": (res.stdout + res.stderr)[-1000:],
            }
        except Exception as error:
            env["binary_versions"][exe] = {"error": repr(error)}
    return env


def summary(records, engines):
    report = {}
    keys = sorted({(r["case"], r["prime"], r["order"]) for r in records})
    for case, prime, order in keys:
        relevant = [
            r for r in records if (r["case"], r["prime"], r["order"]) == (case, prime, order)
        ]
        group = {}
        for engine in engines:
            r = [x for x in relevant if x["engine"] == engine]
            completed = [
                x for x in r if x["status"] == "ok" and x.get("internal_seconds") is not None
            ]
            group[engine] = {
                "completed": len(completed),
                "attempts": len(r),
                "status": (
                    "complete"
                    if len(completed) == len(r) and r
                    else "unavailable"
                    if r and all(x["status"] == "unavailable" for x in r)
                    else "incomplete"
                ),
                "median_internal_seconds": (
                    statistics.median(x["internal_seconds"] for x in completed)
                    if completed
                    else None
                ),
                "median_wall_seconds": (
                    statistics.median(x["wall_seconds"] for x in completed) if completed else None
                ),
                "verified": (
                    all(x.get("independent_sympy_agreement", False) for x in completed)
                    if completed and all("independent_sympy_agreement" in x for x in completed)
                    else None
                ),
            }
        known = {
            x.get("fingerprint") for x in relevant if x["status"] == "ok" and x.get("fingerprint")
        }
        known_engines = {
            x["engine"] for x in relevant if x["status"] == "ok" and x.get("fingerprint")
        }
        field_name = "QQ" if prime == 0 else f"GF({prime})"
        report[f"{case}|{field_name}|{order}"] = {
            "engines": group,
            "same_completed_fingerprint": len(known) == 1 if len(known_engines) >= 2 else None,
        }
    return report


def write_scripts(target: Path, case_names: list[str], primes: list[int], orders: list[str]):
    target.mkdir(parents=True, exist_ok=True)
    for name in case_names:
        case = build_case(name)
        for prime in primes:
            for order in orders:
                try:
                    scripts = external_scripts(case, prime, order)
                except ValueError:
                    continue
                for engine, text in scripts.items():
                    suffix = {"singular": "sing", "macaulay2": "m2"}[engine]
                    (target / f"{name}_p{prime}_{order}.{suffix}").write_text(text)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--list-cases", action="store_true")
    ap.add_argument("--suite", choices=(*SUITES.keys(), "all"), default="core")
    ap.add_argument("--cases", nargs="+")
    ap.add_argument("--case", default="cyclic4")
    ap.add_argument("--primes", type=int, nargs="+", default=[101, 1009])
    ap.add_argument("--prime", type=int, default=1009)
    ap.add_argument(
        "--qq-mode",
        choices=("native_candidate", "native_certified"),
        default="native_certified",
    )
    ap.add_argument("--max-primes", type=int, default=12)
    ap.add_argument("--orders", nargs="+", choices=ORDER_TRANSLATIONS.keys(), default=["grevlex"])
    ap.add_argument("--order", choices=ORDER_TRANSLATIONS.keys(), default="grevlex")
    ap.add_argument("--engines", nargs="+", choices=CAS_NAMES, default=list(CAS_NAMES))
    ap.add_argument("--engine", choices=CAS_NAMES, default="algroots")
    ap.add_argument("--repetitions", type=int, default=3)
    ap.add_argument("--wall-seconds", type=int, default=30)
    ap.add_argument("--cpu-seconds", type=int, default=28)
    ap.add_argument("--memory-mib", type=int, default=2048, help="address-space cap; 0 disables it")
    ap.add_argument("--max-pairs", type=int, default=100000)
    ap.add_argument("--max-basis", type=int, default=1024)
    ap.add_argument("--max-terms", type=int, default=500000)
    ap.add_argument("--max-reduction-steps", type=int, default=6000000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument(
        "--verify",
        action="store_true",
        help="independent SymPy reference equality outside the timed interval",
    )
    ap.add_argument("--output", default="four_cas_results.json")
    ap.add_argument("--export-scripts", default=None)
    ap.add_argument(
        "--export-only",
        action="store_true",
        help="Generate CAS input scripts without running engines",
    )
    ap.add_argument(
        "--resume", action="store_true", help="Resume an identical configuration from existing JSON"
    )
    args = ap.parse_args()
    if args.list_cases:
        print(json.dumps({"suites": {k: list(v) for k, v in SUITES.items()}}, indent=2))
        return
    if args.worker:
        try:
            if args.engine in ("algroots", "sympy"):
                result = worker_local(args)
            else:
                result = worker_external(args)
        except subprocess.TimeoutExpired:
            result = {"status": "timeout"}
        except Exception as error:
            result = {"status": "error", "exception": f"{type(error).__name__}: {error}"}
        print(json.dumps(result, default=str))
        return
    if (
        args.repetitions < 1
        or args.wall_seconds < 1
        or args.cpu_seconds < 1
        or (args.memory_mib != 0 and args.memory_mib < 100)
        or args.max_primes < 2
    ):
        ap.error("invalid resource or repetitions parameters")
    names = args.cases if args.cases else list(names_for_suite(args.suite))
    for name in names:
        build_case(name)
    if any(p != 0 and not is_prime(p) for p in args.primes):
        ap.error("characteristics must be zero (QQ) or prime")
    if args.export_only and not args.export_scripts:
        ap.error("--export-only requires --export-scripts")
    if args.export_scripts:
        write_scripts(Path(args.export_scripts), names, args.primes, args.orders)
    if args.export_only:
        print(f"Exported scripts to {args.export_scripts}")
        return
    report = {
        "environment": version_info(),
        "configuration": {
            "cases": names,
            "primes": args.primes,
            "orders": args.orders,
            "engines": args.engines,
            "repetitions": args.repetitions,
            "wall_seconds": args.wall_seconds,
            "memory_mib": args.memory_mib,
            "cpu_seconds": args.cpu_seconds,
            "verify": args.verify,
            "algroots_algorithm": "f4_tiled_finish",
            "algroots_qq_mode": args.qq_mode,
            "timer_policy": "internal compute times only; process wall separately",
            "validation_policy": "independent_sympy_agreement outside timed GB",
        },
        "trials": [],
        "summary": {},
        "warnings": [],
        "input_provenance": {},
    }
    missing = [exe for exe in ("Singular", "M2") if shutil.which(exe) is None]
    if missing:
        report["warnings"].append(
            "Missing external executables: " + ", ".join(missing) + "; results marked unavailable"
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    if args.resume and output.exists():
        prior = json.loads(output.read_text(encoding="utf-8"))
        if prior["configuration"] != report["configuration"]:
            ap.error("Cannot resume: benchmark configuration differs from existing results")
        for key in ("native_f4_source_sha256", "native_f4_binary_sha256", "sympy_version"):
            if prior["environment"].get(key) != report["environment"].get(key):
                ap.error(f"Cannot resume: environment {key} differs")
        prior_versions = prior["environment"].get("binary_versions", {})
        current_versions = report["environment"].get("binary_versions", {})
        for engine_name in ("Singular", "M2"):
            old_info = prior_versions.get(engine_name, {})
            new_info = current_versions.get(engine_name, {})
            if old_info.get("status") != "unavailable" and old_info != new_info:
                ap.error(f"Cannot resume: {engine_name} version or installation changed")
            if old_info.get("status") == "unavailable" and new_info.get("status") != "unavailable":
                prior.setdefault("warnings", []).append(
                    f"{engine_name} installed after first run; filling previously unavailable trials"
                )
                prior_versions[engine_name] = new_info
        report = prior
        seen = {
            (r["case"], r["prime"], r["order"], r["repetition"], r["engine"])
            for r in report["trials"]
            if r["status"] != "unavailable"
        }
    rng = random.Random(20261106)
    for case in names:
        for prime in args.primes:
            for order in args.orders:
                try:
                    if prime != 0:
                        _validate_exact_lift(case, prime)
                except ValueError as error:
                    report["warnings"].append(f"skip {case} GF({prime}): {error}")
                    continue
                current_case = build_case(case)
                sparse_inputs = input_polynomials(current_case, prime)
                input_sha = hashlib.sha256(
                    repr(
                        sorted(
                            tuple(sorted((tuple(m), str(c)) for m, c in poly.items()))
                            for poly in sparse_inputs
                        )
                    ).encode("utf8")
                ).hexdigest()
                key = f"{case}|" + ("QQ" if prime == 0 else f"GF({prime})") + f"|{order}"
                report["input_provenance"][key] = {
                    "input_sha256": input_sha,
                    "source": current_case.source,
                    "note": current_case.note,
                    "nvars": current_case.nvars,
                    "ngenerators": len(current_case.polynomials),
                    "term_order": order,
                }
                for repetition in range(args.repetitions):
                    engines = list(args.engines)
                    rng.shuffle(engines)
                    for engine in engines:
                        trial_key = (case, prime, order, repetition, engine)
                        if trial_key in seen:
                            continue
                        # Replacing an old unavailable entry permits filling in a
                        # CAS engine later, after it has been installed.
                        report["trials"] = [
                            r
                            for r in report["trials"]
                            if (r["case"], r["prime"], r["order"], r["repetition"], r["engine"])
                            != trial_key
                        ]
                        record = _exec_worker(
                            args,
                            case,
                            prime,
                            order,
                            engine,
                            "f4_tiled_finish" if engine == "algroots" else None,
                        )
                        record.update(
                            {
                                "case": case,
                                "prime": prime,
                                "order": order,
                                "repetition": repetition,
                                "engine": engine,
                                "nvars": current_case.nvars,
                                "ngenerators": len(current_case.polynomials),
                                "input_sha256": input_sha,
                                "case_source": current_case.source,
                                "algorithm": "f4_tiled_finish" if engine == "algroots" else None,
                            }
                        )
                        report["trials"].append(record)
                        if record["status"] != "unavailable":
                            seen.add(trial_key)
                        output.write_text(
                            json.dumps(report, indent=2, default=str), encoding="utf-8"
                        )
                        field_label = "QQ" if prime == 0 else f"GF({prime})"
                        print(
                            f"{case} {field_label} {order} rep={repetition} {engine}: {record['status']} "
                            f"{record.get('internal_seconds', '-')}s",
                            flush=True,
                        )
                    report["summary"] = summary(report["trials"], args.engines)
                    output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print("Saved", output)


if __name__ == "__main__":
    main()
