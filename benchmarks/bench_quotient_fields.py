"""Measure exact quotient pipelines, reusable proofs, and algebraic function fields.

Run from the source tree with PYTHONPATH=src. Kernel, verification, and complete
operation timings are distinct. Every recorded operation checks its result;
missing optional native kernels are listed without substituting another engine.
"""

import argparse
import json
import random
import statistics
from pathlib import Path
from time import perf_counter

import sympy as sp

from algroots import (
    AlgebraicFunctionField,
    NativeF4Limits,
    QuotientAlgebra,
    certify_groebner_basis,
    certify_rational_univariate_representation,
    groebner_backends,
    groebner_basis,
)
from algroots.profiling import GroebnerProfiler


def main():
    """Write raw samples, exact checks, preparation costs, and nested profiles."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("repetitions must be positive")
    random.seed(1729)
    x, y, t, a, b, z = sp.symbols("x y t a b z")
    records = []

    def measure(name, operation, **metadata):
        operation()
        samples = []
        with GroebnerProfiler() as profiler:
            for _ in range(args.repetitions):
                start = perf_counter()
                operation()
                samples.append(perf_counter() - start)
        records.append(
            dict(
                name=name,
                seconds=samples,
                median_seconds=statistics.median(samples),
                profiles=profiler.records,
                **metadata,
            )
        )

    for degree in (32, 128):

        def quotient_operation(degree=degree):
            q = QuotientAlgebra.from_polynomials([x**degree - 2], (x,))
            assert q.minimal_polynomial(x, z) == sp.Poly(z**degree - 2, z, domain=sp.QQ)
            assert q.normal_form(q.inverse(x + 1) * (x + 1) - 1) == 0

        measure("sparse_quotient", quotient_operation, dimension=degree)
    source = (x * y - 1, y * y - 1)
    start = perf_counter()
    proof = certify_groebner_basis(source, (x, y))
    setup = perf_counter() - start

    def replay():
        assert groebner_basis(source, (x, y), certificate=proof) == proof.basis

    measure("basis_certificate_reuse", replay, proof_construction_seconds=setup)

    def reference():
        assert groebner_basis(source, (x, y)) == proof.basis

    measure("basis_reference", reference)
    start = perf_counter()
    rur_proof = certify_rational_univariate_representation(
        [x**4 - t, y - x], (x, y), parameters=(t,)
    )
    setup = perf_counter() - start

    def rur_reuse():
        assert rur_proof.specialize({t: 2}).solution_count == 4

    measure("rur_certificate_specialization", rur_reuse, proof_construction_seconds=setup)

    def rur_rebuild():
        q = QuotientAlgebra.from_polynomials([x**4 - 2, y - x], (x, y))
        from algroots.rational_univariate.construction import rur_from_quotient

        assert rur_from_quotient(q).solution_count == 4

    measure("rur_rebuild", rur_rebuild)

    def quadratic():
        return AlgebraicFunctionField(sp.Poly(a * a - t, a, domain=sp.QQ.frac_field(t)))

    def tower():
        field = quadratic()
        return field.extend(sp.Poly(b * b - a - 1, b, domain=field))

    def mixed():
        return AlgebraicFunctionField(
            sp.Poly(
                a * a - t - sp.sqrt(2), a, domain=sp.QQ.algebraic_field(sp.sqrt(2)).frac_field(t)
            )
        )

    fields = []
    for name, build in (("quadratic", quadratic), ("tower", tower), ("number_parameter", mixed)):
        start = perf_counter()
        field = build()
        fields.append((name, field, perf_counter() - start))
    info = groebner_backends()["native"]
    unavailable = []
    native_ready = info.available and "algebraic function fields and towers" in info.domains
    if not native_ready:
        unavailable.append(
            dict(
                backend="native",
                reason=info.reason or "FLINT extension-field kernel is unavailable",
            )
        )
    limits = NativeF4Limits(collect_batch_metrics=True)
    for name, field, setup in fields:
        root = field.tower_symbols[-1]
        source = (x * x - root, y - x)
        expected = groebner_basis(source, (x, y), domain=field)
        for backend in ("sympy", "native") if native_ready else ("sympy",):
            for verification in (
                ("reference", "certificate") if backend == "native" else ("reference",)
            ):

                def operation(
                    source=source,
                    field=field,
                    backend=backend,
                    verification=verification,
                    expected=expected,
                    root=root,
                ):
                    result = groebner_basis(
                        source,
                        (x, y),
                        domain=field,
                        backend=backend,
                        limits=limits if backend == "native" else None,
                        verification=verification,
                    )
                    assert result == expected
                    quotient = QuotientAlgebra.from_groebner_basis(result)
                    assert quotient.dimension == 2
                    assert quotient.normal_form(x * x - root) == 0

                measure(
                    name,
                    operation,
                    backend=backend,
                    verification=verification,
                    field_preparation_seconds=setup,
                    field_degree=field.rank,
                    dimension=2,
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            dict(sympy=sp.__version__, seed=1729, records=records, unavailable=unavailable),
            indent=2,
            default=str,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
