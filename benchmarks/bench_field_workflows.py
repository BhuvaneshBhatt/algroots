"""Seeded exact before/after measurements; select sources through PYTHONPATH."""

import json
import random
import statistics
import sys
import time
import tracemalloc

import sympy as sp

from algroots import (
    QuotientAlgebra,
    certify_groebner_basis,
    certify_rational_univariate_representation,
)
from algroots.rational_univariate.construction import rur_from_quotient

random.seed(1729)
x, y, t, z = sp.symbols("x y t z")
records = []


def measure(name, operation, **metadata):
    operation()
    samples = []
    for _ in range(5):
        start = time.perf_counter()
        operation()
        samples.append(time.perf_counter() - start)
    tracemalloc.start()
    operation()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    records.append(
        dict(
            name=name,
            seconds=samples,
            median=statistics.median(samples),
            peak_python_bytes=peak,
            **metadata,
        )
    )


for degree in (16, 32):

    def trace(degree=degree):
        q = QuotientAlgebra.from_polynomials([x**degree - 2], (x,))
        assert q.geometric_solution_count == degree
        assert q.trace_pairing[0, 0] == degree

    measure("trace_pairing", trace, dimension=degree)

    def rur(degree=degree):
        q = QuotientAlgebra.from_polynomials([x**degree - 2, y - x], (x, y))
        r = rur_from_quotient(q)
        assert r.solution_count == degree
        assert r.normalized_coordinate_polynomials()[0] == r.normalized_coordinate_polynomials()[1]

    measure("rur", rur, dimension=degree)


def noncyclic():
    q = QuotientAlgebra.from_polynomials([x**8, y**8], (x, y))
    assert q.characteristic_polynomial(x + y, z) == sp.Poly(z**64, z, domain=sp.QQ)
    assert q.geometric_solution_count == 1


measure("noncyclic_characteristic_and_trace", noncyclic, dimension=64)
proof = certify_groebner_basis([t * x, y - x], (x, y), domain=sp.QQ.frac_field(t))
start = time.perf_counter()
prepared = proof.prepare() if hasattr(proof, "prepare") else None
setup = time.perf_counter() - start
if prepared is None:
    from algroots import parametric_groebner_basis

    family = parametric_groebner_basis([t * x, y - x], (x, y), (t,))
else:
    family = prepared


def basis_fibers():
    for point in (1, 2, 3, 4):
        assert set(family.specialize({t: point})) == {x, y}


measure("four_basis_fibers", basis_fibers, prepared=prepared is not None, preparation_seconds=setup)
rur_proof = certify_rational_univariate_representation([x * x - t, y - x], (x, y), parameters=(t,))
start = time.perf_counter()
family = rur_proof.prepare() if hasattr(rur_proof, "prepare") else rur_proof
setup = time.perf_counter() - start


def rur_fibers():
    for point in (1, 2, 3, 4):
        assert family.specialize({t: point}).solution_count == 2


measure(
    "four_rur_fibers", rur_fibers, prepared=hasattr(rur_proof, "prepare"), preparation_seconds=setup
)
print(json.dumps(dict(python=sys.version, sympy=sp.__version__, records=records), indent=2))
