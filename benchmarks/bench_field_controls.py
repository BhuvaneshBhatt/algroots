"""Exact field controls for sparse trace and RUR arithmetic."""

import json
import statistics
import time

import sympy as sp

from algroots import AlgebraicFunctionField, QuotientAlgebra
from algroots.rational_univariate.construction import rur_from_quotient

x, y, t, a = sp.symbols("x y t a")
records = []
for name, domain, coefficient in [
    ("number", sp.QQ.algebraic_field(sp.sqrt(2)), sp.sqrt(2)),
    ("parameter", sp.QQ.frac_field(t), t),
    ("tower", AlgebraicFunctionField(sp.Poly(a * a - t, a, domain=sp.QQ.frac_field(t))), a),
]:

    def operation(domain=domain, coefficient=coefficient):
        q = QuotientAlgebra.from_polynomials([x**8 - coefficient, y - x], (x, y), domain=domain)
        r = rur_from_quotient(q)
        assert r.solution_count == 8
        c = r.normalized_coordinate_polynomials()
        assert c[0] == c[1]

    operation()
    samples = []
    for _ in range(5):
        start = time.perf_counter()
        operation()
        samples.append(time.perf_counter() - start)
    records.append(dict(field=name, seconds=samples, median=statistics.median(samples)))
print(json.dumps(records, indent=2))
