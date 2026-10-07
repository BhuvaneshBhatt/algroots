"""Resolve closely spaced exact roots.

Guarantee: Distinct exact endpoint proofs.
Run from the repository root: python examples/08_clustered_roots.py
"""

import sympy as sp

from algroots import polysolve


def main():
    x = sp.Symbol("x")
    delta = sp.Rational(1, 10**25)
    result = polysolve(
        ((x - 1) * (x - 1 - delta),),
        (x,),
        digits=60,
        max_precision_digits=180,
        recognize=False,
        certify="required",
    )
    points = {attempt.certificate.point for attempt in result.root_certifications}
    assert points == {(sp.S.One,), (1 + delta,)}
    print("Exact points:", points)
    print("Distinct roots:", result.geometric_solution_count)


if __name__ == "__main__":
    main()
