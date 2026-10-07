"""Solve and certify an algebraic-field system.

Guarantee: Exact point certificate; automatic boxes remain rational-only.
Run from the repository root: python examples/06_algebraic_coefficients.py
"""

import sympy as sp

from algroots import polysolve
from algroots.certification import certify_isolated_root


def main():
    x, y = sp.symbols("x y")
    equations = (x - sp.sqrt(2) * y, y**2 - 1)
    result = polysolve(equations, (x, y), method="rur", digits=40, recognize=False)
    certificate = certify_isolated_root(equations, (x, y), (sp.sqrt(2), 1))
    assert len(result.roots) == 2 and certificate.verify()
    print("Numerical roots:", result.roots)
    print("Exact certified point:", certificate.point)


if __name__ == "__main__":
    main()
