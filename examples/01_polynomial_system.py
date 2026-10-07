"""Solve a coupled polynomial system.

Guarantee: Conditional finite-root extraction.
Run from the repository root: python examples/01_polynomial_system.py
"""

import sympy as sp

from algroots import polysolve


def main():
    x, y = sp.symbols("x y")
    result = polysolve((x**2 + y**2 - 3, x * y - 1), (x, y), digits=40, recognize=False)
    assert len(result.roots) == result.geometric_solution_count == 4
    assert result.completeness.status == "conditional"
    print("Variables:", result.variables)
    print("Roots:", result.roots)
    print("Evidence:", result.completeness)


if __name__ == "__main__":
    main()
