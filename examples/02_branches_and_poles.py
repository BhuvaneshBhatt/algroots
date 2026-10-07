"""Retain principal branches and exclude poles.

Guarantee: Original expression and domain checks.
Run from the repository root: python examples/02_branches_and_poles.py
"""

import sympy as sp

from algroots import algsolve


def main():
    x = sp.Symbol("x")
    radical = algsolve((sp.sqrt(x) - (x - 2),), (x,), digits=40, recognize=False)
    assert len(radical.roots) == 1 and abs(radical.roots[0][0] - 4) < sp.Rational(1, 10**30)
    rational = algsolve(((x + 1) / (x - 2) - 3,), (x,), recognize=False)
    assert len(rational.roots) == 1 and abs(rational.roots[0][0] - sp.Rational(7, 2)) < sp.Rational(
        1, 10**20
    )
    print("Principal square root keeps x=4 and rejects x=1:", radical.roots)
    print("Rational equation keeps x=7/2 and excludes the pole x=2:", rational.roots)


if __name__ == "__main__":
    main()
