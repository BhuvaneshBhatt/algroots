"""Recognize and jointly certify numerical coordinates.

Guarantee: Coordinate and joint equation certification.
Run from the repository root: python examples/03_exact_recognition.py
"""

import sympy as sp

from algroots import polysolve
from algroots.recognition import recognize_system_roots


def main():
    x = sp.Symbol("x")
    numerical = polysolve((x**2 - 2,), (x,), digits=60, recognize=False)
    recognized = recognize_system_roots(numerical, max_degree=2, require_certified=True)
    assert len(recognized) == 2 and all(root.certified for root in recognized)
    print("Recognized roots:", recognized)
    print("Recognition does not change global evidence:", numerical.completeness.status)


if __name__ == "__main__":
    main()
