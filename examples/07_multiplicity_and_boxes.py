"""Certify individual multiplicities with automatic boxes.

Guarantee: Exact isolated endpoint proofs.
Run from the repository root: python examples/07_multiplicity_and_boxes.py
"""

import sympy as sp

from algroots import polysolve


def main():
    x, y = sp.symbols("x y")
    result = polysolve((x**2, y**2 - 1), (x, y), recognize=False, certify="required")
    assert result.total_multiplicity == 4 and result.geometric_solution_count == 2
    assert not result.is_radical and result.has_multiple_roots
    for attempt in result.root_certifications:
        assert attempt.status == "certified" and attempt.certificate.multiplicity == 2
        assert attempt.certificate.verify()
        print(
            "Point, multiplicity, attempts:",
            attempt.certificate.point,
            attempt.certificate.multiplicity,
            attempt.box_attempts,
        )
    print("Endpoint proofs are separate from result-wide evidence:", result.completeness.status)


if __name__ == "__main__":
    main()
