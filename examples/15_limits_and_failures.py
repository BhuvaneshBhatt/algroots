"""Handle unsupported problems and proof budgets.

Guarantee: Explicit refusal; no weakened proof.
Run from the repository root: python examples/15_limits_and_failures.py
"""

import sympy as sp

from algroots import NotZeroDimensionalError, polysolve


def main():
    x, y = sp.symbols("x y")
    try:
        polysolve((x * y,), (x, y), recognize=False)
    except NotZeroDimensionalError as error:
        print("Positive-dimensional input refused:", error)
    else:
        raise AssertionError("a finite all-roots list should not be returned")
    result = polysolve(
        (x**2 - 2,), (x,), recognize=False, certify="auto", certification_max_dimension=1
    )
    assert all(attempt.status == "unavailable" for attempt in result.root_certifications)
    print("Numerical roots remain available:", result.roots)
    print("Proof budget refusal:", result.root_certifications[0].error)


if __name__ == "__main__":
    main()
