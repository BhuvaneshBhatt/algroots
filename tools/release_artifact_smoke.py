"""Correctness smoke test for an installed algroots distribution."""

from __future__ import annotations

import sympy as sp

from algroots import (
    algsolve,
    compute_border_basis,
    polysolve,
)
from algroots.numerical import flint_available


def _close(value, expected, tolerance=1e-18):
    return abs(complex(sp.N(value - expected, 30))) < tolerance


def main() -> None:
    if not flint_available():
        raise RuntimeError("release artifact smoke test requires python-flint")

    x, y = sp.symbols("x y")

    polynomial = polysolve(
        [x**2 - 2, y**2 - 3],
        (x, y),
        method="action",
        digits=45,
    )
    assert polynomial.method == "action_matrix"
    assert polynomial.quotient_dimension == 4
    assert len(polynomial.roots) == 4
    assert polynomial.max_relative_residual < sp.Float("1e-25")
    assert polynomial.recognition_attempted
    assert polynomial.recognized_roots is not None
    assert all(root.certified for root in polynomial.recognized_roots)

    rur = polysolve(
        [x**2 - sp.sqrt(2), y - x],
        (x, y),
        method="rur",
        digits=40,
        recognize=False,
    )
    assert rur.method == "rational_univariate"
    assert len(rur.roots) == 2
    assert rur.rational_univariate_representation is not None
    assert rur.rational_univariate_representation.defining_polynomial.domain.is_AlgebraicField

    homotopy = polysolve(
        [x**2 - 1, y**2 - 1],
        (x, y),
        method="homotopy",
        digits=30,
        recognize=False,
        homotopy_gamma_attempts=2,
    )
    assert homotopy.method == "total_degree_homotopy"
    assert len(homotopy.roots) == 4
    assert homotopy.homotopy_paths_total == 4
    assert homotopy.homotopy_paths_succeeded == 4
    assert homotopy.homotopy_paths_failed == 0

    border = compute_border_basis((x**2 - 1, y - x), (x, y))
    assert border.dimension == 2
    assert border.has_commuting_multiplication_matrices()

    radical = algsolve(
        [sp.sqrt(x) - (x - 2)],
        (x,),
        digits=45,
    )
    assert len(radical.roots) == 1
    assert _close(radical.roots[0][0], 4)

    rational = sp.Mul(
        x**2 - 1,
        sp.Pow(x - 1, -1, evaluate=False),
        evaluate=False,
    )
    filtered = algsolve([rational], (x,), digits=40)
    assert len(filtered.roots) == 1
    assert _close(filtered.roots[0][0], -1)

    print("algroots installed-artifact smoke test passed")


if __name__ == "__main__":
    main()
