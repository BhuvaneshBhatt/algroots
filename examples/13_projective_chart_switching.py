"""Track a point through an automatic chart change.

Guarantee: Numerical projective tracking only.
Run from the repository root: python examples/13_projective_chart_switching.py
"""

import sympy as sp

from algroots.endgames import projective_homotopy
from algroots.projective_tracking import track_projective_path


def main():
    x, t = sp.symbols("x t")
    projective = projective_homotopy((x - t,), (x,), t, patch=(1, 0))
    path = track_projective_path(projective, projective.lift((1,)), t_start=1, t_end=0)
    assert path.success and path.classification == "finite" and path.chart_switches
    assert abs(path.affine_endpoint()[0]) < sp.Rational(1, 10**15)
    print("Chart switches:", path.chart_switches)
    print("Finite numerical endpoint:", path.affine_endpoint())


if __name__ == "__main__":
    main()
