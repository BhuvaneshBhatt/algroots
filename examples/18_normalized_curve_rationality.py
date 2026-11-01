"""Distinguish normalized genus, birational maps, and ground-field rationality.

Guarantee: Exact normalized genus and verified maps; QQ conic obstruction.
Run from the repository root: python examples/18_normalized_curve_rationality.py
"""

import sympy as sp

from algroots.curves import PlaneCurve


def main():
    x, y, t = sp.symbols("x y t")
    cusp = PlaneCurve.from_polynomial(y * y - x**3, (x, y)).rational_parametrization(t)
    assert cusp.rational and cusp.verify()
    assert cusp.parametrization.affine_coordinates == (t * t, t**3)
    print("Cusp normalization:", cusp.parametrization.affine_coordinates)
    singular = PlaneCurve.from_polynomial(y * y - x**5, (x, y)).normalization()
    assert singular.genus == 0 and singular.delta_sum == 6 and singular.verify()
    print(
        "Quintic arithmetic genus, delta, normalized genus:",
        singular.arithmetic_genus,
        singular.delta_sum,
        singular.genus,
    )
    elliptic = PlaneCurve.from_polynomial(y * y - x**3 + x, (x, y)).rational_parametrization(t)
    assert elliptic.status == "positive_genus" and elliptic.verify()
    conic = PlaneCurve.from_polynomial(x * x + y * y - 3, (x, y))
    obstruction = conic.rational_parametrization(t)
    assert obstruction.status == "no_ground_field_point" and obstruction.verify()
    extended = conic.rational_parametrization(t, extend_constants=True)
    assert extended.status == "rational_over_extension" and extended.verify()
    print("Conic obstruction over QQ; splitting field:", obstruction.splitting_field)


if __name__ == "__main__":
    main()
