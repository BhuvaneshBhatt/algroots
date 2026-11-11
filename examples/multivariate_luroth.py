"""Recover a multivariate Lüroth generator and verify field membership.

Guarantee: Exact generator maps and transcendence-degree-one decisions.
Run from the repository root: python examples/multivariate_luroth.py
"""

import sympy as sp

from algroots.function_fields import coefficient_field, luroth_generator


def main():
    x, y, t, u, v = sp.symbols("x y t u v")
    h = x / y
    result = luroth_generator([h**2, h**3], variables=(x, y), parameter=t, generator_symbols=(u, v))
    assert result.verify() and result.generator == h
    assert result.generator_expression == v / u
    assert result.coefficient_images == (t**2, t**3)
    assert result.express(1 / (h + 1)).expression == 1 / (t + 1)
    assert not result.contains(x) and result.express(x).verify()
    print("Generated field:", result.generator, "inverse:", result.generator_expression)
    proper = coefficient_field([h**2, h**4], ambient=sp.QQ.frac_field(x, y))
    assert proper.verify() and proper.contains(h**2) and not proper.contains(h)
    print("Proper subfield excludes x/y:", proper.generator)
    independent = luroth_generator([x, y], variables=(x, y))
    assert independent.verify() and independent.status == "higher_transcendence_degree"
    assert independent.transcendence_degree is None
    print("Independent inputs have transcendence degree at least two.")


if __name__ == "__main__":
    main()
