"""Construct subfields over an existing base and over chosen constants.

Guarantee: Exact membership and replayable embeddings; generic rational maps.
Run from the repository root: python examples/generated_subfields.py
"""

import sympy as sp

from algroots.function_fields import coefficient_field
from algroots.quotient import QuotientAlgebra


def main():
    x, z, u, v, T = sp.symbols("x z u v T")
    A = QuotientAlgebra.from_polynomials([x**4 - z], (x,), domain=sp.QQ.frac_field(z))
    E = A.subfield([x * x], generator_symbols=(u,))
    assert E.dimension == 2 and E.verify() and not E.contains(x)
    member = E.express(1 / (x * x + 1))
    assert member.verify()
    primitive = E.primitive_element(T)
    assert primitive.verify() and primitive.to_source(x * x + 1) == T + 1
    print("Fixed-base subfield dimension:", E.dimension)
    print("Constructive inverse:", member.expression)

    B = QuotientAlgebra.from_polynomials([x * x - z * x + z**3], (x,), domain=sp.QQ.frac_field(z))
    C = coefficient_field([x], ambient=B, generator_symbols=(u,))
    assert C.verify() and C.contains(x) and not C.contains(z)
    assert B.subfield([x]).contains(z)
    print("Coefficient field over constants:", C.presentation)

    F = coefficient_field(
        [z * z, z**3], ambient=sp.QQ.frac_field(z), generator_symbols=(u, v), parameter=T
    )
    parametrization = F.rational_parametrization()
    assert parametrization.verify()
    assert parametrization.generator == z
    assert parametrization.generator_expression == v / u
    assert parametrization.coefficient_images == (T * T, T**3)
    print("Recovered rational generator:", parametrization.generator)
    print("Inverse chart guard:", parametrization.inverse_guard)


if __name__ == "__main__":
    main()
