"""Build one exact recursive chain and distinguish resolved and open leaves."""

import sympy as sp

from algroots.functional_decomposition import (
    algebraic_decomposition_chain,
    rational_decomposition_chain,
)


def main():
    x, y, z = sp.symbols("x y z")
    chain = rational_decomposition_chain(x**8, x)
    assert chain.complete and chain.verify() and len(chain.components) == 3
    assert chain.compose() == x**8
    print(
        "Rational component degrees:",
        [sp.degree(c.expression, c.variables[0]) for c in chain.components],
    )

    multi = rational_decomposition_chain((x**2 + y**2) ** 4, (x, y))
    assert multi.complete and multi.verify() and len(multi.components) == 3
    print("Multivariate chain:", multi.status)

    implicit = algebraic_decomposition_chain(x**8 - z, (x, z))
    assert implicit.complete and implicit.verify() and len(implicit.components) == 3
    assert implicit.resultant() == implicit.unit * implicit.relation**implicit.exponent
    print("Implicit chain:", implicit.status, "resultant power", implicit.exponent)

    limited = rational_decomposition_chain(x**8, x, max_depth=1)
    assert limited.verify() and not limited.complete
    print("Open leaves:", [leaf.stop_reason for leaf in limited.leaves])


if __name__ == "__main__":
    main()
