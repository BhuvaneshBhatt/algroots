"""Compute normal forms, eliminate a variable, and change monomial order."""

import sympy as sp

from algroots import groebner_backends, groebner_basis


def main():
    x, y = sp.symbols("x y")
    equations = [x + y - 3, x * y - 2]
    graded = groebner_basis(equations, [x, y], backend="sympy")
    lexicographic = graded.fglm("lex")
    assert list(lexicographic) == [x + y - 3, y**2 - 3 * y + 2]
    assert lexicographic.reduce(y**2 - 3 * y + 2)[1] == 0
    assert lexicographic.reduce(y)[1] == y
    assert groebner_backends()["sympy"].available
    print("Elimination basis:", list(lexicographic))


if __name__ == "__main__":
    main()
