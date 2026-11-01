"""Exact relative coordinates for extensions of algebraic-number constants."""

from functools import cached_property

import sympy as sp


def number_degree(domain):
    return 1 if domain == sp.QQ else int(domain.mod.degree())


def extension_root(poly):
    """One root of an irreducible polynomial, preserving its specified embedding."""
    K = poly.domain
    if poly.degree() == 1:
        return K, K.to_sympy(-poly.rep.TC() / poly.rep.LC())
    if poly.degree() == 2:
        a, b, c = poly.all_coeffs()
        root = (-b + sp.sqrt(b * b - 4 * a * c)) / (2 * a)
    else:
        root = sp.polys.polytools.all_roots(poly, extension=True, radicals=False)[0]
    # AlgebraicField.algebraic_field may retain nested AlgebraicNumber atoms.
    # Flatten their roots before compositum construction; otherwise converting
    # its own output back can trigger enormous minimal-polynomial computations.
    while root.has(sp.AlgebraicNumber):
        root = root.xreplace({a: a.as_expr() for a in root.atoms(sp.AlgebraicNumber)})
    base_root = None if K == sp.QQ else K.ext.as_expr()
    while base_root is not None and base_root.has(sp.AlgebraicNumber):
        base_root = base_root.xreplace(
            {a: a.as_expr() for a in base_root.atoms(sp.AlgebraicNumber)}
        )
    E = sp.QQ.algebraic_field(root) if K == sp.QQ else sp.QQ.algebraic_field(base_root, root)
    return E, root


class ConstantCoordinates:
    """An explicit K-basis of E and exact coordinates, even when K != QQ."""

    def __init__(self, extension, base):
        self.extension, self.base = extension, base
        self.absolute_degree = number_degree(extension)
        self.base_degree = number_degree(base)
        self.degree = self.absolute_degree // self.base_degree
        self.theta = sp.Integer(0) if extension == base else extension.ext.as_expr()
        self.alpha = sp.Integer(0) if base == sp.QQ else base.ext.as_expr()
        # Conversion also checks the chosen embedding of K into E.
        extension.convert(self.alpha)

    def _absolute(self, value):
        value = self.extension.convert(value)
        if self.extension == sp.QQ:
            return sp.Matrix([sp.QQ.to_sympy(value)])
        coefficients = list(reversed(value.to_list()))
        coefficients += [sp.QQ.zero] * (self.absolute_degree - len(coefficients))
        return sp.Matrix([sp.QQ.to_sympy(c) for c in coefficients])

    @cached_property
    def _inverse(self):
        columns = [
            self._absolute(self.theta**i * self.alpha**j)
            for i in range(self.degree)
            for j in range(self.base_degree)
        ]
        return sp.Matrix.hstack(*columns).inv()

    def coordinates(self, value):
        if self.extension == self.base:
            return (self.base.to_sympy(self.base.convert(value)),)
        values = self._inverse * self._absolute(value)
        return tuple(
            self.base.to_sympy(
                self.base.convert(
                    sum(
                        values[i * self.base_degree + j] * self.alpha**j
                        for j in range(self.base_degree)
                    )
                )
            )
            for i in range(self.degree)
        )

    def polynomial(self, symbol):
        coefficients = self.coordinates(self.theta**self.degree)
        return symbol**self.degree - sum(c * symbol**i for i, c in enumerate(coefficients))
