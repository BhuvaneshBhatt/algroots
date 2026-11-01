"""Exact polynomial operations for multivariate constructive Lüroth maps.

Generator discovery follows Gutiérrez–Rubio–Sevilla (2002), Algorithm 3.1.
The ambient variables and their independent comparison copies stay distinct.
Membership uses the prime generic graph p(x)-t*q(x), rather than a finite
quotient of the multivariate ambient field.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import sympy as sp
from sympy.polys.polyerrors import CoercionFailed

from .domains import field_element, groebner, polynomial
from .errors import QuotientAlgebraError
from .quotient import QuotientAlgebra


def rational_values(ambient, values):
    """Normalize exact rational functions, rejecting undeclared coefficients."""
    normalized = []
    for value in values:
        value = value.as_expr() if isinstance(value, sp.Poly) else sp.sympify(value)
        if value.has(sp.Float):
            raise QuotientAlgebraError("function-field elements must be exact")
        try:
            normalized.append(ambient.to_sympy(field_element(value, ambient)))
        except (
            CoercionFailed,
            sp.PolynomialError,
            ValueError,
            TypeError,
            ZeroDivisionError,
        ) as exc:
            raise QuotientAlgebraError(
                "an exact rational function over the declared constants and variables is required"
            ) from exc
    return tuple(normalized)


def separated_gcd(ambient, values, comparison):
    """Monic gcd of p_i(Y)-f_i(X)*q_i(Y), in graded lexicographic order."""
    renaming = dict(zip(ambient.symbols, comparison, strict=True))
    gcd = None
    for value in values:
        numerator, denominator = sp.fraction(value)
        difference = polynomial(
            numerator.xreplace(renaming) - value * denominator.xreplace(renaming),
            comparison,
            ambient,
        )
        if difference.is_zero:
            continue
        gcd = difference if gcd is None else gcd.gcd(difference)
        if gcd.total_degree() == 0:
            return sp.Poly(1, *comparison, domain=ambient)
    if gcd is None:
        return None
    leading = gcd.terms(order="grlex")[0][1]
    monic = gcd.mul_ground(1 / leading)
    # Re-embed expression coefficients to cancel algebraic-number units.
    # SymPy's FractionField may retain equivalent noncanonical units after gcd.
    return sp.Poly.from_dict(
        {powers: field_element(coefficient, ambient) for powers, coefficient in monic.terms()},
        comparison,
        domain=ambient,
    )


def smallest_coefficient(gcd, ambient):
    """A nonconstant coefficient of least rational total degree, up to sign."""
    variables, constants = ambient.symbols, ambient.domain

    def degree(value):
        numerator, denominator = sp.fraction(value)
        return max(
            sp.Poly(numerator, *variables, domain=constants).total_degree(),
            sp.Poly(denominator, *variables, domain=constants).total_degree(),
        )

    candidates = [
        ambient.to_sympy(field_element(-coefficient, ambient))
        for _, coefficient in gcd.terms(order="grlex")
        if coefficient.free_symbols & set(variables)
    ]
    if not candidates:
        raise QuotientAlgebraError("Luroth gcd has no nonconstant coefficient")
    return min(candidates, key=degree)


def inverse_coordinates(images, parameter, constants, symbols):
    """Express the parameter in its generating images by finite field closure.

    Work over K(one input), where K(parameter) is finite of known rational
    degree. The existing finite-subfield engine expresses the parameter in the
    other inputs using exact linear algebra, avoiding graph elimination among
    several formal inputs (and its coefficient growth over number fields).
    """

    def degree(value):
        numerator, denominator = sp.fraction(value)
        return max(sp.degree(numerator, parameter), sp.degree(denominator, parameter))

    index = min(
        (i for i, image in enumerate(images) if parameter in image.free_symbols),
        key=lambda i: degree(images[i]),
    )
    coordinate = sp.Dummy("luroth_inverse_coordinate")
    numerator, denominator = sp.fraction(images[index])
    relation = (numerator - symbols[index] * denominator).xreplace({parameter: coordinate})
    ambient = QuotientAlgebra.from_polynomials(
        [relation], (coordinate,), domain=constants.frac_field(symbols[index])
    )
    others = tuple(i for i in range(len(images)) if i != index)
    field = ambient.subfield(
        [images[i].xreplace({parameter: coordinate}) for i in others],
        generator_symbols=tuple(symbols[i] for i in others),
    )
    result = field.express(coordinate)
    if not result.belongs:
        raise QuotientAlgebraError("Luroth images did not generate the rational parameter")
    return rational_values(constants.frac_field(*symbols), (result.expression,))[0]


@dataclass(frozen=True)
class RationalCoordinateMap:
    """Exact membership and coordinates in K(h(X)) inside K(X).

    For reduced h=p/q, p(X)-t*q(X) generates a prime ideal in K(t)[X].
    For v=N/D, v is in K(h) precisely when the normal forms of N and D
    are proportional over K(t). The normal form of D cannot vanish: the
    inclusion of K[X] in the generic graph's fraction field is injective.
    This works in positive-dimensional ambient spaces as well as K(x).
    """

    ambient: object
    generator: sp.Expr
    parameter: sp.Symbol

    @cached_property
    def basis(self):
        (generator,) = rational_values(self.ambient, (self.generator,))
        numerator, denominator = sp.fraction(generator)
        return groebner(
            [numerator - self.parameter * denominator],
            self.ambient.symbols,
            self.ambient.domain.frac_field(self.parameter),
        )

    def express(self, value):
        (value,) = rational_values(self.ambient, (value,))
        numerator, denominator = sp.fraction(value)
        base = self.basis.domain
        numerator = polynomial(
            self.basis.reduce(polynomial(numerator, self.ambient.symbols, base))[1],
            self.ambient.symbols,
            base,
        )
        denominator = polynomial(
            self.basis.reduce(polynomial(denominator, self.ambient.symbols, base))[1],
            self.ambient.symbols,
            base,
        )
        if denominator.is_zero:
            raise QuotientAlgebraError("generic graph has a vanishing query denominator")
        if numerator.is_zero:
            return sp.S.Zero
        ratio = base.to_sympy(
            field_element(numerator.LC(), base) / field_element(denominator.LC(), base)
        )
        if not (numerator - denominator.mul_ground(ratio)).is_zero:
            return None
        return ratio
