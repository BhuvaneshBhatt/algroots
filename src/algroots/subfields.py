"""Constructive subfields of finite characteristic-zero quotient fields."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import sympy as sp
from sympy.polys.matrices import DomainMatrix

from .domains import domain_matrix
from .errors import QuotientAlgebraError
from .quotient import QuotientAlgebra


def field_normal_form(ambient: QuotientAlgebra, expression) -> sp.Expr:
    """Reduce a rational expression, requiring its denominator to be a unit."""
    expression = sp.sympify(expression)
    if expression.has(sp.Float):
        raise QuotientAlgebraError("field elements must have exact coefficients")
    numerator, denominator = sp.fraction(sp.cancel(expression))
    if denominator == 1:
        return ambient.normal_form(numerator)
    return ambient.normal_form(numerator * ambient.inverse(denominator))


def _coordinates(columns: DomainMatrix, vector: DomainMatrix):
    """Unique span coordinates, or None for a vector outside the column space."""
    reduced, pivots = columns.hstack(vector).rref()
    n = columns.shape[1]
    if n in pivots:
        return None
    if len(pivots) != n:
        raise QuotientAlgebraError("subfield embedding columns are not independent")
    rows = reduced.to_list()
    return tuple(rows[i][n] for i in range(n))


@dataclass(frozen=True)
class SubfieldMembership:
    """Constructive positive or negative membership with exact replay."""

    subfield: FiniteSubfield
    element: sp.Expr
    coordinates: tuple[sp.Expr, ...] | None

    @property
    def belongs(self) -> bool:
        """Whether the element has an exact expression in the generated subfield."""
        return self.coordinates is not None

    def __bool__(self):
        return self.belongs

    @property
    def expression(self) -> sp.Expr | None:
        """Expression in the subfield's formal generator symbols, when a member."""
        if self.coordinates is None:
            return None
        return sp.expand(
            sum(
                c * b
                for c, b in zip(self.coordinates, self.subfield.basis_expressions, strict=True)
            )
        )

    def verify(self) -> bool:
        """Replay the span decision and, for a member, its explicit expression."""
        E = self.subfield
        if not E.verify():
            return False
        expected = E.express(self.element)
        if self.coordinates != expected.coordinates:
            return False
        if self.belongs:
            return E.ambient.normal_form(self.element - E.to_ambient(self.expression)) == 0
        return not expected.belongs


@dataclass(frozen=True)
class FiniteSubfield:
    """The generated field over the ambient quotient's *existing* base field.

    ``basis`` consists of ambient representatives, with expressions in the
    supplied generators recorded independently in ``basis_expressions``.
    This construction does not remove transcendental coefficient parameters.
    """

    ambient: QuotientAlgebra
    generators: tuple[sp.Expr, ...]
    generator_symbols: tuple[sp.Symbol, ...]
    basis: tuple[sp.Expr, ...]
    basis_expressions: tuple[sp.Expr, ...]

    @classmethod
    def generate(cls, ambient, generators, *, over=None, generator_symbols=None):
        """Construct the subfield closure and its exact ambient embedding."""
        if over is not None and over != ambient.domain:
            raise QuotientAlgebraError(
                "fixed-base subfields require over=ambient.domain; use coefficient_field to change constants"
            )
        if not ambient.is_field:
            raise QuotientAlgebraError("subfield construction requires a certified ambient field")
        generators = tuple(field_normal_form(ambient, g) for g in generators)
        symbols = (
            tuple(sp.Dummy(f"subfield_{i}") for i in range(len(generators)))
            if generator_symbols is None
            else tuple(generator_symbols)
        )
        if (
            len(symbols) != len(generators)
            or any(not isinstance(s, sp.Symbol) for s in symbols)
            or len(set(symbols)) != len(symbols)
        ):
            raise QuotientAlgebraError("one distinct formal Symbol is required per generator")
        forbidden = set(ambient.variables) | set(getattr(ambient.domain, "symbols", ()))
        if set(symbols) & forbidden:
            raise QuotientAlgebraError(
                "formal generator symbols must be distinct from ambient variables and parameters"
            )
        basis, words = [sp.Integer(1)], [sp.Integer(1)]
        columns = domain_matrix(ambient.coordinate_vector(1), ambient.domain)
        cursor = 0
        while cursor < len(basis):
            for generator, symbol in zip(generators, symbols, strict=True):
                candidate = ambient.normal_form(basis[cursor] * generator)
                vector = domain_matrix(ambient.coordinate_vector(candidate), ambient.domain)
                if _coordinates(columns, vector) is None:
                    basis.append(candidate)
                    words.append(words[cursor] * symbol)
                    columns = columns.hstack(vector)
            cursor += 1
        return cls(ambient, generators, symbols, tuple(basis), tuple(words))

    @property
    def domain(self):
        """The base coefficient field of this algebraic presentation."""
        return self.ambient.domain

    @property
    def dimension(self):
        """The vector-space dimension, including nonreduced multiplicities."""
        return len(self.basis)

    @cached_property
    def embedding_matrix(self):
        """Exact injection of subfield basis coordinates into ambient coordinates."""
        return sp.ImmutableMatrix.hstack(
            *(sp.ImmutableMatrix(self.ambient.coordinate_vector(b)) for b in self.basis)
        )

    @cached_property
    def _domain_embedding(self):
        return domain_matrix(self.embedding_matrix, self.domain)

    def express(self, element) -> SubfieldMembership:
        """Return exact membership coordinates and their ambient embedding evidence."""
        element = field_normal_form(self.ambient, element)
        vector = domain_matrix(self.ambient.coordinate_vector(element), self.domain)
        values = _coordinates(self._domain_embedding, vector)
        coefficients = None if values is None else tuple(self.domain.to_sympy(c) for c in values)
        return SubfieldMembership(self, element, coefficients)

    def contains(self, element) -> bool:
        """Test exact membership in the generated coefficient field."""
        return self.express(element).belongs

    def to_ambient(self, expression) -> sp.Expr:
        """Embed an expression from this presentation in its original ambient field."""
        expression = sp.sympify(expression)
        allowed = set(self.generator_symbols) | set().union(
            *(s.free_symbols for s in getattr(self.domain, "symbols", ()))
        )
        if expression.free_symbols - allowed:
            raise QuotientAlgebraError(
                "expression must use only formal generators and base-field parameters"
            )
        value = expression.xreplace(dict(zip(self.generator_symbols, self.generators, strict=True)))
        return field_normal_form(self.ambient, value)

    def verify(self) -> bool:
        """Replay generation, independence, closure and the recorded embedding."""
        if len(self.generators) != len(self.generator_symbols) or len(
            set(self.generator_symbols)
        ) != len(self.generator_symbols):
            return False
        if not self.ambient.is_field or len(self.basis) != len(self.basis_expressions):
            return False
        if self._domain_embedding.rank() != self.dimension or not self.basis or self.basis[0] != 1:
            return False
        for value, word in zip(self.basis, self.basis_expressions, strict=True):
            if self.ambient.normal_form(value - self.to_ambient(word)) != 0:
                return False
            if any(not self.contains(value * g) for g in self.generators):
                return False
        return all(self.contains(g) for g in self.generators)

    def primitive_element(self, parameter: sp.Symbol | None = None) -> PrimitiveSubfield:
        """Construct a simple presentation and both directions of its embedding.

        In characteristic zero a moment-curve linear combination of r generators
        has at most (r-1)*n*(n-1)/2 bad integer parameters. This exact bound avoids
        an arbitrary search cutoff, using n equal to the *subfield* dimension.
        """
        parameter = sp.Dummy("subfield_theta") if parameter is None else parameter
        self.ambient._polynomial_parameter(parameter)
        n, r = self.dimension, len(self.generators)
        for choice in range(max(0, r - 1) * n * (n - 1) // 2 + 1):
            theta = (
                sum(choice**i * g for i, g in enumerate(self.generators)) if r else sp.Integer(0)
            )
            theta = self.ambient.normal_form(theta)
            polynomial = self.ambient.minimal_polynomial(theta, parameter)
            if polynomial.degree() != n:
                continue
            field = QuotientAlgebra.from_polynomials(
                [polynomial.as_expr()], (parameter,), domain=self.domain
            )
            columns = domain_matrix(
                sp.Matrix.hstack(
                    *(
                        self.ambient.coordinate_vector(self.ambient.normal_form(theta**i))
                        for i in range(n)
                    )
                ),
                self.domain,
            )
            images = []
            for generator in self.generators:
                coords = _coordinates(
                    columns, domain_matrix(self.ambient.coordinate_vector(generator), self.domain)
                )
                images.append(
                    sum(self.domain.to_sympy(c) * parameter**i for i, c in enumerate(coords))
                )
            return PrimitiveSubfield(self, field, theta, tuple(images))
        raise QuotientAlgebraError(
            "primitive-element search exhausted its characteristic-zero bound"
        )


@dataclass(frozen=True)
class PrimitiveSubfield:
    """Simple subfield presentation with constructive forward and inverse maps."""

    subfield: FiniteSubfield
    field: QuotientAlgebra
    ambient_generator: sp.Expr
    generator_images: tuple[sp.Expr, ...]

    @cached_property
    def _powers(self):
        A = self.subfield.ambient
        theta = self.ambient_generator
        return domain_matrix(
            sp.Matrix.hstack(
                *(A.coordinate_vector(A.normal_form(theta**i)) for i in range(self.field.dimension))
            ),
            A.domain,
        )

    def to_ambient(self, value):
        """Embed an expression from this presentation in its original ambient field."""
        value = field_normal_form(self.field, value)
        return field_normal_form(
            self.subfield.ambient, value.subs(self.field.variables[0], self.ambient_generator)
        )

    def to_source(self, value):
        """Express an ambient element in the coordinates of this presentation."""
        A = self.subfield.ambient
        value = field_normal_form(A, value)
        coordinates = _coordinates(
            self._powers, domain_matrix(A.coordinate_vector(value), A.domain)
        )
        if coordinates is None:
            raise QuotientAlgebraError("element does not belong to the subfield")
        return sum(
            A.domain.to_sympy(c) * self.field.variables[0] ** i for i, c in enumerate(coordinates)
        )

    def verify(self):
        """Replay the simple field presentation and both directions of its embedding."""
        if (
            self.field.domain != self.subfield.domain
            or len(self.field.variables) != 1
            or not self.field.is_field
        ):
            return False
        if not self.subfield.verify() or self.field.dimension != self.subfield.dimension:
            return False
        if self._powers.rank() != self.field.dimension:
            return False
        if any(
            field_normal_form(
                self.subfield.ambient,
                p.as_expr().subs(self.field.variables[0], self.ambient_generator),
            )
            != 0
            for p in self.field.groebner_basis.polys
        ):
            return False
        if len(self.generator_images) != len(self.subfield.generators):
            return False
        A = self.subfield.ambient
        return all(
            A.normal_form(self.to_ambient(image) - generator) == 0
            for image, generator in zip(
                self.generator_images, self.subfield.generators, strict=True
            )
        )


__all__ = ["FiniteSubfield", "SubfieldMembership", "PrimitiveSubfield"]
