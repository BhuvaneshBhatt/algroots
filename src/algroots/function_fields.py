"""Coefficient-generated fields over constants and constructive Lüroth maps.

One-variable rational ambient fields and multivariate rational fields of
transcendence degree at most one have constructive generators over supported
characteristic-zero constants. Finite algebraic ambient fields additionally support saturated image
presentations and membership after rebasing to a nonconstant coefficient. This
module connects algebraic coefficient fields to normalized plane-curve genus
and genus-zero rationality through ``algroots.curves``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import sympy as sp
from sympy.polys.domains import Domain
from sympy.polys.polyerrors import CoercionFailed

from ._coefficient_curves import CoefficientCurveModel, CoefficientFieldParametrization
from ._luroth import (
    RationalCoordinateMap,
    inverse_coordinates,
    rational_values,
    separated_gcd,
    smallest_coefficient,
)
from .domains import field_element, groebner, polynomial
from .errors import QuotientAlgebraError
from .quotient import QuotientAlgebra, _require_exact_domain
from .subfields import field_normal_form


def _constant_domain(domain):
    domain = _require_exact_domain(domain)
    if getattr(domain, "is_FractionField", False):
        raise QuotientAlgebraError("constants must be a rational or algebraic number field")
    return domain


def _ambient_data(ambient, constants):
    base = ambient.domain if isinstance(ambient, QuotientAlgebra) else ambient
    if (
        not isinstance(base, Domain)
        or not getattr(base, "is_FractionField", False)
        or len(base.symbols) != 1
    ):
        raise QuotientAlgebraError("a one-variable rational-function base is required")
    if _constant_domain(base.domain) != constants:
        raise QuotientAlgebraError(
            "constants must match the ambient rational-function constant field"
        )
    if isinstance(ambient, QuotientAlgebra) and not ambient.is_field:
        raise QuotientAlgebraError("coefficient fields require a certified ambient field")
    return base, base.symbols[0]


def _normal(ambient, value):
    if isinstance(ambient, QuotientAlgebra):
        return field_normal_form(ambient, value)
    value = sp.sympify(value)
    if value.has(sp.Float):
        raise QuotientAlgebraError("function-field elements must be exact")
    return ambient.to_sympy(field_element(value, ambient))


def _formal_symbols(values, symbols, forbidden):
    symbols = (
        tuple(sp.Dummy(f"coefficient_{i}") for i in range(len(values)))
        if symbols is None
        else tuple(symbols)
    )
    if (
        len(symbols) != len(values)
        or len(set(symbols)) != len(symbols)
        or any(not isinstance(s, sp.Symbol) for s in symbols)
    ):
        raise QuotientAlgebraError("one distinct formal Symbol is required per coefficient")
    if set(symbols) & set(forbidden):
        raise QuotientAlgebraError("formal coefficients must be distinct from ambient symbols")
    return symbols


def _fraction(value):
    return sp.fraction(sp.together(sp.cancel(value)))


def _image_relations(ambient, constants, values, symbols):
    """Kernel of the rational graph map, saturated at all presentation poles."""
    base, z = _ambient_data(ambient, constants)
    source_symbols = (*ambient.variables, z) if isinstance(ambient, QuotientAlgebra) else (z,)
    local = tuple(sp.Dummy(f"source_{i}") for i in range(len(source_symbols)))
    renaming = dict(zip(source_symbols, local, strict=True))
    equations, denominators = [], []
    if isinstance(ambient, QuotientAlgebra):
        for poly in ambient.groebner_basis.polys:
            num, den = _fraction(poly.as_expr())
            equations.append(num.xreplace(renaming))
            denominators.append(den.xreplace(renaming))
    for value, symbol in zip(values, symbols, strict=True):
        num, den = _fraction(value)
        equations.append(symbol * den.xreplace(renaming) - num.xreplace(renaming))
        denominators.append(den.xreplace(renaming))
    saturation = sp.Dummy("image_saturation")
    equations.append(saturation * sp.prod(denominators) - 1)
    basis = groebner(equations, (saturation, *local, *symbols), constants)
    outputs = set(symbols)
    return tuple(p.as_expr() for p in basis.polys if p.as_expr().free_symbols <= outputs)


def _presentation(constants, relations, symbols, base_index):
    if base_index is None:
        base, variables = constants, tuple(symbols)
    else:
        base = constants.frac_field(symbols[base_index])
        variables = tuple(s for i, s in enumerate(symbols) if i != base_index)
    if not variables:
        return base
    if not relations:
        return None
    basis = groebner(relations, variables, base)
    if not basis.is_zero_dimensional:
        return None
    return QuotientAlgebra.from_groebner_basis(basis)


def _degree(presentation):
    return presentation.dimension if isinstance(presentation, QuotientAlgebra) else 1


def _image_membership(field, value):
    value = _normal(field.ambient, value)
    query = sp.Dummy("coefficient_query")
    symbols = (*field.generator_symbols, query)
    relations = _image_relations(
        field.ambient, field.constants, (*field.generators, value), symbols
    )
    if field.base_index is None:
        base = field.constants
    else:
        base = field.constants.frac_field(field.generator_symbols[field.base_index])
    others = tuple(s for i, s in enumerate(field.generator_symbols) if i != field.base_index)
    if not relations:
        return None
    # The query is the leading lexicographic variable, so a degree-one extension
    # reduces it to a rational function of the original coefficient coordinates.
    basis = groebner(relations, (query, *others), base)
    if not basis.is_zero_dimensional:
        return None
    extension = QuotientAlgebra.from_groebner_basis(basis)
    if extension.dimension != field.degree:
        return None
    expression = extension.normal_form(query)
    if query in expression.free_symbols:
        raise QuotientAlgebraError("membership projection did not eliminate its query coordinate")
    return sp.cancel(expression)


@dataclass(frozen=True)
class CoefficientMembership:
    """Membership in a constant-generated field, with a constructive expression."""

    field: object
    element: sp.Expr
    expression: sp.Expr | None

    @property
    def belongs(self):
        """Whether the element has an exact expression in the generated subfield."""
        return self.expression is not None

    def __bool__(self):
        return self.belongs

    def verify(self):
        """Check the proposed coefficient expression against the original ambient element."""
        expected = self.field.express(self.element)
        if expected.belongs != self.belongs:
            return False
        if not self.belongs:
            return self.field.verify()
        return (
            self.field.verify()
            and _normal(self.field.ambient, self.field.to_ambient(self.expression) - self.element)
            == 0
        )


@dataclass(frozen=True)
class AlgebraicCoefficientField:
    """Saturated presentation of K(coefficients), embedded in an ambient field.

    If a coefficient is transcendental over K, ``base_index`` names that
    coordinate and the presentation is finite over K(coefficient). The original
    ambient parameter is never adjoined automatically.
    """

    ambient: QuotientAlgebra
    constants: Domain
    generators: tuple[sp.Expr, ...]
    generator_symbols: tuple[sp.Symbol, ...]
    base_index: int | None
    relations: tuple[sp.Expr, ...]
    presentation: object

    @property
    def transcendence_degree(self):
        """Transcendence degree of the generated field over its constants."""
        return int(self.base_index is not None)

    @property
    def degree(self):
        """Degree of the finite presentation over its chosen transcendence base."""
        return _degree(self.presentation)

    @cached_property
    def _curve_model(self):
        from ._coefficient_curves import coefficient_curve_model

        return coefficient_curve_model(self)

    def curve_model(self):
        """A birational plane model, including original-coefficient embeddings."""
        return self._curve_model

    @property
    def genus(self):
        """Geometric genus of the projective normalization of this coefficient field."""
        return self.curve_model().curve.genus

    def rationality(self, parameter=None, **options):
        """Genus/rationality decision with a conic obstruction when needed."""
        return self.curve_model().curve.rational_parametrization(parameter, **options)

    def rational_parametrization(self, parameter=None, *, point=None):
        """A rational generator in the original ambient field, over the same constants."""
        from ._coefficient_curves import CoefficientFieldParametrization
        from .curves import CurveError

        result = self.rationality(parameter, point=point)
        if not result.rational:
            raise CurveError(
                f"coefficient field is not parametrized over its constants: {result.status}; inspect rationality()"
            )
        return CoefficientFieldParametrization(self.curve_model(), result.parametrization)

    def express(self, value):
        """Return a checked coefficient-field expression for an ambient element, when one exists."""
        value = _normal(self.ambient, value)
        return CoefficientMembership(self, value, _image_membership(self, value))

    def contains(self, value):
        """Test exact membership in the generated coefficient field."""
        return self.express(value).belongs

    def to_ambient(self, expression):
        """Embed an expression from this presentation in its original ambient field."""
        expression = sp.sympify(expression)
        if expression.free_symbols - set(self.generator_symbols):
            raise QuotientAlgebraError("expression must use only the coefficient coordinates")
        return _normal(
            self.ambient,
            expression.xreplace(dict(zip(self.generator_symbols, self.generators, strict=True))),
        )

    def verify(self):
        """Replay the generating relations, degree and ambient coefficient embeddings."""
        if self.relations != _image_relations(
            self.ambient, self.constants, self.generators, self.generator_symbols
        ):
            return False
        if any(self.to_ambient(relation) != 0 for relation in self.relations):
            return False
        expected = _presentation(
            self.constants, self.relations, self.generator_symbols, self.base_index
        )
        if isinstance(expected, QuotientAlgebra):
            return isinstance(self.presentation, QuotientAlgebra) and (
                expected.variables == self.presentation.variables
                and expected.domain == self.presentation.domain
                and expected.groebner_basis == self.presentation.groebner_basis
            )
        return expected is not None and expected == self.presentation


@dataclass(frozen=True)
class RationalSubfieldParametrization:
    """Bidirectional rational maps with generic denominator exclusions."""

    field: RationalFunctionSubfield

    @property
    def parameter(self):
        """The parameter of the rational presentation."""
        return self.field.parameter

    @property
    def generator(self):
        """The generating element in the original ambient field."""
        return self.field.generator

    @property
    def generator_expression(self):
        """The generating rational function in the original formal coordinates."""
        return self.field.generator_expression

    @property
    def coefficient_images(self):
        """Images of the original coefficient generators in the parameter field."""
        return tuple(self.field.express(c).expression for c in self.field.generators)

    @property
    def forward_guard(self):
        """Nonvanishing condition under which the forward rational map is defined."""
        return sp.And(*(sp.Ne(_fraction(c)[1], 0) for c in self.coefficient_images))

    @property
    def inverse_guard(self):
        """Nonvanishing condition under which the inverse rational map is defined."""
        return sp.Ne(_fraction(self.generator_expression)[1], 0)

    def to_ambient(self, value):
        """Embed an expression from this presentation in its original ambient field."""
        return self.field.to_ambient(value)

    def to_source(self, value):
        """Express an ambient element in the coordinates of this presentation."""
        result = self.field.express(value)
        if not result.belongs:
            raise QuotientAlgebraError("element does not belong to the rational subfield")
        return result.expression

    def verify(self):
        """Check both rational coefficient maps and their inverse identities."""
        return self.field.verify()


@dataclass(frozen=True)
class RationalFunctionSubfield:
    """Constructive Lüroth field K(f1(t),...,fr(t)) inside K(t)."""

    ambient: Domain
    constants: Domain
    generators: tuple[sp.Expr, ...]
    generator_symbols: tuple[sp.Symbol, ...]
    parameter: sp.Symbol
    generator: sp.Expr | None
    minimal_ambient_polynomial: sp.Poly | None
    generator_expression: sp.Expr | None

    @property
    def transcendence_degree(self):
        """Transcendence degree of the generated field over its constants."""
        return int(self.generator is not None)

    @property
    def extension_degree(self):
        """[K(t):S], or None for the infinite extension over constants alone."""
        return (
            None
            if self.minimal_ambient_polynomial is None
            else self.minimal_ambient_polynomial.degree()
        )

    @cached_property
    def _ambient_presentation(self):
        _, t = _ambient_data(self.ambient, self.constants)
        x = sp.Dummy("rational_ambient")
        p, q = _fraction(self.generator)
        relation = p.subs(t, x) - self.parameter * q.subs(t, x)
        return QuotientAlgebra.from_polynomials(
            [relation], (x,), domain=self.constants.frac_field(self.parameter)
        )

    def express(self, value):
        """Return a rational expression in the generator together with exact membership evidence."""
        value = _normal(self.ambient, value)
        _, t = _ambient_data(self.ambient, self.constants)
        if self.generator is None:
            expression = value if t not in value.free_symbols else None
        else:
            A = self._ambient_presentation
            expression = field_normal_form(A, value.subs(t, A.variables[0]))
            if A.variables[0] in expression.free_symbols:
                expression = None
        return CoefficientMembership(self, value, expression)

    def contains(self, value):
        """Test exact membership in the generated coefficient field."""
        return self.express(value).belongs

    def to_ambient(self, expression):
        """Embed an expression from this presentation in its original ambient field."""
        expression = sp.sympify(expression)
        if expression.free_symbols - {self.parameter}:
            raise QuotientAlgebraError("expression must use only the rational generator parameter")
        if self.generator is None:
            return self.constants.to_sympy(self.constants.convert(expression))
        return _normal(self.ambient, expression.subs(self.parameter, self.generator))

    @property
    def genus(self):
        """Return the genus of the associated coefficient curve."""
        from .curves import CurveError

        if self.generator is None:
            raise CurveError("genus requires a field of transcendence degree one")
        return 0

    def rational_parametrization(self):
        """Construct a checked rational parametrization with its inverse map."""
        if self.generator is None:
            raise QuotientAlgebraError("a constant field has no transcendental rational generator")
        return RationalSubfieldParametrization(self)

    def verify(self):
        """Check the generator presentation and its ambient embedding."""
        _, t = _ambient_data(self.ambient, self.constants)
        if self.generator is None:
            return (
                self.minimal_ambient_polynomial is None
                and self.generator_expression is None
                and all(t not in g.free_symbols for g in self.generators)
            )
        if self.minimal_ambient_polynomial is None or self.generator_expression is None:
            return False
        if self.generator_expression.free_symbols - set(self.generator_symbols):
            return False
        gcd = _luroth_polynomial(self.ambient, self.generators, self.minimal_ambient_polynomial.gen)
        if gcd != self.minimal_ambient_polynomial:
            return False
        p, q = _fraction(self.generator)
        if max(sp.degree(p, t), sp.degree(q, t)) != gcd.degree():
            return False
        inverse = self.generator_expression.xreplace(
            dict(zip(self.generator_symbols, self.generators, strict=True))
        )
        if _normal(self.ambient, inverse - self.generator) != 0:
            return False
        return all(
            self.express(g).belongs
            and _normal(self.ambient, self.to_ambient(self.express(g).expression) - g) == 0
            for g in self.generators
        )


def _luroth_polynomial(ambient, values, x):
    t = ambient.symbols[0]
    gcd = None
    for value in values:
        p, q = _fraction(value)
        poly = polynomial(p.subs(t, x) - value * q.subs(t, x), (x,), ambient)
        if not poly.is_zero:
            gcd = poly.monic() if gcd is None else gcd.gcd(poly).monic()
    return gcd


@dataclass(frozen=True)
class LurothGeneratorResult:
    """GRS02 generator decision and exact maps for K(f_i(X)) inside K(X).

    ``status`` is ``constant``, ``rational``, or ``higher_transcendence_degree``.
    The last case certifies a lower bound of two, not an exact dimension.
    ``coefficient_images`` use ``parameter``; ``generator_expression`` uses
    ``generator_symbols``. No finite ambient-extension degree is asserted.
    """

    ambient: Domain
    constants: Domain
    generators: tuple[sp.Expr, ...]
    generator_symbols: tuple[sp.Symbol, ...]
    parameter: sp.Symbol
    comparison_symbols: tuple[sp.Symbol, ...]
    gcd_polynomial: sp.Poly | None
    status: str
    generator: sp.Expr | None
    generator_expression: sp.Expr | None
    coefficient_images: tuple[sp.Expr, ...] | None

    @property
    def variables(self):
        """Coordinate symbols in their declared order."""
        return self.ambient.symbols

    @property
    def transcendence_degree(self):
        """The proven transcendence degree, or None when only a lower bound is known."""
        return {"constant": 0, "rational": 1, "higher_transcendence_degree": None}[self.status]

    @property
    def transcendence_degree_lower_bound(self):
        """The certified lower bound on the generated field's transcendence degree."""
        return {"constant": 0, "rational": 1, "higher_transcendence_degree": 2}[self.status]

    @property
    def rational(self):
        """Whether a rational presentation over the declared constants was established."""
        return self.status == "rational"

    def _require_supported_field(self):
        if self.status not in ("constant", "rational"):
            raise QuotientAlgebraError(
                "field has transcendence degree greater than one; no Luroth coordinates exist"
            )

    @cached_property
    def _coordinate_map(self):
        return RationalCoordinateMap(self.ambient, self.generator, self.parameter)

    def express(self, value):
        """Decide membership, with coordinates in K(parameter) when rational."""
        self._require_supported_field()
        (value,) = rational_values(self.ambient, (value,))
        if self.status == "constant":
            expression = value if not value.free_symbols & set(self.variables) else None
        else:
            expression = self._coordinate_map.express(value)
        return CoefficientMembership(self, value, expression)

    def contains(self, value):
        """Test exact membership in the generated coefficient field."""
        return self.express(value).belongs

    def to_ambient(self, expression):
        """Embed an expression from this presentation in its original ambient field."""
        self._require_supported_field()
        source = (
            self.constants
            if self.status == "constant"
            else self.constants.frac_field(self.parameter)
        )
        (expression,) = rational_values(source, (expression,))
        if self.status == "rational":
            expression = expression.xreplace({self.parameter: self.generator})
        return rational_values(self.ambient, (expression,))[0]

    @property
    def genus(self):
        """Return the genus of the associated coefficient curve."""
        if not self.rational:
            raise QuotientAlgebraError("genus requires a field of transcendence degree one")
        return 0

    def rational_parametrization(self):
        """Construct a checked rational parametrization with its inverse map."""
        if not self.rational:
            raise QuotientAlgebraError(
                "a rational parametrization requires transcendence degree one"
            )
        return RationalSubfieldParametrization(self)

    def verify(self):
        """Recompute the gcd decision and replay both rational field embeddings."""
        try:
            if (
                _constant_domain(self.constants) != self.constants
                or self.ambient != self.constants.frac_field(*self.variables)
                or rational_values(self.ambient, self.generators) != self.generators
                or _formal_symbols(self.generators, self.generator_symbols, self.variables)
                != self.generator_symbols
            ):
                return False
            _luroth_symbols(self.variables, self.parameter, self.generator_symbols)
            if (
                len(self.comparison_symbols) != len(self.variables)
                or len(set(self.comparison_symbols)) != len(self.comparison_symbols)
                or any(not isinstance(s, sp.Symbol) for s in self.comparison_symbols)
                or set(self.comparison_symbols)
                & {*self.variables, *self.generator_symbols, self.parameter}
            ):
                return False
            gcd = separated_gcd(self.ambient, self.generators, self.comparison_symbols)
            if gcd is None and self.gcd_polynomial is not None:
                return False
            if gcd is not None and (
                not isinstance(self.gcd_polynomial, sp.Poly)
                or gcd.gens != self.gcd_polynomial.gens
                or gcd.domain != self.gcd_polynomial.domain
                or not (gcd - self.gcd_polynomial).is_zero
            ):
                return False
            if gcd is None:
                return (
                    self.status == "constant"
                    and self.generator is None
                    and self.generator_expression is None
                    and self.coefficient_images == self.generators
                )
            if gcd.total_degree() == 0:
                return (
                    self.status == "higher_transcendence_degree"
                    and self.generator is None
                    and self.generator_expression is None
                    and self.coefficient_images is None
                )
            if (
                self.status != "rational"
                or self.generator is None
                or self.generator_expression is None
                or self.coefficient_images is None
                or len(self.coefficient_images) != len(self.generators)
            ):
                return False
            (generator,) = rational_values(self.ambient, (self.generator,))
            if not generator.free_symbols & set(self.variables):
                return False
            generator_gcd = separated_gcd(self.ambient, (generator,), self.comparison_symbols)
            if not (generator_gcd - gcd).is_zero:
                return False
            formal = self.constants.frac_field(*self.generator_symbols)
            (inverse,) = rational_values(formal, (self.generator_expression,))
            inverse = inverse.xreplace(
                dict(zip(self.generator_symbols, self.generators, strict=True))
            )
            if rational_values(self.ambient, (inverse - generator,))[0] != 0:
                return False
            return all(
                rational_values(self.ambient, (self.to_ambient(image) - original,))[0] == 0
                for image, original in zip(self.coefficient_images, self.generators, strict=True)
            )
        except (
            QuotientAlgebraError,
            CoercionFailed,
            sp.PolynomialError,
            ValueError,
            TypeError,
            AttributeError,
            ZeroDivisionError,
        ):
            return False


def _luroth_symbols(variables, parameter, symbols):
    if (
        not variables
        or len(set(variables)) != len(variables)
        or any(not isinstance(v, sp.Symbol) or not v.is_commutative for v in variables)
    ):
        raise QuotientAlgebraError("variables must be nonempty distinct commutative Symbols")
    if any(not symbol.is_commutative for symbol in symbols):
        raise QuotientAlgebraError("formal coefficients must be commutative Symbols")
    if (
        not isinstance(parameter, sp.Symbol)
        or not parameter.is_commutative
        or parameter in {*variables, *symbols}
    ):
        raise QuotientAlgebraError("rational parameter must be a fresh commutative Symbol")


def luroth_generator(
    functions, *, variables, constants=sp.QQ, parameter=None, generator_symbols=None
):
    """Decide and construct K(f_1,...,f_m)=K(h) in K(variables), by GRS02.

    This implements Algorithm 3.1 for rational/algebraic-number constants.
    A unit separated gcd certifies transcendence degree greater than one;
    constant-only inputs are handled separately. Discovery uses a multivariate
    gcd, membership a generic graph, and the inverse map finite field closure.
    These exact computations have no wall-clock or degree budget.
    """
    constants = _constant_domain(constants)
    variables = tuple(variables)
    parameter = sp.Dummy("rational_generator") if parameter is None else parameter
    _luroth_symbols(variables, parameter, ())
    ambient = constants.frac_field(*variables)
    values = rational_values(ambient, functions)
    symbols = _formal_symbols(values, generator_symbols, (*variables, parameter))
    _luroth_symbols(variables, parameter, symbols)
    comparison = tuple(sp.Dummy(f"luroth_comparison_{i}") for i in range(len(variables)))
    gcd = separated_gcd(ambient, values, comparison)
    generator = inverse = images = None
    if gcd is None:
        status, images = "constant", values
    elif gcd.total_degree() == 0:
        status = "higher_transcendence_degree"
    else:
        status = "rational"
        generator = smallest_coefficient(gcd, ambient)
        coordinates = RationalCoordinateMap(ambient, generator, parameter)
        images = tuple(coordinates.express(value) for value in values)
        if any(image is None for image in images):
            raise QuotientAlgebraError("Luroth generator did not recover all input coordinates")
        inverse = inverse_coordinates(images, parameter, constants, symbols)
    return LurothGeneratorResult(
        ambient,
        constants,
        values,
        symbols,
        parameter,
        comparison,
        gcd,
        status,
        generator,
        inverse,
        images,
    )


def coefficient_field(
    coefficients, *, ambient, constants=sp.QQ, generator_symbols=None, parameter=None
):
    """Construct K(coefficients), retaining explicit embeddings into ``ambient``.

    Rational ambient fields use constructive Lüroth gcds. Multivariate rational
    ambient fields return ``LurothGeneratorResult`` for transcendence degree at
    most one; use ``luroth_generator`` to inspect higher-degree decisions. Algebraic ambient
    fields use saturated graph elimination and a finite presentation over a
    transcendental coefficient. All constants and coefficients must be exact.
    Algebraic coefficient fields expose plane models, normalized genus and
    genus-zero rationality through the separate curve layer.
    """
    constants = _constant_domain(constants)
    if (
        isinstance(ambient, Domain)
        and getattr(ambient, "is_FractionField", False)
        and len(ambient.symbols) > 1
    ):
        if _constant_domain(ambient.domain) != constants:
            raise QuotientAlgebraError(
                "constants must match the ambient rational-function constant field"
            )
        result = luroth_generator(
            coefficients,
            variables=ambient.symbols,
            constants=constants,
            parameter=parameter,
            generator_symbols=generator_symbols,
        )
        if result.status == "higher_transcendence_degree":
            raise QuotientAlgebraError(
                "coefficient field has transcendence degree greater than one; inspect luroth_generator"
            )
        return result
    base, z = _ambient_data(ambient, constants)
    values = tuple(_normal(ambient, c) for c in coefficients)
    forbidden = (*ambient.variables, z) if isinstance(ambient, QuotientAlgebra) else (z,)
    symbols = _formal_symbols(values, generator_symbols, forbidden)
    if isinstance(ambient, QuotientAlgebra):
        parameter_index = None
        for i, value in enumerate(values):
            polynomial = ambient.minimal_polynomial(value, sp.Dummy("constant_test"))
            if any(z in coefficient.free_symbols for coefficient in polynomial.all_coeffs()):
                parameter_index = i
                break
        relations = _image_relations(ambient, constants, values, symbols)
        presentation = _presentation(constants, relations, symbols, parameter_index)
        if presentation is None:
            raise QuotientAlgebraError(
                "coefficient image did not yield a finite rebased presentation"
            )
        return AlgebraicCoefficientField(
            ambient, constants, values, symbols, parameter_index, relations, presentation
        )
    parameter = sp.Dummy("rational_generator") if parameter is None else parameter
    if not isinstance(parameter, sp.Symbol) or parameter in {*symbols, z}:
        raise QuotientAlgebraError("rational parameter must be a fresh Symbol")
    x = sp.Dummy("luroth_x")
    polynomial = _luroth_polynomial(base, values, x)
    if polynomial is None:
        return RationalFunctionSubfield(
            base, constants, values, symbols, parameter, None, None, None
        )
    generator = next((-c for c in polynomial.all_coeffs()[1:] if z in c.free_symbols), None)
    if generator is None:
        raise QuotientAlgebraError("Luroth polynomial has no nonconstant coefficient")
    # Recover the generator as a rational expression in the *original*
    # coefficients, not merely as an expression in the ambient parameter.
    first = next(i for i, c in enumerate(values) if z in c.free_symbols)
    relations = _image_relations(base, constants, values, symbols)
    presentation = _presentation(constants, relations, symbols, first)
    if presentation is None:
        raise QuotientAlgebraError("rational coefficient image did not yield a finite presentation")
    image = AlgebraicCoefficientField(
        base, constants, values, symbols, first, relations, presentation
    )
    inverse = _image_membership(image, generator)
    if inverse is None:
        raise QuotientAlgebraError(
            "could not certify the Luroth generator in the coefficient field"
        )
    return RationalFunctionSubfield(
        base, constants, values, symbols, parameter, sp.cancel(generator), polynomial, inverse
    )


__all__ = [
    "coefficient_field",
    "luroth_generator",
    "LurothGeneratorResult",
    "CoefficientMembership",
    "AlgebraicCoefficientField",
    "RationalFunctionSubfield",
    "RationalSubfieldParametrization",
]

# Public result types live here alongside their coefficient-field factories.

__all__ += ["CoefficientCurveModel", "CoefficientFieldParametrization"]
