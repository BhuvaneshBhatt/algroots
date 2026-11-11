"""Certified algebraic function fields with explicit primitive-element maps."""

from dataclasses import dataclass
from functools import cached_property

import sympy as sp
from sympy.polys.agca.extensions import ExtensionElement, MonogenicFiniteExtension
from sympy.polys.domains.field import Field
from sympy.polys.polyerrors import CoercionFailed

from ._validation import validate_integer
from .groebner_errors import GroebnerLimitError, GroebnerVerificationError


def _norm_coordinates(expressions, variables):
    """Separate elimination coordinates from symbols bound by algebraic roots.

    Keep ordinary coordinates unchanged to reuse symbolic arithmetic caches.
    Only a coordinate also bound inside a RootOf needs a private replacement.
    """
    bound = set().union(
        *(
            root.args[0].free_symbols
            for expression in expressions
            for root in expression.atoms(sp.CRootOf)
        )
    )
    return tuple(
        sp.Dummy(variable.name) if variable in bound else variable for variable in variables
    )


@dataclass(frozen=True)
class TowerCompression:
    """Exact maps for adjoining one root and compressing a separable tower.

    ``old_image`` and ``new_image`` express both generators in ``primitive``.
    ``primitive_expression`` gives the inverse map in the original generators.
    The norm is irreducible over the rational-function base.
    """

    old_modulus: sp.Poly
    new_modulus: sp.Poly
    primitive_modulus: sp.Poly
    old_image: sp.Expr
    new_image: sp.Expr
    primitive_expression: sp.Expr

    def __reduce__(self):
        from ._coefficient_state import polynomial_state

        return _restore_tower_compression, (
            polynomial_state(self.old_modulus),
            polynomial_state(self.new_modulus),
            polynomial_state(self.primitive_modulus),
            self.old_image,
            self.new_image,
            self.primitive_expression,
        )

    def verify(self):
        """Replay relations and the inverse primitive map over the declared base."""
        try:
            q = self.primitive_modulus
            z = q.gen
            if (
                not q.is_irreducible
                or q.degree() != self.old_modulus.degree() * self.new_modulus.degree()
            ):
                return False
            a, b = self.old_modulus.gen, self.new_modulus.gen

            def rem(expression):
                from .domains import polynomial

                return polynomial(expression, (z,), q.domain).rem(q)

            substitutions = {a: self.old_image, b: self.new_image}
            return (
                rem(self.old_modulus.as_expr().subs(a, self.old_image)).is_zero
                and rem(self.new_modulus.as_expr().subs(substitutions, simultaneous=True)).is_zero
                and rem(
                    self.primitive_expression.subs(substitutions, simultaneous=True) - z
                ).is_zero
            )
        except (ValueError, TypeError, ArithmeticError, sp.PolynomialError, CoercionFailed):
            return False


def _restore_tower_compression(old, new, primitive, old_image, new_image, inverse):
    from ._coefficient_state import restore_polynomial

    return TowerCompression(
        restore_polynomial(old),
        restore_polynomial(new),
        restore_polynomial(primitive),
        old_image,
        new_image,
        inverse,
    )


class AlgebraicFunctionField(MonogenicFiniteExtension, Field):
    """A certified finite field extension of QQ(parameters) or K(parameters).

    Construct from an irreducible univariate ``Poly`` over QQ or a rational
    function field over QQ/a number field. ``extend`` accepts a modulus over
    this field, checks irreducibility through a squarefree norm, and compresses
    the result. Generator names remain valid coefficient expressions. A symbol
    denotes a formal algebraic root; real or complex embeddings are chosen
    explicitly when specializing.
    """

    is_AlgebraicFunctionField = True

    def __init__(self, modulus):
        if not isinstance(modulus, sp.Poly) or not modulus.is_univariate or modulus.degree() < 1:
            raise ValueError("modulus must be a positive-degree univariate Poly")
        base = modulus.domain
        ground = base.domain if base.is_FractionField else base
        if not base.is_Field or (ground != sp.QQ and not ground.is_AlgebraicField):
            raise ValueError("the base must be QQ, a number field or its rational function field")
        parameters = tuple(base.symbols) if base.is_FractionField else ()
        if not isinstance(modulus.gen, sp.Symbol) or any(
            modulus.gen.name == p.name for p in parameters
        ):
            raise ValueError("algebraic generator and parameters must have distinct names")
        from .domains import polynomial_domain

        modulus = modulus.set_domain(polynomial_domain(base)).monic()
        if not modulus.is_irreducible:
            raise ValueError("the algebraic modulus is reducible over the base field")
        super().__init__(modulus)
        self.parameters = parameters
        self.tower_symbols = (self.symbol,)
        self.images = ((self.symbol, self.symbol),)
        self.relations = (modulus,)
        self.compressions = ()

    def __eq__(self, other):
        return type(self) is type(other) and self.modulus == other.modulus

    __hash__ = MonogenicFiniteExtension.__hash__

    def __reduce__(self):
        from ._coefficient_state import domain_state, restore_domain

        return restore_domain, (domain_state(self),)

    def of_type(self, value):
        """Whether a value uses this field's own element representation."""
        return isinstance(value, ExtensionElement) and value.ext == self

    def convert(self, value, base=None):
        """Embed a scalar or an element of a compatible coefficient field."""
        if isinstance(value, ExtensionElement):
            if value.ext == self:
                return value
            if isinstance(value.ext, AlgebraicFunctionField):
                moduli = (self.modulus, *(proof.old_modulus for proof in self.compressions))
                if any(
                    p.gen == value.ext.modulus.gen and p == value.ext.modulus.set_domain(p.domain)
                    for p in moduli
                ):
                    return self.from_sympy(value.ext.to_sympy(value))
            raise CoercionFailed("an explicit embedding is required for this extension")
        if isinstance(value, sp.Basic):
            return self.from_sympy(value)
        return super().convert(value, base)

    def from_sympy(self, value):
        """Interpret a rational expression using exact operations in the field generators."""
        value = (
            sp.sympify(value).subs(dict(self.images), simultaneous=True)
            if hasattr(self, "images")
            else sp.sympify(value)
        )
        # Parse rational expressions in the generator by field operations.
        # The underlying polynomial-ring parser cannot invert its generator.
        from .domains import polynomial

        numerator, denominator = sp.together(value).as_numer_denom()
        # Share the coefficient parser with basis construction: a CRootOf can
        # bind the same printed symbol as this field's formal generator.
        first = super().convert(polynomial(numerator, (self.symbol,), self.domain).rep)
        second = super().convert(polynomial(denominator, (self.symbol,), self.domain).rep)
        if not second:
            raise ZeroDivisionError("coefficient denominator is zero in the algebraic field")
        return first / second

    def exquo(self, left, right):
        """Divide by a nonzero element in the declared coefficient field."""
        return left / right

    def extend(self, modulus, *, max_degree=64, max_attempts=32):
        """Adjoin an irreducible root, returning a field with checked inverse maps.

        A squarefree Trager norm has the full product degree. Its exact
        factorization decides irreducibility over the existing field. Budgets
        bound the degree and the search for a primitive element.
        """
        validate_integer(max_degree, "max_degree", minimum=1)
        validate_integer(max_attempts, "max_attempts", minimum=1)
        if (
            not isinstance(modulus, sp.Poly)
            or not modulus.is_univariate
            or modulus.domain != self
            or modulus.degree() < 1
        ):
            raise ValueError("extension modulus must be a positive-degree Poly over this field")
        if modulus.gen.name in {p.name for p in self.parameters + self.tower_symbols}:
            raise ValueError("tower generator names must be distinct")
        if self.rank * modulus.degree() > max_degree:
            raise GroebnerLimitError("algebraic tower degree budget exceeded")
        from .domains import polynomial

        a, b = self.symbol, modulus.gen
        modulus = modulus.monic()
        z = sp.Dummy("theta")
        # The resultant binds its elimination coordinate. Use a private symbol
        # so a RootOf constant binding the printed generator stays a constant.
        (root,) = _norm_coordinates((self.modulus.as_expr(), modulus.as_expr()), (a,))
        old_expression = self.modulus.as_expr().subs(a, root)
        expression = modulus.as_expr().subs(a, root)
        for shift in range(1, max_attempts + 1):
            second = expression.subs(b, z - shift * root)
            norm = polynomial(sp.resultant(old_expression, second, root), (z,), self.domain).monic()
            if norm.degree() != self.rank * modulus.degree() or not norm.is_sqf:
                continue
            if not norm.is_irreducible:
                raise ValueError("the extension modulus is reducible over the existing field")
            result = AlgebraicFunctionField(norm)
            first_poly = polynomial(old_expression, (root,), result)
            second_poly = polynomial(second, (root,), result)
            common = first_poly.gcd(second_poly).monic()
            if common.degree() != 1:
                raise GroebnerVerificationError(
                    "primitive-element recovery did not produce a linear gcd"
                )
            old_image = result.to_sympy(-common.rep.to_list()[-1])
            new_image = result.to_sympy(result.convert(z - shift * old_image))
            proof = TowerCompression(
                self.modulus, modulus, norm, old_image, new_image, b + shift * a
            )
            if not proof.verify():
                raise GroebnerVerificationError("tower compression failed exact replay")
            result.tower_symbols = self.tower_symbols + (b,)
            result.images = tuple(
                (symbol, result.to_sympy(result.convert(image.subs(a, old_image))))
                for symbol, image in self.images
            ) + ((b, new_image),)
            # Include old primitive symbols so elements already converted in an
            # earlier compressed tower retain their exact coefficient embedding.
            result.images += ((a, old_image),) if a not in self.tower_symbols else ()
            result.relations = self.relations + (modulus,)
            result.compressions = self.compressions + (proof,)
            return result
        raise GroebnerLimitError("primitive-element search exhausted max_attempts")

    def with_parameters(self, *parameters):
        """Adjoin independent parameters while preserving every tower embedding."""
        if not parameters:
            return self
        symbols = self.parameters + self.tower_symbols + (self.symbol,)
        if (
            any(not isinstance(p, sp.Symbol) for p in parameters)
            or len(set(parameters)) != len(parameters)
            or any(p.name == q.name for p in parameters for q in symbols)
        ):
            raise ValueError(
                "new parameters must be distinct Symbols outside the coefficient tower"
            )
        cache = self.__dict__.setdefault("_parameter_extensions", {})
        if parameters in cache:
            return cache[parameters]
        from .domains import polynomial

        combined = self.parameters + parameters
        ground = self.domain.domain if self.domain.is_FractionField else self.domain
        base = ground.frac_field(*combined)
        converted = {}

        def rebase(field):
            if field in converted:
                return converted[field]
            result = AlgebraicFunctionField(
                polynomial(field.modulus.as_expr(), field.modulus.gens, base)
            )
            converted[field] = result
            result.tower_symbols, result.images = field.tower_symbols, field.images

            def poly(p):
                target = rebase(p.domain) if isinstance(p.domain, AlgebraicFunctionField) else base
                return polynomial(p.as_expr(), p.gens, target)

            result.relations = tuple(poly(p) for p in field.relations)
            result.compressions = tuple(
                TowerCompression(
                    poly(p.old_modulus),
                    poly(p.new_modulus),
                    poly(p.primitive_modulus),
                    p.old_image,
                    p.new_image,
                    p.primitive_expression,
                )
                for p in field.compressions
            )
            return result

        result = rebase(self)
        cache[parameters] = result
        return result

    def rational_model(self):
        """Return a QQ(parameter) power basis and exact maps for native images.

        Number-field constants are merged with the algebraic function generator
        through the same certified primitive-element construction as a tower.
        """
        cached = self.__dict__.get("_rational_model")
        if cached is not None:
            return cached
        ground = self.domain.domain if self.domain.is_FractionField else self.domain
        if ground == sp.QQ:
            return self, self.symbol, lambda value: value
        c = sp.Dummy("constant")
        rational = sp.QQ.frac_field(*self.parameters) if self.parameters else sp.QQ
        constants = AlgebraicFunctionField(
            sp.Poly.from_list(ground.mod.to_list(), c, domain=rational)
        )

        def constant(value):
            return sum(sp.Rational(v) * c**i for i, v in enumerate(reversed(value.to_list())))

        def base_expression(value):
            if not self.parameters:
                return constant(value)

            def expression(poly):
                return sum(
                    constant(v) * sp.prod(p**e for p, e in zip(self.parameters, m, strict=True))
                    for m, v in poly.items()
                )

            return expression(value.numer) / expression(value.denom)

        def expression(value):
            return sum(
                base_expression(v) * self.symbol**i
                for i, v in enumerate(reversed(value.rep.to_list()))
            )

        modulus = sum(
            base_expression(v) * self.symbol**i
            for i, v in enumerate(reversed(self.modulus.rep.to_list()))
        )
        flat = constants.extend(
            sp.Poly(modulus, self.symbol, domain=constants),
            max_degree=max(64, self.rank * constants.rank),
        )
        primitive_image = flat.compressions[-1].primitive_expression.subs(c, ground.ext.as_expr())

        def convert(value):
            return flat.convert(expression(value))

        self._rational_model = flat, primitive_image, convert
        return self._rational_model

    def factor_univariate(self, polynomial, *, max_attempts=32, max_degree=256):
        """Factor exactly using squarefree Trager norms and recover factors by gcd.

        Each irreducible norm factor yields one irreducible factor in this
        field when the norm is squarefree. Reconstruction is checked in the
        original field. The return value follows ``Poly.factor_list``.
        """
        validate_integer(max_attempts, "max_attempts", minimum=1)
        validate_integer(max_degree, "max_degree", minimum=1)
        if (
            not isinstance(polynomial, sp.Poly)
            or not polynomial.is_univariate
            or polynomial.domain != self
        ):
            raise ValueError("a univariate Poly over this field is required")
        if polynomial.is_zero:
            return sp.S.Zero, []
        if polynomial.degree() * self.rank > max_degree:
            raise GroebnerLimitError("factorization norm degree budget exceeded")
        unit = polynomial.LC()
        result = []
        from .domains import polynomial as exact_polynomial

        variable = polynomial.gen
        x, a = _norm_coordinates(
            (self.modulus.as_expr(), polynomial.as_expr()), (variable, self.symbol)
        )
        old_expression = self.modulus.as_expr().subs(self.symbol, a)
        for squarefree, multiplicity in polynomial.monic().sqf_list()[1]:
            expression = squarefree.as_expr().subs({variable: x, self.symbol: a}, simultaneous=True)
            for shift in range(max_attempts):
                norm = exact_polynomial(
                    sp.resultant(old_expression, expression.subs(x, x - shift * a), a),
                    (x,),
                    self.domain,
                ).monic()
                if norm.degree() != squarefree.degree() * self.rank or not norm.is_sqf:
                    continue
                for norm_factor, _ in norm.factor_list()[1]:
                    lifted = exact_polynomial(
                        norm_factor.as_expr().subs(x, variable + shift * self.symbol),
                        (variable,),
                        self,
                    )
                    factor = squarefree.gcd(lifted).monic()
                    if factor.degree() > 0:
                        result.append((factor, multiplicity))
                break
            else:
                raise GroebnerLimitError("factorization primitive search exhausted max_attempts")
        reconstructed = exact_polynomial(unit, (variable,), self)
        for factor, multiplicity in result:
            reconstructed *= factor**multiplicity
        if reconstructed != polynomial:
            raise GroebnerVerificationError("function-field factors failed exact reconstruction")
        return unit, result

    @cached_property
    def _tower_presentation(self):
        from .field_specialization import TowerPresentation

        return TowerPresentation(self)

    def original_expression(self, value):
        """Return exact coordinates in the original, uncompressed tower basis."""
        return self._tower_presentation.expression(value)

    def specializations(self, values, *, real=False, max_branches=64, limits=None):
        """Enumerate all exact compatible branches, optionally keeping real roots only."""
        from .field_specialization import enumerate_specializations

        return enumerate_specializations(
            self, values, real=real, max_branches=max_branches, limits=limits
        )

    def specialize(self, values, *, roots, limits=None):
        """Choose exact roots and a local coefficient specialization of the tower.

        All parameter values and generator roots are required. Relations and
        coefficient poles are checked in the resulting number field; branch
        selection is never inferred from a floating-point approximation.
        """
        if set(values) != set(self.parameters) or set(roots) != set(self.tower_symbols):
            raise ValueError("supply every parameter and every tower generator root")
        assignments = {key: sp.sympify(value) for key, value in {**values, **roots}.items()}
        if any(
            value.has(sp.Float) or value.is_algebraic is not True for value in assignments.values()
        ):
            raise ValueError("specialization values and roots must be exact algebraic constants")
        constants = tuple(value for value in assignments.values() if not value.is_Rational)
        ground = self.domain.domain if self.domain.is_FractionField else self.domain
        if ground.is_AlgebraicField:
            constants += tuple(ground.orig_ext)
        target = sp.QQ.algebraic_field(*constants) if constants else sp.QQ
        from .native_f4 import NativeF4Limits

        limits = NativeF4Limits() if limits is None else limits
        if not isinstance(limits, NativeF4Limits):
            raise TypeError("limits must be NativeF4Limits")
        historical = {}
        for proof in self.compressions:
            historical[proof.primitive_modulus.gen] = proof.primitive_expression.subs(
                historical, simultaneous=True
            ).subs(assignments, simultaneous=True)
        from .field_specialization import specialized_polynomial

        for index, generator in enumerate(self.tower_symbols):
            relation = specialized_polynomial(
                self,
                index,
                {p: assignments[p] for p in self.parameters},
                {p: assignments[p] for p in self.tower_symbols[:index]},
                limits,
            )
            if target.convert(relation.as_expr().subs(generator, assignments[generator])):
                raise ValueError("chosen roots do not satisfy a defined specialized tower relation")
        primitive = self.symbol
        image = assignments[primitive] if primitive in roots else historical[primitive]
        return FieldSpecialization(
            self,
            tuple((p, assignments[p]) for p in self.parameters),
            tuple((p, assignments[p]) for p in self.tower_symbols),
            target,
            image,
            limits,
        )


@dataclass(frozen=True)
class FieldSpecialization:
    """A checked local specialization at a parameter point with exact root choices."""

    source: AlgebraicFunctionField
    values: tuple
    roots: tuple
    domain: object
    primitive_image: object
    limits: object = None

    def verify(self):
        """Recheck root choices and the primitive image without trusting this record."""
        try:
            expected = self.source.specialize(
                dict(self.values), roots=dict(self.roots), limits=self.limits
            )
            return (
                expected.domain == self.domain and expected.primitive_image == self.primitive_image
            )
        except (ValueError, TypeError, KeyError, ArithmeticError, CoercionFailed):
            return False

    def __call__(self, value):
        """Specialize a field element, rejecting a pole in its reduced coordinates."""
        if not self.verify():
            raise GroebnerVerificationError("field specialization failed verification")
        return self._evaluate(value)

    def _evaluate(self, value):
        from .field_specialization import evaluate_local
        from .native_f4 import NativeF4Limits

        return evaluate_local(
            self, value, self.limits if self.limits is not None else NativeF4Limits()
        )
