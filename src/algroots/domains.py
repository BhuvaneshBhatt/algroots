"""Native coefficient conversion for polynomial and rational-function domains."""

from functools import lru_cache

import sympy as sp
from sympy.polys.domains import FractionField
from sympy.polys.fields import FracElement, FracField
from sympy.polys.groebnertools import groebner as native_groebner
from sympy.polys.matrices import DomainMatrix
from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.polyoptions import build_options
from sympy.polys.rings import PolyElement, PolyRing

from .finite_fields import is_finite_field


class _MonicFraction(FracElement):
    """Normalize coefficient-field units at each rational-function construction.

    Polynomial gcd cancellation need not make denominators monic over number
    fields. Repeated scalar divisions can then produce enormous units in both
    numerator and denominator. Scaling the leading coefficient bounds that
    growth and makes the fraction canonical over the ground field.
    """

    def __init__(self, field, numer, denom=None):
        if denom is None:
            denom = field.ring.one
        if not denom:
            raise ZeroDivisionError("zero denominator")
        if not numer:
            denom = field.ring.one
        elif field.domain.one != denom.LC:
            inverse = field.domain.exquo(field.domain.one, denom.LC)
            numer, denom = numer.mul_ground(inverse), denom.mul_ground(inverse)
        super().__init__(field, numer, denom)


class _MonicFractionField(FracField):
    """A local arithmetic representation whose elements have monic denominators."""

    def __new__(cls, symbols, domain, order):
        field = super().__new__(cls, symbols, domain, order)
        previous = field.gens
        field.dtype = _MonicFraction(field, field.ring.zero).raw_new
        field.zero, field.one = field.dtype(field.ring.zero), field.dtype(field.ring.one)
        field.gens = field._gens()
        for symbol, old, new in zip(field.symbols, previous, field.gens, strict=True):
            if isinstance(symbol, sp.Symbol) and field.__dict__.get(symbol.name) is old:
                setattr(field, symbol.name, new)
        return field

    def __eq__(self, other):
        # Keep computational representations distinct so domain conversion
        # rebuilds raw fractions with monic denominators, even inside a Poly.
        return type(self) is type(other) and super().__eq__(other)

    __hash__ = FracField.__hash__


@lru_cache(maxsize=128)
def polynomial_domain(domain):
    """Select a coefficient representation for native polynomial algorithms.

    Rational functions over number or finite fields use monic denominators after every
    arithmetic operation. Other domains are returned unchanged. Callers retain
    their declared field and convert final native coefficients back with
    domain.convert(value, computational_domain).
    """
    if domain.is_FractionField and (
        domain.domain.is_AlgebraicField or is_finite_field(domain.domain)
    ):
        if isinstance(domain.field, _MonicFractionField):
            return domain
        return FractionField(_MonicFractionField(domain.symbols, domain.domain, domain.field.order))
    return domain


def _algebraic_element(value, domain):
    """Evaluate rational operations in K before invoking an algebraic embedding.

    Converting a large rational expression as one algebraic number can trigger
    numerical field-isomorphism heuristics even when its primitive constants
    are already known. Native arithmetic avoids those heuristics and preserves
    small nonzero units. Products such as I*sqrt(3) still need a whole-expression
    embedding when their separate factors lie outside K=QQ(sqrt(-3)).
    """
    try:
        if value.is_Add:
            return sum((_algebraic_element(term, domain) for term in value.args), domain.zero)
        if value.is_Mul:
            result = domain.one
            for factor in value.args:
                result *= _algebraic_element(factor, domain)
            return result
        if value.is_Pow and value.exp.is_Integer:
            return _algebraic_element(value.base, domain) ** int(value.exp)
    except CoercionFailed:
        return domain.convert(value)
    return domain.convert(value)


def field_element(value, domain):
    """Embed an expression without splitting algebraic constants outside their field.

    FractionField.from_sympy recursively parses expression factors. A constant
    such as sqrt(-3)=sqrt(3)*I may belong to the ground field even though its
    displayed factors do not. Convert whole polynomial coefficients over the
    ground field before constructing the rational function natively.
    """
    if domain.of_type(value):
        return value
    value = sp.sympify(value)
    if value.has(sp.Float):
        raise ValueError("exact coefficients are required")
    if is_finite_field(domain):
        if value.is_Rational:
            return domain.convert(int(value.p)) / domain.convert(int(value.q))
        if value.is_Add:
            return sum((field_element(v, domain) for v in value.args), domain.zero)
        if value.is_Mul:
            result = domain.one
            for factor in value.args:
                result *= field_element(factor, domain)
            return result
        if value.is_Pow and value.exp.is_Integer:
            return field_element(value.base, domain) ** int(value.exp)
        return domain.convert(value)
    if domain.is_AlgebraicField:
        return _algebraic_element(value, domain)
    if not domain.is_FractionField:
        return domain.convert(value)
    if not domain.domain.is_AlgebraicField and not is_finite_field(domain.domain):
        return domain.convert(value)
    numerator, denominator = sp.together(value).as_numer_denom()
    parts = [polynomial(p, domain.symbols, domain.domain) for p in (numerator, denominator)]
    return domain.field.new(*(domain.field.ring.from_dict(p.rep.to_dict()) for p in parts))


def coefficient_domain(value):
    """Normalize a SymPy domain or an adapter field with an exact as_domain method."""
    from sympy.polys.domains import Domain

    if not isinstance(value, Domain) and hasattr(value, "as_domain"):
        value = value.as_domain()
    result = sp.polys.polyoptions.Domain.preprocess(value)
    if getattr(result, "is_FractionField", False) and getattr(
        result.domain, "is_AlgebraicFunctionField", False
    ):
        result = result.domain.with_parameters(*result.symbols)
    if not isinstance(result, Domain):
        raise TypeError("as_domain must return an exact SymPy coefficient domain")
    return result


def infer_coefficient_field(expressions, variables):
    """Infer a field while shielding bound root symbols and clearing rational parts."""
    expressions = tuple(map(sp.sympify, expressions))
    variables = tuple(variables)
    constants = set().union(*(p.atoms(sp.CRootOf, sp.AlgebraicNumber) for p in expressions))
    if any(constant.has(*variables) for constant in constants):
        coordinates = tuple(sp.Dummy(v.name) for v in variables)
        substitutions = dict(zip(variables, coordinates, strict=True))
        expressions = tuple(p.subs(substitutions, simultaneous=True) for p in expressions)
        variables = coordinates
    if not expressions:
        return sp.QQ
    parts = tuple(
        part for expression in expressions for part in sp.together(expression).as_numer_denom()
    )
    _, options = sp.parallel_poly_from_expr(parts, *variables, extension=True)
    return options.domain.get_field()


def polynomial(expression, variables, domain):
    """Construct a polynomial over the declared field using native coefficient elements."""
    variables = tuple(variables)
    if (
        isinstance(expression, sp.Poly)
        and expression.gens == variables
        and expression.domain == domain
    ):
        return expression
    expression = (
        expression.as_expr()
        if isinstance(expression, (sp.Poly, PolyElement))
        else sp.sympify(expression)
    )
    if expression.has(sp.Float):
        raise ValueError("exact polynomial coefficients are required")
    ground = domain.domain if domain.is_FractionField else domain
    if ground.is_AlgebraicField or getattr(domain, "is_AlgebraicFunctionField", False):
        # CRootOf's defining polynomial binds its own symbols. Hide that
        # bound syntax while identifying the free coordinate monomials.
        roots = tuple(
            root
            for root in expression.atoms(sp.CRootOf)
            if root.args[0].free_symbols.intersection(variables)
        )
        if roots:
            replacements = {root: sp.Dummy("algebraic_constant") for root in roots}
            inverse = {dummy: root for root, dummy in replacements.items()}
            parsed = sp.Poly(expression.xreplace(replacements), *variables, domain=sp.EX)
            return sp.Poly.from_dict(
                {
                    monomial: field_element(coefficient.xreplace(inverse), domain)
                    for monomial, coefficient in parsed.terms()
                },
                *variables,
                domain=domain,
            )
    if (
        not ground.is_AlgebraicField
        and not is_finite_field(ground)
        and not getattr(domain, "is_AlgebraicFunctionField", False)
    ):
        return sp.Poly(expression, *variables, domain=domain)
    parsed = sp.Poly(expression, *variables, domain=sp.EX)
    return sp.Poly.from_dict(
        {m: field_element(c, domain) for m, c in parsed.terms()}, *variables, domain=domain
    )


def groebner(expressions, variables, domain, order="lex", *, backend="native", limits=None):
    """Compute a SymPy GroebnerBasis while preserving native coefficient embeddings."""
    if backend != "sympy" or limits is not None:
        from .basis import groebner_basis

        return groebner_basis(
            expressions, variables, domain=domain, order=str(order), backend=backend, limits=limits
        )
    variables = tuple(variables)
    options = build_options(variables, {"domain": domain, "order": order, "polys": False})
    ring = PolyRing(variables, polynomial_domain(domain), order)
    inputs = [
        ring.from_dict(polynomial(p, variables, ring.domain).rep.to_dict()) for p in expressions
    ]
    basis = native_groebner([p for p in inputs if p], ring)
    polys = [
        sp.Poly.from_dict(
            {m: domain.convert(c, ring.domain) for m, c in p.items()}, *variables, domain=domain
        )
        for p in basis
    ]
    return sp.GroebnerBasis._new(polys, options)


def domain_matrix(matrix, domain):
    """Convert an expression matrix to exact native entries, including empty shapes."""
    entries = {}
    for (row, column), value in matrix.todok().items():
        entries.setdefault(row, {})[column] = field_element(value, domain)
    return DomainMatrix(entries, matrix.shape, domain)


__all__ = ["field_element", "polynomial", "polynomial_domain", "groebner", "domain_matrix"]
