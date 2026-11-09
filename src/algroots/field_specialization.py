"""Exact local evaluation and branch enumeration for algebraic function fields."""

from functools import cached_property
from itertools import product

import sympy as sp
from sympy.polys.matrices import DomainMatrix
from sympy.polys.rings import PolyRing

from .domains import polynomial
from .groebner_errors import GroebnerLimitError, GroebnerVerificationError
from .groebner_proofs import _tracked_buchberger


class TowerPresentation:
    """Original triangular power basis, independent of a compressed primitive model."""

    def __init__(self, field):
        self.field = field
        degrees = tuple(p.degree() for p in field.relations)
        self.exponents = tuple(product(*(range(d) for d in degrees)))
        self.monomials = tuple(
            sp.prod(g**e for g, e in zip(field.tower_symbols, exponent, strict=True))
            for exponent in self.exponents
        )

    @cached_property
    def inverse_coordinates(self):
        field = self.field
        entries = {}
        for j, monomial in enumerate(self.monomials):
            coefficients = list(reversed(field.convert(monomial).rep.to_list()))
            for i, coefficient in enumerate(coefficients):
                if coefficient:
                    entries.setdefault(i, {})[j] = coefficient
        matrix = DomainMatrix(entries, (field.rank, field.rank), field.domain)
        return matrix.inv()

    def expression(self, value):
        field = self.field
        coefficients = list(reversed(field.convert(value).rep.to_list()))
        vector = DomainMatrix(
            {i: {0: c} for i, c in enumerate(coefficients) if c}, (field.rank, 1), field.domain
        )
        coordinates = (self.inverse_coordinates * vector).to_dod()
        return sum(
            field.domain.to_sympy(row[0]) * self.monomials[i] for i, row in coordinates.items()
        )

    @cached_property
    def relations(self):
        result = []
        for relation in self.field.relations:
            base = relation.domain
            coefficients = relation.rep.to_list()
            expressions = (
                base._tower_presentation.expression(c)
                if getattr(base, "is_AlgebraicFunctionField", False)
                else base.to_sympy(c)
                for c in coefficients
            )
            result.append(
                sum(c * relation.gen**i for i, c in enumerate(reversed(tuple(expressions))))
            )
        return tuple(result)


def regular_expression(embedding, value, limits):
    """Evaluate a regular element in the original model's local ring.

    For N/D, regularity at the selected point means some h in the colon
    ideal (I,D):N is nonzero there. A tracked intersection computation gives
    h*N = g*D modulo I, so g/h is an independently checked regular expression.
    A missing witness is a pole/nonregular element in this model, not a
    guessed numerical limit.
    """
    field, target = embedding.source, embedding.domain
    expression = field._tower_presentation.expression(value)
    assignments = {**dict(embedding.values), **dict(embedding.roots)}
    numerator, denominator = sp.together(expression).as_numer_denom()

    def at(expr):
        result = expr.subs(assignments, simultaneous=True)
        if result.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
            raise ValueError("field element has a pole at this specialization")
        return target.convert(result)

    den = at(denominator)
    if den:
        return numerator / denominator
    ground = field.domain.domain if field.domain.is_FractionField else field.domain
    coordinates = field.parameters + field.tower_symbols
    auxiliary = sp.Dummy("intersection")
    ring = PolyRing((auxiliary, *coordinates), ground, "lex")

    def native(expr):
        return ring.from_dict(polynomial(expr, ring.symbols, ground).rep.to_dict())

    relations = [
        sp.together(p).as_numer_denom()[0] for p in local_model_relations(embedding, limits)
    ]
    original = [native(p) for p in relations] + [native(denominator)]
    num = native(numerator)
    u = ring.gens[0]
    generators = [u * p for p in original] + [(1 - u) * num]
    basis, representations = _tracked_buchberger(
        tuple(generators), ring, max_basis=limits.max_basis, max_pairs=limits.max_pairs
    )
    if (
        sum(map(len, basis)) + sum(len(p) for row in representations for p in row)
        > limits.max_proof_nodes
    ):
        raise GroebnerLimitError("local specialization witness-node budget exceeded")
    for candidate, row in zip(basis, representations, strict=True):
        if any(m[0] for m in candidate):
            continue
        quotient, remainder = candidate.div([num])
        if remainder:
            raise GroebnerVerificationError("local intersection failed exact divisibility")
        h = quotient[0].as_expr()
        hvalue = at(h)
        if not hvalue:
            continue
        if sum((a * b for a, b in zip(row, generators, strict=True)), ring.zero) != candidate:
            raise GroebnerVerificationError("local specialization identity failed replay")
        g = row[-2].as_expr().subs(auxiliary, 1)
        regular = g / h
        if field.convert(regular) != field.convert(value):
            raise GroebnerVerificationError("local specialization changed the field element")
        return regular
    raise ValueError("field element has a pole or is not regular in the selected local model")


def evaluate_local(embedding, value, limits):
    """Evaluate an exactly checked expression regular at the selected point."""
    expression = regular_expression(embedding, value, limits)
    result = expression.subs({**dict(embedding.values), **dict(embedding.roots)}, simultaneous=True)
    return embedding.domain.to_sympy(embedding.domain.convert(result))


def _relation_coefficients(field, index, values, roots, limits, *, local=False):
    relation = field.relations[index]
    base = relation.domain
    if getattr(base, "is_AlgebraicFunctionField", False):
        prefix = base.specialize(
            {p: values[p] for p in base.parameters},
            roots={p: roots[p] for p in base.tower_symbols},
            limits=limits,
        )
        return tuple(
            regular_expression(prefix, c, limits) if local else prefix._evaluate(c)
            for c in relation.rep.to_list()
        )
    coefficients = []
    for c in relation.rep.to_list():
        expression = base.to_sympy(c)
        evaluated = expression.subs(values, simultaneous=True)
        if evaluated.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
            raise ValueError("tower coefficient has a pole at this specialization")
        coefficients.append(expression if local else evaluated)
    return tuple(coefficients)


def specialized_polynomial(field, index, values, roots, limits):
    """Specialize a defining relation through regular prefix-field coefficients."""
    generator = field.tower_symbols[index]
    coefficients = _relation_coefficients(field, index, values, roots, limits)
    expression = sum(c * generator**i for i, c in enumerate(reversed(coefficients)))
    ground = field.domain.domain if field.domain.is_FractionField else field.domain
    extensions = tuple(ground.orig_ext) if ground.is_AlgebraicField else ()
    extensions += tuple(v for v in (*values.values(), *roots.values()) if not v.is_Rational)
    target = sp.QQ.algebraic_field(*extensions) if extensions else sp.QQ
    return polynomial(expression, (generator,), target)


def local_model_relations(embedding, limits):
    """Use unit-denominator prefix coefficients to define the correct local model."""
    field = embedding.source
    values, roots = dict(embedding.values), dict(embedding.roots)
    result = []
    for index, generator in enumerate(field.tower_symbols):
        coefficients = _relation_coefficients(field, index, values, roots, limits, local=True)
        result.append(sum(c * generator**i for i, c in enumerate(reversed(coefficients))))
    return tuple(result)


def enumerate_specializations(field, values, *, real=False, max_branches=64, limits=None):
    """Enumerate exact compatible root choices without choosing an implicit branch."""
    from ._validation import validate_integer

    validate_integer(max_branches, "max_branches", minimum=1)
    if not isinstance(real, bool):
        raise TypeError("real must be Boolean")
    if set(values) != set(field.parameters):
        raise ValueError("supply every parameter value")
    values = {p: sp.sympify(v) for p, v in values.items()}
    if any(v.has(sp.Float) or v.is_algebraic is not True for v in values.values()):
        raise ValueError("parameter values must be exact algebraic constants")
    branches = [{}]
    from .native_f4 import NativeF4Limits

    limits = NativeF4Limits() if limits is None else limits
    if not isinstance(limits, NativeF4Limits):
        raise TypeError("limits must be NativeF4Limits")
    for index, generator in enumerate(field.tower_symbols):
        following = []
        defined = 0
        for roots in branches:
            try:
                poly = specialized_polynomial(field, index, values, roots, limits)
            except ValueError:
                continue
            if poly.is_zero or poly.degree() < 1:
                raise ValueError("specialized relation does not define a finite algebraic tower")
            defined += 1
            for root, _ in poly.all_roots(multiple=False, radicals=False):
                if real and root.is_real is not True:
                    continue
                following.append({**roots, generator: root})
                if len(following) > max_branches:
                    raise GroebnerLimitError("algebraic specialization branch budget exceeded")
        if branches and not defined:
            raise ValueError(
                "specialization lies outside the tower coefficient domain on every branch"
            )
        branches = following
    return tuple(field.specialize(values, roots=roots, limits=limits) for roots in branches)
