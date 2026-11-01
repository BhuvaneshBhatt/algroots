"""Certified rational decompositions and supplied algebraic-factor reconstruction.

Rational search enumerates two-factor decompositions over the declared constants,
up to a fractional linear change of intermediate coordinate. Algebraic discovery
constructs coefficient extensions using universal monic-factor algebras; callers
may alternatively supply a factor in an explicit extension.
All identities describe rational functions or Zariski-closed correspondences;
affine real branches and domains belong in a downstream semialgebraic layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import islice

import sympy as sp
from sympy.polys.polyerrors import BasePolynomialError

from ._coefficient_curves import CoefficientFieldParametrization
from .function_fields import _constant_domain, coefficient_field
from .quotient import QuotientAlgebra
from .subfields import field_normal_form


class FunctionalDecompositionError(ValueError):
    """Invalid decomposition input or a failed exact reconstruction."""


class DecompositionLimitError(FunctionalDecompositionError):
    """The input exceeds a pre-computation degree limit."""


def _symbols(variables):
    if any(not isinstance(v, sp.Symbol) for v in variables) or len(set(variables)) != len(
        variables
    ):
        raise FunctionalDecompositionError("distinct Symbol variables are required")


def _integer(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise FunctionalDecompositionError(f"{name} must be an integer >= {minimum}")


def _rational(expression, variable, constants):
    expression = sp.sympify(expression)
    if expression.has(sp.Float) or expression.free_symbols - {variable}:
        raise FunctionalDecompositionError(
            "an exact rational function over the constants is required"
        )
    try:
        field = constants.frac_field(variable)
        return field.to_sympy(field.convert(expression))
    except (ValueError, TypeError, BasePolynomialError) as exc:
        raise FunctionalDecompositionError(
            "an exact rational function over the constants is required"
        ) from exc


def _parts(expression, variable, constants):
    p, q = sp.fraction(sp.cancel(expression))
    return sp.Poly(p, variable, domain=constants), sp.Poly(q, variable, domain=constants)


def _degree(expression, variable, constants):
    p, q = _parts(expression, variable, constants)
    return int(max(p.degree(), q.degree()))


def _relation(expression, variables, constants):
    expression = sp.sympify(expression)
    if expression.has(sp.Float) or expression.free_symbols - set(variables):
        raise FunctionalDecompositionError("an exact relation over the constants is required")
    # Clearing a rational expression is only used internally. Public algebraic
    # inputs must already be polynomials; their excluded loci cannot be inferred.
    numerator = sp.fraction(sp.cancel(expression))[0]
    return sp.Poly(numerator, *variables, domain=constants).monic()


def _input_relation(expression, variables, constants):
    expression = sp.sympify(expression)
    if expression.has(sp.Float) or expression.free_symbols - set(variables):
        raise FunctionalDecompositionError("an exact polynomial relation is required")
    try:
        polynomial = sp.Poly(expression, *variables, domain=constants)
    except (ValueError, TypeError, BasePolynomialError) as exc:
        raise FunctionalDecompositionError("an exact polynomial relation is required") from exc
    if any(polynomial.degree(v) <= 0 for v in variables) or not polynomial.is_irreducible:
        raise FunctionalDecompositionError(
            "the relation must be irreducible and depend on both variables"
        )
    return polynomial.monic()


@dataclass(frozen=True)
class RationalDecomposition:
    """An exact two-factor identity, with the original affine input exclusions."""

    original_expression: sp.Expr
    variable: sp.Symbol
    parameter: sp.Symbol
    constants: object
    outer: sp.Expr
    inner: sp.Expr

    @property
    def function(self):
        return _rational(self.original_expression, self.variable, self.constants)

    @property
    def degrees(self):
        return (
            _degree(self.outer, self.parameter, self.constants),
            _degree(self.inner, self.variable, self.constants),
        )

    @property
    def input_guard(self):
        """Visible poles before cancellation, including poles of nested terms."""
        return sp.And(*(sp.Ne(p, 0) for p in _visible_denominators(self.original_expression)))

    def verify(self):
        """Replay constant-domain, degree and exact composition checks."""
        try:
            _symbols((self.variable, self.parameter))
            if _constant_domain(self.constants) != self.constants:
                return False
            outer = _rational(self.outer, self.parameter, self.constants)
            inner = _rational(self.inner, self.variable, self.constants)
            a, b = self.degrees
            return (
                a > 1
                and b > 1
                and a * b == _degree(self.function, self.variable, self.constants)
                and sp.cancel(outer.subs(self.parameter, inner) - self.function) == 0
            )
        except (ValueError, TypeError, AttributeError, BasePolynomialError, ZeroDivisionError):
            return False


def _visible_denominators(expression):
    expression = sp.sympify(expression)
    denominators = [sp.together(expression).as_numer_denom()[1]]
    for term in sp.preorder_traversal(expression):
        if term.is_Pow and term.exp.is_Integer and term.exp < 0:
            numerator, denominator = sp.together(term.base).as_numer_denom()
            denominators.extend((numerator, denominator))
    return tuple(dict.fromkeys(d for d in denominators if d != 1))


def _cross_polynomial(function, variable, comparison, constants):
    p, q = _parts(function, variable, constants)
    return sp.Poly(
        p.as_expr().subs(variable, comparison) * q.as_expr()
        - q.as_expr().subs(variable, comparison) * p.as_expr(),
        comparison,
        variable,
        domain=constants,
    ).monic()


def _factor_indices(factors, variable, comparison):
    diagonal = next(
        i
        for i, f in enumerate(factors)
        if f.degree(comparison) == 1 and f.as_expr().subs(comparison, variable) == 0
    )
    return diagonal, tuple(i for i in range(len(factors)) if i != diagonal)


def _product_data(factors, degree, variable, comparison):
    diagonal, others = _factor_indices(factors, variable, comparison)
    targets = tuple(d - 1 for d in range(2, degree) if degree % d == 0)
    # Count eligible products without materializing an exponential powerset.
    ways = {(len(others), 0): 1}
    for i in range(len(others) - 1, -1, -1):
        weight = factors[others[i]].degree(comparison)
        for d in range(degree):
            ways[i, d] = ways.get((i + 1, d), 0) + ways.get((i + 1, d - weight), 0)
    return diagonal, others, targets, ways


def _product_count(factors, degree, variable, comparison):
    _, _, targets, ways = _product_data(factors, degree, variable, comparison)
    return sum(ways.get((0, d), 0) for d in targets)


def _products(factors, degree, variable, comparison):
    """Every possible right-factor polynomial contains the diagonal factor."""
    diagonal, others, targets, ways = _product_data(factors, degree, variable, comparison)

    def subsets(i, remaining, chosen):
        if not ways.get((i, remaining), 0):
            return
        if i == len(others):
            yield (diagonal, *chosen)
            return
        yield from subsets(i + 1, remaining, chosen)
        yield from subsets(
            i + 1, remaining - factors[others[i]].degree(comparison), (*chosen, others[i])
        )

    for target in targets:
        yield from subsets(0, target, ())


def _recover_rational(original, variable, parameter, constants, comparison, factors, indices):
    function = _rational(original, variable, constants)
    product = sp.prod(factors[i].as_expr() for i in indices)
    polynomial = sp.Poly(product, comparison, domain=constants.frac_field(variable)).monic()
    d = polynomial.degree()
    inner = next((c for c in polynomial.all_coeffs()[1:] if variable in c.free_symbols), None)
    if inner is None or _degree(inner, variable, constants) != d:
        return None
    # A nonconstant coefficient of a monic near-separated polynomial is a
    # fractional linear transform of its right factor. Verify that it gives
    # exactly this polynomial before recovering the uniquely determined outer.
    near = _cross_polynomial(inner, variable, comparison, constants)
    if (
        sp.Poly(near.as_expr(), comparison, domain=constants.frac_field(variable)).monic()
        != polynomial
    ):
        return None
    p, q = _parts(inner, variable, constants)
    source = sp.Dummy("decomposition_source")
    while source.name == parameter.name:
        source = sp.Dummy()
    A = QuotientAlgebra.from_polynomials(
        [(p.as_expr() - parameter * q.as_expr()).xreplace({variable: source})],
        (source,),
        domain=constants.frac_field(parameter),
    )
    outer = field_normal_form(A, function.xreplace({variable: source}))
    if source in outer.free_symbols:
        return None
    result = RationalDecomposition(
        original, variable, parameter, constants, outer, sp.cancel(inner)
    )
    return result if result.verify() else None


@dataclass(frozen=True)
class RationalDecompositionResult:
    """All visited two-factor answers, with replayable exhaustive-search status.

    Completeness is over ``constants`` and modulo fractional linear changes of
    the intermediate coordinate. It does not assert completeness over a larger
    constant field or enumerate recursive decomposition chains.
    """

    original_expression: sp.Expr
    variable: sp.Symbol
    parameter: sp.Symbol
    constants: object
    comparison_variable: sp.Symbol
    factors: tuple[sp.Poly, ...]
    examined: int
    candidate_count: int
    decompositions: tuple[RationalDecomposition, ...]

    @property
    def complete(self):
        return self.examined == self.candidate_count

    @property
    def status(self):
        if self.decompositions:
            return "decomposable"
        if not self.complete:
            return "incomplete"
        if (
            _degree(
                _rational(self.original_expression, self.variable, self.constants),
                self.variable,
                self.constants,
            )
            < 2
        ):
            return "degree_below_two"
        return "indecomposable"

    def verify(self):
        """Replay factorization, the searched prefix, answers and completeness."""
        try:
            _symbols((self.variable, self.parameter, self.comparison_variable))
            if _constant_domain(self.constants) != self.constants:
                return False
            _integer(self.examined, "examined", 0)
            _integer(self.candidate_count, "candidate_count", 0)
            f = _rational(self.original_expression, self.variable, self.constants)
            n = _degree(f, self.variable, self.constants)
            if n < 2:
                return (
                    self.factors == ()
                    and self.examined == self.candidate_count == 0
                    and not self.decompositions
                )
            cross = _cross_polynomial(f, self.variable, self.comparison_variable, self.constants)
            if not self.factors or any(
                p.gens != cross.gens
                or p.domain != self.constants
                or p.LC() != 1
                or not p.is_irreducible
                for p in self.factors
            ):
                return False
            if (
                sp.Poly(
                    sp.prod(p.as_expr() for p in self.factors), *cross.gens, domain=self.constants
                ).monic()
                != cross
            ):
                return False
            count = _product_count(self.factors, n, self.variable, self.comparison_variable)
            if count != self.candidate_count or self.examined > count:
                return False
            if self.complete:
                # A separate checker proves coverage from factor products and
                # near-separated identities, without invoking candidate recovery.
                from ._decomposition_verification import verify_complete_rational_search

                return verify_complete_rational_search(self)
            expected = tuple(
                answer
                for indices in islice(
                    _products(self.factors, n, self.variable, self.comparison_variable),
                    self.examined,
                )
                if (
                    answer := _recover_rational(
                        self.original_expression,
                        self.variable,
                        self.parameter,
                        self.constants,
                        self.comparison_variable,
                        self.factors,
                        indices,
                    )
                )
                is not None
            )
            return self.decompositions == expected and all(d.verify() for d in self.decompositions)
        except (
            ValueError,
            TypeError,
            AttributeError,
            BasePolynomialError,
            ZeroDivisionError,
            StopIteration,
        ):
            return False


def rational_decompose(
    expression, variable, *, parameter=None, constants=sp.QQ, max_candidates=256, max_degree=32
):
    """Find rational two-factor decompositions, exhaustively up to the budget.

    ``max_candidates`` bounds eligible factor products, not factorization time.
    A stopped prefix has ``complete=False`` and is never called indecomposable.
    ``max_degree`` refuses large inputs before polynomial factorization.
    A tuple/list of input variables dispatches to multivariate rational search;
    a Symbol retains the existing univariate result type and contract.
    """
    if isinstance(variable, (tuple, list)):
        return multivariate_rational_decompose(
            expression,
            variable,
            parameter=parameter,
            constants=constants,
            max_candidates=max_candidates,
            max_degree=max_degree,
        )
    parameter = sp.Dummy("intermediate") if parameter is None else parameter
    comparison = sp.Dummy("comparison")
    _symbols((variable, parameter, comparison))
    constants = _constant_domain(constants)
    _integer(max_candidates, "max_candidates", 0)
    _integer(max_degree, "max_degree", 1)
    original = sp.sympify(expression)
    f = _rational(original, variable, constants)
    n = _degree(f, variable, constants)
    if n > max_degree:
        raise DecompositionLimitError(f"rational degree {n} exceeds max_degree={max_degree}")
    factors, count, answers = (), 0, []
    if n >= 2:
        cross = _cross_polynomial(f, variable, comparison, constants)
        _, factorization = cross.factor_list()
        if any(m != 1 for _, m in factorization):
            raise ArithmeticError(
                "the characteristic-zero difference polynomial must be squarefree"
            )
        factors = tuple(p.monic() for p, _ in factorization)
        count = _product_count(factors, n, variable, comparison)
        for indices in islice(_products(factors, n, variable, comparison), max_candidates):
            answer = _recover_rational(
                original, variable, parameter, constants, comparison, factors, indices
            )
            if answer is not None:
                answers.append(answer)
    return RationalDecompositionResult(
        original,
        variable,
        parameter,
        constants,
        comparison,
        factors,
        min(max_candidates, count),
        count,
        tuple(answers),
    )


@dataclass(frozen=True)
class AlgebraicDecomposition:
    """An irreducible resultant decomposition of an implicit relation."""

    relation: sp.Poly
    outer: sp.Poly
    inner: sp.Poly
    exponent: int
    unit: sp.Expr

    @property
    def output(self):
        return self.relation.gens[0]

    @property
    def input(self):
        return self.relation.gens[1]

    @property
    def intermediate(self):
        return self.outer.gens[1]

    @property
    def constants(self):
        return self.relation.domain

    def verify(self):
        """Check all polynomials, nontriviality, degrees, power and scalar unit."""
        try:
            x, z, y = self.output, self.input, self.intermediate
            _symbols((x, y, z))
            if _constant_domain(self.constants) != self.constants:
                return False
            _integer(self.exponent, "exponent", 1)
            if self.outer.gens != (x, y) or self.inner.gens != (y, z):
                return False
            if any(
                p.domain != self.constants
                or not p.is_irreducible
                or any(p.degree(v) <= 0 for v in p.gens)
                for p in (self.relation, self.outer, self.inner)
            ):
                return False
            if any(max(p.degree(v) for v in p.gens) <= 1 for p in (self.outer, self.inner)):
                return False
            if sp.sympify(self.unit).has(sp.Float):
                return False
            unit = self.constants.convert(self.unit)
            if (
                not unit
                or self.exponent * self.relation.degree(x)
                != self.outer.degree(x) * self.inner.degree(y)
                or self.exponent * self.relation.degree(z)
                != self.outer.degree(y) * self.inner.degree(z)
            ):
                return False
            resultant = sp.Poly(
                sp.resultant(self.outer.as_expr(), self.inner.as_expr(), y),
                x,
                z,
                domain=self.constants,
            )
            return resultant == self.relation**self.exponent * self.constants.to_sympy(unit)
        except (
            ValueError,
            TypeError,
            AttributeError,
            IndexError,
            BasePolynomialError,
            ZeroDivisionError,
        ):
            return False


def _candidate(expression, variable, ambient):
    try:
        polynomial = sp.Poly(expression, variable)
        coefficients = [field_normal_form(ambient, c) for c in polynomial.all_coeffs()]
        while coefficients and coefficients[0] == 0:
            coefficients.pop(0)
        if len(coefficients) <= 1:
            raise FunctionalDecompositionError("candidate must have positive degree")
        inverse = ambient.inverse(coefficients[0])
        coefficients = tuple(field_normal_form(ambient, c * inverse) for c in coefficients)
    except (ValueError, TypeError, BasePolynomialError, ZeroDivisionError) as exc:
        raise FunctionalDecompositionError(
            "candidate must be a polynomial over the ambient field"
        ) from exc
    if len(coefficients) <= 1:
        raise FunctionalDecompositionError("candidate must have positive degree")
    return sp.Add(
        *(c * variable ** (len(coefficients) - i - 1) for i, c in enumerate(coefficients))
    )


def _factor_remainder(relation, candidate, variable, ambient):
    left = list(sp.Poly(relation, variable).all_coeffs())
    right = sp.Poly(candidate, variable).all_coeffs()
    while len(left) >= len(right):
        lead = field_normal_form(ambient, left[0])
        for i, c in enumerate(right):
            left[i] = field_normal_form(ambient, left[i] - lead * c)
        left.pop(0)
    return tuple(field_normal_form(ambient, c) for c in left)


@dataclass(frozen=True)
class AlgebraicDecompositionResult:
    """Reconstruction from one supplied factor; never a complete discovery claim."""

    relation: sp.Poly
    ambient: QuotientAlgebra
    candidate: sp.Expr
    parameter: sp.Symbol
    coefficient_field: object
    rationality: object
    parametrization: object | None
    decomposition: AlgebraicDecomposition | None
    reason: str | None

    @property
    def complete(self):
        return False

    @property
    def status(self):
        return "decomposable" if self.decomposition is not None else "candidate_rejected"

    @property
    def decompositions(self):
        return () if self.decomposition is None else (self.decomposition,)

    def verify(self):
        """Replay the supplied-factor embedding, rational maps and final identity."""
        try:
            x, z = self.relation.gens
            if (
                _input_relation(self.relation.as_expr(), (x, z), self.relation.domain)
                != self.relation
            ):
                return False
            _validate_ambient(self.ambient, self.relation.domain, z, x, self.parameter)
            if _candidate(self.candidate, x, self.ambient) != self.candidate or any(
                _factor_remainder(self.relation.as_expr(), self.candidate, x, self.ambient)
            ):
                return False
            E = self.coefficient_field
            coefficients = tuple(
                field_normal_form(self.ambient, c)
                for c in sp.Poly(self.candidate, x).all_coeffs()[1:]
            )
            if (
                E.ambient is not self.ambient
                or E.constants != self.relation.domain
                or E.generators != coefficients
                or not E.verify()
            ):
                return False
            model = E.curve_model()
            if self.rationality.normalization.curve != model.curve or not self.rationality.verify():
                return False
            if not self.rationality.rational:
                return (
                    self.parametrization is None
                    and self.decomposition is None
                    and self.reason == self.rationality.status
                )
            P = self.parametrization
            if (
                P.model != model
                or P.parametrization != self.rationality.parametrization
                or P.parameter != self.parameter
                or not P.verify()
            ):
                return False
            G, H = _components(self.relation, P, self.parameter, self.ambient)
            rejection = _rejection(G, H)
            if rejection is not None:
                return self.decomposition is None and self.reason == rejection
            D = self.decomposition
            return (
                D is not None
                and D.relation == self.relation
                and D.outer == G
                and D.inner == H
                and self.reason is None
                and D.verify()
            )
        except (ValueError, TypeError, AttributeError, BasePolynomialError, ZeroDivisionError):
            return False


def _validate_ambient(ambient, constants, z, x, y):
    if (
        not isinstance(ambient, QuotientAlgebra)
        or ambient.domain != constants.frac_field(z)
        or not ambient.is_field
    ):
        raise FunctionalDecompositionError(
            "ambient must be a certified quotient field over constants(input)"
        )
    _symbols((x, y, z, *ambient.variables))


def _components(relation, parametrization, parameter, ambient):
    x, z = relation.gens
    images = parametrization.coefficient_images
    expression = x ** len(images) + sum(
        c * x ** (len(images) - i - 1) for i, c in enumerate(images)
    )
    G = _relation(expression, (x, parameter), relation.domain)
    H = _relation(
        ambient.minimal_polynomial(parametrization.generator, parameter).as_expr(),
        (parameter, z),
        relation.domain,
    )
    return G, H


def _rejection(G, H):
    if any(not p.is_irreducible for p in (G, H)):
        return "reducible_candidate"
    if any(
        any(p.degree(v) <= 0 for v in p.gens) or max(p.degree(v) for v in p.gens) <= 1
        for p in (G, H)
    ):
        return "trivial_component"
    return None


def algebraic_decompose(
    relation,
    variables,
    *,
    candidate=None,
    ambient=None,
    parameter=None,
    constants=sp.QQ,
    point=None,
    max_steps=256,
    max_dimension=256,
    factor_degrees=None,
    max_degree=6,
    max_factor_dimension=32,
    max_separating_attempts=64,
    max_candidates=64,
    exhaustive=False,
):
    """Discover or reconstruct minimal coefficient-field decompositions.

    ``variables=(output,input)`` and ``candidate`` is a polynomial in ``output``
    whose coefficients lie in ``ambient``, an explicitly supplied finite field
    over ``constants(input)``. The output variable must be fresh in that field.
    Its coefficient field is formed over constants, without adjoining input.
    With neither candidate nor ambient, construct all requested factor extensions
    automatically and return an AlgebraicDecompositionSearch. The default stops
    at its first verified answer; exhaustive=True visits all candidates within
    limits. A supplied candidate returns AlgebraicDecompositionResult as before.
    """
    x, z = tuple(variables)
    parameter = sp.Dummy("intermediate") if parameter is None else parameter
    constants = _constant_domain(constants)
    if candidate is None and ambient is None:
        from ._algebraic_discovery import search_algebraic_decomposition

        if point is not None:
            raise FunctionalDecompositionError("point witnesses require a supplied candidate")
        return search_algebraic_decomposition(
            relation,
            (x, z),
            parameter=parameter,
            constants=constants,
            factor_degrees=factor_degrees,
            max_degree=max_degree,
            max_factor_dimension=max_factor_dimension,
            max_separating_attempts=max_separating_attempts,
            max_candidates=max_candidates,
            exhaustive=exhaustive,
            max_steps=max_steps,
            max_dimension=max_dimension,
        )
    if candidate is None or ambient is None:
        raise FunctionalDecompositionError("supply both candidate and ambient, or neither")
    if factor_degrees is not None:
        raise FunctionalDecompositionError("factor_degrees applies only to automatic discovery")
    if not isinstance(ambient, QuotientAlgebra):
        raise FunctionalDecompositionError("ambient must be a QuotientAlgebra field")
    _validate_ambient(ambient, constants, z, x, parameter)
    F = _input_relation(relation, (x, z), constants)
    g = _candidate(candidate, x, ambient)
    if any(_factor_remainder(F.as_expr(), g, x, ambient)):
        raise FunctionalDecompositionError(
            "candidate does not divide the input in the ambient field"
        )
    coefficients = sp.Poly(g, x).all_coeffs()[1:]
    E = coefficient_field(coefficients, ambient=ambient, constants=constants)
    if E.transcendence_degree != 1:
        raise FunctionalDecompositionError("candidate coefficients must generate a function field")
    R = E.rationality(parameter, point=point, max_steps=max_steps, max_dimension=max_dimension)
    if not R.rational:
        return AlgebraicDecompositionResult(F, ambient, g, parameter, E, R, None, None, R.status)
    P = CoefficientFieldParametrization(E.curve_model(), R.parametrization)
    G, H = _components(F, P, parameter, ambient)
    reason = _rejection(G, H)
    D = None
    if reason is None:
        resultant = sp.Poly(
            sp.resultant(G.as_expr(), H.as_expr(), parameter), x, z, domain=constants
        )
        exponent, remainder = divmod(G.degree(x) * H.degree(parameter), F.degree(x))
        if remainder or exponent < 1:
            raise FunctionalDecompositionError(
                "component degrees do not give an integral resultant power"
            )
        unit = constants.to_sympy(constants.convert(resultant.LC() / (F**exponent).LC()))
        D = AlgebraicDecomposition(F, G, H, exponent, unit)
        if not D.verify():
            raise FunctionalDecompositionError(
                "candidate produced extra resultant components or failed exact reconstruction"
            )
    return AlgebraicDecompositionResult(F, ambient, g, parameter, E, R, P, D, reason)


__all__ = [
    "rational_decompose",
    "algebraic_decompose",
    "RationalDecomposition",
    "RationalDecompositionResult",
    "AlgebraicDecomposition",
    "AlgebraicDecompositionResult",
    "FunctionalDecompositionError",
    "DecompositionLimitError",
]


from ._algebraic_discovery import (  # noqa: E402
    AlgebraicCandidateAttempt,
    AlgebraicDecompositionSearch,
    AlgebraicDiscoveryIssue,
    AlgebraicFactorDegree,
    AlgebraicFactorDiscovery,
    DiscoveredAlgebraicFactor,
    discover_algebraic_factors,
)

__all__ += [
    "discover_algebraic_factors",
    "DiscoveredAlgebraicFactor",
    "AlgebraicFactorDegree",
    "AlgebraicFactorDiscovery",
    "AlgebraicDiscoveryIssue",
    "AlgebraicCandidateAttempt",
    "AlgebraicDecompositionSearch",
]

from ._multivariate_decomposition import (  # noqa: E402
    MultivariateRationalDecomposition,
    MultivariateRationalDecompositionResult,
    multivariate_rational_decompose,
)

__all__ += [
    "multivariate_rational_decompose",
    "MultivariateRationalDecomposition",
    "MultivariateRationalDecompositionResult",
]

from ._decomposition_chains import (  # noqa: E402
    AlgebraicDecompositionChain,
    DecompositionChainNode,
    RationalChainComponent,
    RationalDecompositionChain,
    algebraic_decomposition_chain,
    rational_decomposition_chain,
)

__all__ += [
    "rational_decomposition_chain",
    "algebraic_decomposition_chain",
    "RationalDecompositionChain",
    "AlgebraicDecompositionChain",
    "DecompositionChainNode",
    "RationalChainComponent",
]
