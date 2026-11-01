"""GRS02 Algorithm 2.1: uni-multivariate rational functional decomposition.

Products are counted by their total degrees in the two independent variable
blocks and whether they vanish on the diagonal. Near-separated recognition
uses a coefficient of the monic candidate, followed by an exact polynomial
identity. No specialization or random point selection is needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import islice

import sympy as sp
from sympy.polys.polyerrors import BasePolynomialError

from ._luroth import RationalCoordinateMap, rational_values, smallest_coefficient
from .function_fields import _constant_domain


def _symbols(variables, parameter, comparison=()):
    # Keep facade imports local so either module can be imported first.
    from .functional_decomposition import FunctionalDecompositionError

    symbols = (*variables, parameter, *comparison)
    if (
        not variables
        or any(not isinstance(v, sp.Symbol) or not v.is_commutative for v in symbols)
        or len(set(symbols)) != len(symbols)
        or (comparison and len(comparison) != len(variables))
    ):
        raise FunctionalDecompositionError("distinct commutative Symbol variables are required")


def _normal(expression, variables, constants):
    from .functional_decomposition import FunctionalDecompositionError

    try:
        return rational_values(constants.frac_field(*variables), (expression,))[0]
    except (ValueError, TypeError, AttributeError, BasePolynomialError, ZeroDivisionError) as exc:
        raise FunctionalDecompositionError(
            "an exact rational function over the declared constants and variables is required"
        ) from exc


def _degree(expression, variables, constants):
    expression = _normal(expression, variables, constants)
    numerator, denominator = sp.fraction(expression)
    return int(
        max(
            sp.Poly(numerator, *variables, domain=constants).total_degree(),
            sp.Poly(denominator, *variables, domain=constants).total_degree(),
        )
    )


def _targets(degree):
    return tuple(d for d in range(2, degree) if degree % d == 0)


def _cross(function, variables, comparison, constants):
    numerator, denominator = sp.fraction(_normal(function, variables, constants))
    renaming = dict(zip(variables, comparison, strict=True))
    return sp.Poly(
        numerator.xreplace(renaming) * denominator - denominator.xreplace(renaming) * numerator,
        *comparison,
        *variables,
        domain=constants,
    ).monic()


def _same_polynomial(left, right):
    return (
        isinstance(left, sp.Poly)
        and isinstance(right, sp.Poly)
        and left.gens == right.gens
        and left.domain == right.domain
        and (left - right).is_zero
    )


def _product_data(factors, degree, variables, comparison, constants):
    """Suffix counts, retaining both block degrees and diagonal vanishing."""
    targets = _targets(degree)
    cap = max(targets, default=0)
    split = len(variables)
    renaming = dict(zip(comparison, variables, strict=True))
    weights = tuple(
        (
            max(sum(m[:split]) for m in factor.monoms()),
            max(sum(m[split:]) for m in factor.monoms()),
            sp.Poly(factor.as_expr().xreplace(renaming), *variables, domain=constants).is_zero,
        )
        for factor in factors
    )
    ways = [None] * (len(factors) + 1)
    ways[-1] = {(0, 0, False): 1}
    for i in range(len(factors) - 1, -1, -1):
        wx, wy, diagonal = weights[i]
        suffix = ways[i + 1]
        current = dict(suffix)
        for (dx, dy, hit), count in suffix.items():
            if dx + wx <= cap and dy + wy <= cap:
                key = (dx + wx, dy + wy, hit or diagonal)
                current[key] = current.get(key, 0) + count
        ways[i] = current
    count = sum(ways[0].get((d, d, True), 0) for d in targets)
    return targets, weights, ways, count


def _products(data):
    targets, weights, ways, _ = data

    def subsets(i, dx, dy, needs_diagonal, chosen):
        possible = ways[i].get((dx, dy, True), 0)
        if not needs_diagonal:
            possible += ways[i].get((dx, dy, False), 0)
        if not possible:
            return
        if i == len(weights):
            yield chosen
            return
        yield from subsets(i + 1, dx, dy, needs_diagonal, chosen)
        wx, wy, diagonal = weights[i]
        yield from subsets(i + 1, dx - wx, dy - wy, needs_diagonal and not diagonal, (*chosen, i))

    for target in targets:
        yield from subsets(0, target, target, True, ())


@dataclass(frozen=True)
class MultivariateRationalDecomposition:
    """An exact f(X)=outer(inner(X)), with a univariate outer function."""

    original_expression: sp.Expr
    variables: tuple[sp.Symbol, ...]
    parameter: sp.Symbol
    constants: object
    outer: sp.Expr
    inner: sp.Expr

    @property
    def function(self):
        return _normal(self.original_expression, self.variables, self.constants)

    @property
    def degrees(self):
        """Univariate outer degree and rational total degree of the inner."""
        return (
            _degree(self.outer, (self.parameter,), self.constants),
            _degree(self.inner, self.variables, self.constants),
        )

    @property
    def input_guard(self):
        from .functional_decomposition import _visible_denominators

        return sp.And(*(sp.Ne(p, 0) for p in _visible_denominators(self.original_expression)))

    def verify(self):
        try:
            _symbols(self.variables, self.parameter)
            if _constant_domain(self.constants) != self.constants:
                return False
            outer = _normal(self.outer, (self.parameter,), self.constants)
            inner = _normal(self.inner, self.variables, self.constants)
            a, b = self.degrees
            return (
                a > 1
                and b > 1
                and a * b == _degree(self.function, self.variables, self.constants)
                and _normal(
                    outer.xreplace({self.parameter: inner}) - self.function,
                    self.variables,
                    self.constants,
                )
                == 0
            )
        except (ValueError, TypeError, AttributeError, BasePolynomialError, ZeroDivisionError):
            return False


def _recover(original, function, variables, parameter, constants, comparison, factors, indices):
    product = sp.Poly(
        sp.prod(factors[i].as_expr() for i in indices), *comparison, *variables, domain=constants
    ).monic()
    ambient = constants.frac_field(*variables)
    polynomial = sp.Poly(product.as_expr(), *comparison, domain=ambient).monic()
    inner = smallest_coefficient(polynomial, ambient)
    degree = max(sum(m[: len(comparison)]) for m in product.monoms())
    if _degree(inner, variables, constants) != degree:
        return None
    # For a true symmetric near-separated polynomial, every nonconstant monic
    # coefficient is a fractional linear transform of its right component.
    # This identity rejects nonsymmetric and higher-rank candidates exactly.
    if not _same_polynomial(_cross(inner, variables, comparison, constants), product):
        return None
    outer = RationalCoordinateMap(ambient, inner, parameter).express(function)
    if outer is None:
        return None
    outer = _normal(outer, (parameter,), constants)
    result = MultivariateRationalDecomposition(
        original, variables, parameter, constants, outer, inner
    )
    return result if result.verify() else None


@dataclass(frozen=True)
class MultivariateRationalDecompositionResult:
    """A searched prefix of all two-factor classes over the declared constants.

    Completeness is modulo Möbius changes of intermediate coordinate, with
    inner total degree >1 and outer degree >1. It does not assert recursive
    chains, larger constant fields, or other multivariate decomposition notions.
    """

    original_expression: sp.Expr
    variables: tuple[sp.Symbol, ...]
    parameter: sp.Symbol
    constants: object
    comparison_variables: tuple[sp.Symbol, ...]
    factors: tuple[sp.Poly, ...]
    examined: int
    candidate_count: int
    decompositions: tuple[MultivariateRationalDecomposition, ...]

    @property
    def complete(self):
        return self.examined == self.candidate_count

    @property
    def status(self):
        if self.decompositions:
            return "decomposable"
        if not self.complete:
            return "incomplete"
        if _degree(self.original_expression, self.variables, self.constants) < 2:
            return "degree_below_two"
        return "indecomposable"

    def verify(self):
        from .functional_decomposition import _integer

        try:
            _symbols(self.variables, self.parameter, self.comparison_variables)
            if len(self.comparison_variables) != len(self.variables):
                return False
            if _constant_domain(self.constants) != self.constants:
                return False
            _integer(self.examined, "examined", 0)
            _integer(self.candidate_count, "candidate_count", 0)
            function = _normal(self.original_expression, self.variables, self.constants)
            degree = _degree(function, self.variables, self.constants)
            if not _targets(degree):
                return (
                    self.factors == ()
                    and self.examined == self.candidate_count == 0
                    and not self.decompositions
                )
            cross = _cross(function, self.variables, self.comparison_variables, self.constants)
            if not self.factors or any(
                not isinstance(p, sp.Poly)
                or p.gens != cross.gens
                or p.domain != self.constants
                or p.LC() != 1
                or not p.is_irreducible
                for p in self.factors
            ):
                return False
            reconstructed = sp.Poly(
                sp.prod(p.as_expr() for p in self.factors), *cross.gens, domain=self.constants
            )
            if not _same_polynomial(reconstructed, cross):
                return False
            data = _product_data(
                self.factors, degree, self.variables, self.comparison_variables, self.constants
            )
            if self.candidate_count != data[-1] or self.examined > self.candidate_count:
                return False
            expected = tuple(
                answer
                for indices in islice(_products(data), self.examined)
                if (
                    answer := _recover(
                        self.original_expression,
                        function,
                        self.variables,
                        self.parameter,
                        self.constants,
                        self.comparison_variables,
                        self.factors,
                        indices,
                    )
                )
                is not None
            )
            return self.decompositions == expected and all(d.verify() for d in self.decompositions)
        except (ValueError, TypeError, AttributeError, BasePolynomialError, ZeroDivisionError):
            return False


def multivariate_rational_decompose(
    expression, variables, *, parameter=None, constants=sp.QQ, max_candidates=256, max_degree=32
):
    """Find GRS02 uni-multivariate two-factor decompositions up to a budget.

    Degrees are rational total degrees in the input variables. Candidate
    products have equal proper-divisor degrees in both variable blocks and
    vanish on the diagonal. Counting is polynomial in the degree bound and
    number of factors; enumeration is lazy. Factorization and wall-clock time
    are not bounded by ``max_candidates``. Unsupported exact operations raise.
    """
    from .functional_decomposition import (
        DecompositionLimitError,
        FunctionalDecompositionError,
        _integer,
    )

    try:
        variables = tuple(variables)
    except TypeError as exc:
        raise FunctionalDecompositionError("variables must be a nonempty Symbol sequence") from exc
    parameter = sp.Dummy("intermediate") if parameter is None else parameter
    _symbols(variables, parameter)
    constants = _constant_domain(constants)
    _integer(max_candidates, "max_candidates", 0)
    _integer(max_degree, "max_degree", 1)
    original = expression.as_expr() if isinstance(expression, sp.Poly) else sp.sympify(expression)
    function = _normal(original, variables, constants)
    degree = _degree(function, variables, constants)
    if degree > max_degree:
        raise DecompositionLimitError(
            f"rational total degree {degree} exceeds max_degree={max_degree}"
        )
    comparison = tuple(sp.Dummy(f"comparison_{i}") for i in range(len(variables)))
    factors, count, answers = (), 0, []
    if _targets(degree):
        cross = _cross(function, variables, comparison, constants)
        _, factorization = cross.factor_list()
        if any(m != 1 for _, m in factorization):
            raise ArithmeticError(
                "the characteristic-zero difference polynomial must be squarefree"
            )
        factors = tuple(p.monic() for p, _ in factorization)
        data = _product_data(factors, degree, variables, comparison, constants)
        count = data[-1]
        for indices in islice(_products(data), max_candidates):
            answer = _recover(
                original, function, variables, parameter, constants, comparison, factors, indices
            )
            if answer is not None:
                answers.append(answer)
    return MultivariateRationalDecompositionResult(
        original,
        variables,
        parameter,
        constants,
        comparison,
        factors,
        min(max_candidates, count),
        count,
        tuple(answers),
    )
