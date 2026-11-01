"""Independent exhaustive verification of rational decomposition searches.

The search builds candidates using a weighted subset generator and recovers
outer functions in quotient algebras. This checker uses a separate factor-subset
enumeration and a necessary near-separated-polynomial identity. It does not
reconstruct answers with the search's quotient-algebra algorithm.
"""

from __future__ import annotations

from collections import Counter
from functools import cache

import sympy as sp


def _degree(expression: sp.Expr, variable: sp.Symbol, constants: object) -> int:
    numerator, denominator = sp.fraction(sp.cancel(expression))
    return max(
        sp.Poly(numerator, variable, domain=constants).degree(),
        sp.Poly(denominator, variable, domain=constants).degree(),
    )


def _candidate_products(factors, variable, comparison, degree):
    """Enumerate eligible products independently of the search's generator."""
    diagonal = sp.Poly(comparison - variable, comparison, variable, domain=factors[0].domain)
    diagonal_indices = [index for index, factor in enumerate(factors) if factor == diagonal]
    if len(diagonal_indices) != 1:
        raise ValueError("difference polynomial must have exactly one diagonal factor")
    remaining = tuple(
        factor for index, factor in enumerate(factors) if index != diagonal_indices[0]
    )
    weights = tuple(factor.degree(comparison) for factor in remaining)
    for right_degree in range(2, degree):
        if degree % right_degree:
            continue
        target = right_degree - 1

        @cache
        def feasible(index, remainder):
            if remainder == 0:
                return True
            if index == len(weights) or remainder < 0:
                return False
            return feasible(index + 1, remainder) or feasible(index + 1, remainder - weights[index])

        def visit(index, remainder, selected):
            if remainder == 0:
                yield selected
                return
            if index == len(weights) or remainder < 0 or not feasible(index, remainder):
                return
            yield from visit(index + 1, remainder, selected)
            yield from visit(
                index + 1,
                remainder - weights[index],
                (*selected, remaining[index]),
            )

        for selected in visit(0, target, ()):
            yield right_degree, (diagonal, *selected)


def _normalized_difference(expression, variable, comparison, constants):
    """Compute the monic near-separated polynomial of a rational function."""
    numerator, denominator = sp.fraction(sp.cancel(expression))
    ring = constants.frac_field(variable)
    polynomial = sp.Poly(
        numerator.subs(variable, comparison) * denominator
        - denominator.subs(variable, comparison) * numerator,
        comparison,
        domain=ring,
    )
    return polynomial.monic()


def _near_separated_candidate(product, right_degree, variable, comparison, constants):
    """Check whether a factor product is a rational-function cross polynomial.

    If f = outer(inner), the monic product containing the diagonal factor is
    the near-separated polynomial of inner. Every nonconstant coefficient of
    that monic polynomial is a fractional linear transform of inner. Testing
    all such coefficients avoids relying on the search's first-coefficient
    recovery heuristic.
    """
    ring = constants.frac_field(variable)
    polynomial = sp.Poly(
        sp.prod(factor.as_expr() for factor in product), comparison, domain=ring
    ).monic()
    for coefficient in polynomial.all_coeffs()[1:]:
        if variable not in coefficient.free_symbols:
            continue
        if _degree(coefficient, variable, constants) != right_degree:
            continue
        if _normalized_difference(coefficient, variable, comparison, constants) == polynomial:
            return polynomial
    return None


def verify_complete_rational_search(result) -> bool:
    """Verify exhaustive factor-product coverage without replaying recovery.

    Every eligible right degree divides the input's rational degree. A complete
    search must report one verified decomposition for each product satisfying
    the necessary near-separated identity, and none for other products.
    Factor irreducibility and reconstruction of the full difference polynomial
    are checked here, independently of candidate recovery.
    """
    variable, comparison = result.variable, result.comparison_variable
    constants = result.constants
    degree = _degree(result.original_expression, variable, constants)
    if degree < 2:
        return (
            result.factors == ()
            and result.candidate_count == result.examined == 0
            and not result.decompositions
        )
    if not result.factors or any(
        not factor.is_irreducible
        or factor.gens != (comparison, variable)
        or factor.domain != constants
        or factor.LC() != 1
        for factor in result.factors
    ):
        return False
    difference = _normalized_difference(result.original_expression, variable, comparison, constants)
    product = sp.Poly(
        sp.prod(factor.as_expr() for factor in result.factors),
        comparison,
        domain=constants.frac_field(variable),
    ).monic()
    if product != difference:
        return False
    expected = Counter()
    count = 0
    for right_degree, factors in _candidate_products(result.factors, variable, comparison, degree):
        count += 1
        candidate = _near_separated_candidate(
            factors, right_degree, variable, comparison, constants
        )
        if candidate is not None:
            expected[candidate.as_expr()] += 1
    if count != result.candidate_count or count != result.examined:
        return False
    actual = Counter()
    for decomposition in result.decompositions:
        if (
            decomposition.original_expression != result.original_expression
            or decomposition.variable != variable
            or decomposition.parameter != result.parameter
            or decomposition.constants != constants
            or not decomposition.verify()
        ):
            return False
        actual[
            _normalized_difference(decomposition.inner, variable, comparison, constants).as_expr()
        ] += 1
    return actual == expected
