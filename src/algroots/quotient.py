"""Exact finite quotient algebras for zero-dimensional polynomial ideals.

This module is the canonical home for the equality-only algebra shared by the
RUR, action-matrix, and exact border-basis backends.  It deliberately contains
no semialgebraic Boolean, sign, inequality, or quantifier logic.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import cached_property
from itertools import product
from time import perf_counter

import sympy as sp
from sympy.polys.domains import Domain
from sympy.polys.orderings import grevlex

from ._validation import validate_integer
from .errors import QuotientAlgebraError


def _require_exact_domain(domain: Domain) -> Domain:
    """Return an exact rational/algebraic field accepted by the quotient core."""

    gaussian_domains = tuple(
        item for item in (getattr(sp, "ZZ_I", None), getattr(sp, "QQ_I", None)) if item is not None
    )
    if domain in gaussian_domains:
        # Normalize SymPy's Gaussian domains to the same AlgebraicField family
        # used for all other exact algebraic coefficients.  This keeps result
        # contracts uniform and lets callers rely on ``is_AlgebraicField``.
        return sp.QQ.algebraic_field(sp.I)
    if domain not in (sp.ZZ, sp.QQ) and not getattr(domain, "is_AlgebraicField", False):
        raise QuotientAlgebraError(
            "quotient algebra requires rational or exact algebraic-number coefficients, "
            f"not coefficient domain {domain}"
        )
    return domain if domain.is_Field else domain.get_field()


def _exact_polynomials(
    expressions: Iterable[sp.Expr], variables: Sequence[sp.Symbol]
) -> tuple[tuple[sp.Poly, ...], Domain]:
    raw = tuple(sp.sympify(expr) for expr in expressions)
    try:
        polys, options = sp.parallel_poly_from_expr(raw, *variables, extension=True)
    except (
        sp.PolynomialError,
        sp.polys.polyerrors.CoercionFailed,
        ValueError,
        TypeError,
        NotImplementedError,
    ) as exc:
        raise QuotientAlgebraError(
            "quotient algebra requires polynomial equations over an exact algebraic coefficient field"
        ) from exc
    domain = _require_exact_domain(options["domain"])
    converted: list[sp.Poly] = []
    for poly in polys:
        try:
            exact = sp.Poly(sp.expand(poly.as_expr()), *variables, domain=domain)
        except (
            sp.PolynomialError,
            sp.polys.polyerrors.CoercionFailed,
            ValueError,
            TypeError,
            NotImplementedError,
        ) as exc:
            raise QuotientAlgebraError(
                "could not embed the polynomial system in one exact coefficient field"
            ) from exc
        if exact.total_degree() < 0:
            raise QuotientAlgebraError("zero polynomial is not a valid defining equation")
        converted.append(exact)
    return tuple(converted), domain


def _leading_exponent_grevlex(poly: sp.Poly) -> tuple[int, ...]:
    terms = poly.terms(order=grevlex)
    if not terms:
        raise QuotientAlgebraError("zero polynomial has no leading monomial")
    return tuple(int(value) for value in terms[0][0])


def _leading_exponent(poly: sp.Poly, order) -> tuple[int, ...]:
    try:
        monomial = poly.LM(order=order)
    except (sp.PolynomialError, ValueError, TypeError) as exc:
        raise QuotientAlgebraError("could not determine a Groebner leading monomial") from exc
    return tuple(int(value) for value in monomial.exponents)


def _componentwise_leq(left: Sequence[int], right: Sequence[int]) -> bool:
    return all(a <= b for a, b in zip(left, right, strict=True))


def _minimal_leading_exponents(
    leading_exponents: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
    generators = tuple(
        tuple(int(value) for value in exponent)
        for exponent in leading_exponents
        if any(int(value) for value in exponent)
    )
    return tuple(
        generator
        for generator in generators
        if not any(
            other != generator and _componentwise_leq(other, generator) for other in generators
        )
    )


def _pure_power_bounds(
    leading_exponents: Sequence[Sequence[int]], variable_count: int
) -> tuple[int, ...]:
    bounds: list[int] = []
    for index in range(variable_count):
        pure_powers = [
            int(exponent[index])
            for exponent in leading_exponents
            if int(exponent[index]) > 0
            and all(int(value) == 0 for pos, value in enumerate(exponent) if pos != index)
        ]
        if not pure_powers:
            raise QuotientAlgebraError("system does not expose a finite standard-monomial basis")
        bounds.append(min(pure_powers))
    return tuple(bounds)


def standard_exponent_count(leading_exponents: Sequence[Sequence[int]], variable_count: int) -> int:
    """Count standard monomials without materializing the quotient staircase."""

    bounds = _pure_power_bounds(leading_exponents, variable_count)
    minimal = _minimal_leading_exponents(leading_exponents)
    suffix_sizes = [1] * (variable_count + 1)
    for index in range(variable_count - 1, -1, -1):
        suffix_sizes[index] = suffix_sizes[index + 1] * bounds[index]

    prefix: list[int] = []

    def count_from(index: int) -> int:
        # A generator supported entirely in the assigned prefix already divides
        # every completion, so the whole suffix belongs to the leading ideal.
        for generator in minimal:
            if all(generator[pos] == 0 for pos in range(index, variable_count)) and all(
                generator[pos] <= prefix[pos] for pos in range(index)
            ):
                return 0

        # If each generator is already blocked by an assigned exponent then no
        # suffix completion can enter the leading ideal; count it in one step.
        if minimal and all(
            any(prefix[pos] < generator[pos] for pos in range(index)) for generator in minimal
        ):
            return suffix_sizes[index]

        if index == variable_count:
            return 1
        total = 0
        for exponent in range(bounds[index]):
            prefix.append(exponent)
            total += count_from(index + 1)
            prefix.pop()
        return total

    return count_from(0)


def standard_exponents(
    leading_exponents: Sequence[Sequence[int]], variable_count: int
) -> tuple[tuple[int, ...], ...]:
    """Return the standard-monomial exponent vectors of a finite quotient."""

    bounds = _pure_power_bounds(leading_exponents, variable_count)
    basis = [
        tuple(candidate)
        for candidate in product(*(range(bound) for bound in bounds))
        if not any(_componentwise_leq(leading, candidate) for leading in leading_exponents)
    ]
    basis.sort(key=lambda exponent: (sum(exponent), exponent))
    return tuple(basis)


def monomial_from_exponent(variables: Sequence[sp.Symbol], exponent: Sequence[int]) -> sp.Expr:
    monomial = sp.Integer(1)
    for variable, power in zip(variables, exponent, strict=True):
        monomial *= variable ** int(power)
    return monomial


def _normal_form_for_basis(
    groebner_basis: sp.polys.polytools.GroebnerBasis, expression: sp.Expr
) -> sp.Expr:
    try:
        _, remainder = groebner_basis.reduce(sp.expand(expression))
    except (sp.PolynomialError, ValueError, TypeError) as exc:
        raise QuotientAlgebraError(f"Groebner reduction failed: {exc}") from exc
    return sp.expand(remainder)


def _coefficient_vector(
    expression: sp.Expr,
    variables: Sequence[sp.Symbol],
    basis_exponents: Sequence[Sequence[int]],
    domain: Domain,
) -> sp.Matrix:
    try:
        poly = sp.Poly(sp.expand(expression), *variables, domain=domain)
    except (
        sp.PolynomialError,
        sp.polys.polyerrors.CoercionFailed,
        ValueError,
        TypeError,
    ) as exc:
        raise QuotientAlgebraError(
            "normal form could not be represented over the quotient coefficient field"
        ) from exc
    coefficient_rules = {tuple(int(v) for v in mon): coeff for mon, coeff in poly.terms()}
    return sp.Matrix(
        [
            coefficient_rules.get(tuple(int(v) for v in exponent), sp.Integer(0))
            for exponent in basis_exponents
        ]
    )


@dataclass(frozen=True)
class SeparatingElement:
    """Certified separating element data for a finite quotient algebra."""

    expression: sp.Expr
    coefficients: tuple[int, ...]
    defining_polynomial: sp.Poly
    coordinate_denominator: sp.Poly
    trace_vector: sp.Matrix
    power_vectors: tuple[sp.Matrix, ...]
    geometric_solution_count: int


@dataclass(frozen=True)
class QuotientAlgebra:
    """Exact finite quotient algebra ``K[x_1,...,x_n]/I``.

    This is an expert-facing structural API.  The object owns a Gröbner basis,
    the standard-monomial staircase, exact quotient coordinates, multiplication
    operators, trace pairing, and separating-element construction.  Higher
    algorithms such as RUR, action matrices, and border bases should consume
    this object instead of reimplementing those kernels.
    """

    variables: tuple[sp.Symbol, ...]
    groebner_basis: sp.polys.polytools.GroebnerBasis
    domain: Domain
    standard_exponents: tuple[tuple[int, ...], ...]
    normal_form_cache_size: int = 128

    @classmethod
    def from_polynomials(
        cls,
        polynomials: Iterable[sp.Expr],
        variables: Sequence[sp.Symbol],
        *,
        order: str = "grevlex",
        max_dimension: int | None = None,
        normal_form_cache_size: int = 128,
    ) -> QuotientAlgebra:
        validate_integer(normal_form_cache_size, "normal_form_cache_size", minimum=0)
        if max_dimension is not None:
            validate_integer(max_dimension, "max_dimension")
        variable_tuple = tuple(variables)
        if not variable_tuple:
            raise QuotientAlgebraError("at least one variable is required")
        poly_objects, domain = _exact_polynomials(polynomials, variable_tuple)
        if not poly_objects:
            raise QuotientAlgebraError("at least one polynomial generator is required")
        basis = sp.groebner(
            [poly.as_expr() for poly in poly_objects],
            *variable_tuple,
            order=order,
            domain=domain,
        )
        return cls.from_groebner_basis(
            basis,
            variable_tuple,
            max_dimension=max_dimension,
            normal_form_cache_size=normal_form_cache_size,
        )

    @classmethod
    def from_groebner_basis(
        cls,
        groebner_basis: sp.polys.polytools.GroebnerBasis,
        variables: Sequence[sp.Symbol] | None = None,
        *,
        max_dimension: int | None = None,
        normal_form_cache_size: int = 128,
    ) -> QuotientAlgebra:
        validate_integer(normal_form_cache_size, "normal_form_cache_size", minimum=0)
        if max_dimension is not None:
            validate_integer(max_dimension, "max_dimension")
        variable_tuple = tuple(variables or groebner_basis.gens)
        if not variable_tuple:
            raise QuotientAlgebraError("at least one variable is required")
        if tuple(groebner_basis.gens) != variable_tuple:
            raise QuotientAlgebraError(
                "Groebner basis generators must match the quotient variable order"
            )
        domain = _require_exact_domain(groebner_basis.domain)
        unit = sp.Poly(1, *variable_tuple, domain=domain)
        if len(groebner_basis.polys) == 1 and groebner_basis.polys[0].as_expr() == unit.as_expr():
            return cls(variable_tuple, groebner_basis, domain, tuple(), normal_form_cache_size)
        if not groebner_basis.is_zero_dimensional:
            raise QuotientAlgebraError("quotient algebra requires a zero-dimensional ideal")
        leading = tuple(
            _leading_exponent(poly, groebner_basis.order) for poly in groebner_basis.polys
        )
        count = standard_exponent_count(leading, len(variable_tuple))
        if max_dimension is not None and count > max_dimension:
            raise QuotientAlgebraError(f"quotient dimension exceeds max_dimension={max_dimension}")
        exponents = standard_exponents(leading, len(variable_tuple))
        if len(exponents) != count:  # defensive consistency check
            raise QuotientAlgebraError(
                "standard-monomial counting disagrees with staircase construction"
            )
        return cls(variable_tuple, groebner_basis, domain, exponents, normal_form_cache_size)

    @property
    def dimension(self) -> int:
        """Vector-space dimension of the quotient, counted with multiplicity."""

        return len(self.standard_exponents)

    @cached_property
    def standard_monomials(self) -> tuple[sp.Expr, ...]:
        return tuple(
            monomial_from_exponent(self.variables, exponent) for exponent in self.standard_exponents
        )

    @cached_property
    def _basis_index(self) -> dict[tuple[int, ...], int]:
        return {exponent: index for index, exponent in enumerate(self.standard_exponents)}

    def normal_form(self, expression: sp.Expr) -> sp.Expr:
        """Return the exact Gröbner normal form of ``expression`` modulo the ideal."""

        expression = sp.expand(expression)
        cache, stats = self._normal_form_cache
        if expression in cache:
            stats["hits"] += 1
            cache.move_to_end(expression)
            return cache[expression]
        stats["misses"] += 1
        started = perf_counter()
        remainder = _normal_form_for_basis(self.groebner_basis, expression)
        stats["reduction_seconds"] += perf_counter() - started
        # Do not retain large input/remainder expressions merely to save a reduction.
        budget = max(32, min(256, self.dimension * 2))
        small_coefficients = all(
            max(abs(int(value.p)).bit_length(), int(value.q).bit_length()) <= 8192
            for value in (expression.atoms(sp.Rational) | remainder.atoms(sp.Rational))
        )
        if (
            self.normal_form_cache_size
            and small_coefficients
            and sp.count_ops(expression) <= budget
            and sp.count_ops(remainder) <= budget
        ):
            cache[expression] = remainder
            if len(cache) > self.normal_form_cache_size:
                cache.popitem(last=False)
                stats["evictions"] += 1
        return remainder

    @cached_property
    def _normal_form_cache(self):
        return OrderedDict(), {"hits": 0, "misses": 0, "evictions": 0, "reduction_seconds": 0.0}

    @property
    def normal_form_cache_info(self):
        cache, stats = self._normal_form_cache
        return dict(stats, entries=len(cache), capacity=self.normal_form_cache_size)

    def clear_normal_form_cache(self):
        cache, stats = self._normal_form_cache
        cache.clear()
        stats.update(hits=0, misses=0, evictions=0, reduction_seconds=0.0)

    @cached_property
    def _operation_statistics(self):
        return dict(coordinate_basis_hits=0, basis_stream_cache_hits=0, basis_stream_peak_entries=0)

    @property
    def operation_diagnostics(self):
        """Copy of measured exact-operation costs; not proof evidence."""
        return dict(self._operation_statistics)

    @cached_property
    def _monomial_index(self):
        return {monomial: index for index, monomial in enumerate(self.standard_monomials)}

    def coordinate_vector(self, expression: sp.Expr) -> sp.Matrix:
        """Return quotient coordinates in the standard-monomial basis."""

        if not self.standard_exponents:
            return sp.zeros(0, 1)
        index = self._monomial_index.get(expression)
        if index is not None:
            self._operation_statistics["coordinate_basis_hits"] += 1
            vector = sp.zeros(self.dimension, 1)
            vector[index] = 1
            return vector
        remainder = self.normal_form(expression)
        vector = _coefficient_vector(
            remainder, self.variables, self.standard_exponents, self.domain
        )
        reconstructed = sum(
            vector[index] * self.standard_monomials[index] for index in range(self.dimension)
        )
        if sp.expand(remainder - reconstructed) != 0:
            raise QuotientAlgebraError(
                "Groebner remainder contains a monomial outside the quotient basis"
            )
        return vector

    @cached_property
    def variable_multiplication_matrices(self) -> tuple[sp.Matrix, ...]:
        """Multiplication matrices for the quotient coordinate variables."""

        if self.dimension == 0:
            return tuple(sp.zeros(0, 0) for _ in self.variables)
        started = perf_counter()
        matrices: list[sp.Matrix] = []
        for variable in self.variables:
            columns = [
                self.coordinate_vector(variable * monomial) for monomial in self.standard_monomials
            ]
            entries = {
                (r, c): value
                for c, column in enumerate(columns)
                for r, value in enumerate(column)
                if value != 0
            }
            from .rational_univariate.linear_algebra import preferred_matrix_storage

            matrix = sp.SparseMatrix(self.dimension, self.dimension, entries)
            matrices.append(
                matrix if preferred_matrix_storage(matrix) == "sparse" else sp.Matrix(matrix)
            )
        self._operation_statistics["variable_actions_seconds"] = perf_counter() - started
        return tuple(matrices)

    @cached_property
    def coordinate_normal_forms(self) -> tuple[tuple[sp.Expr, ...], ...]:
        """Coordinates of each variable itself in the quotient basis."""

        forms: list[tuple[sp.Expr, ...]] = []
        for index, variable in enumerate(self.variables):
            unit_exponent = tuple(1 if pos == index else 0 for pos in range(len(self.variables)))
            if unit_exponent in self._basis_index:
                coeffs = [sp.Integer(0)] * self.dimension
                coeffs[self._basis_index[unit_exponent]] = sp.Integer(1)
                forms.append(tuple(coeffs))
            else:
                forms.append(tuple(self.coordinate_vector(variable)))
        return tuple(forms)

    @cached_property
    def basis_multiplication_matrices(self) -> tuple[sp.Matrix, ...]:
        """Multiplication matrices for all standard basis monomials."""

        if self.dimension == 0:
            return tuple()
        identity = sp.eye(self.dimension)
        power_cache: list[dict[int, sp.Matrix]] = [
            {0: identity, 1: matrix} for matrix in self.variable_multiplication_matrices
        ]
        matrices: list[sp.Matrix] = []
        for exponent in self.standard_exponents:
            matrix = identity
            for index, power in enumerate(exponent):
                if not power:
                    continue
                cache = power_cache[index]
                if int(power) not in cache:
                    cache[int(power)] = self.variable_multiplication_matrices[index] ** int(power)
                matrix = matrix * cache[int(power)]
            matrices.append(matrix)
        return tuple(matrices)

    def _monomial_matrix(self, exponent):
        """Stream one basis action; retain only the variable actions."""
        matrix = sp.eye(self.dimension)
        for power, action in zip(exponent, self.variable_multiplication_matrices, strict=True):
            if power:
                matrix = matrix * action ** int(power)
        return matrix

    def _iter_basis_actions(self):
        if self.dimension == 0:
            return
        if len(self.variables) == 1:
            # A univariate staircase is consecutive: reuse the previous action
            # without retaining the entire sequence of powers.
            action = sp.eye(self.dimension)
            variable_action = self.variable_multiplication_matrices[0]
            power = 0
            for exponent in self.standard_exponents:
                while power < exponent[0]:
                    action = action * variable_action
                    power += 1
                yield action
        else:
            # At most eight D-by-D actions, independent of staircase size.
            # Degree-ordered staircases often keep a predecessor in this window.
            # Missing predecessors use the exact original power computation.
            cache = OrderedDict()
            zero = (0,) * len(self.variables)
            cache[zero] = sp.eye(self.dimension)
            for exponent in self.standard_exponents:
                if exponent in cache:
                    action = cache[exponent]
                else:
                    predecessor = None
                    for index, power in enumerate(exponent):
                        if power:
                            candidate = exponent[:index] + (power - 1,) + exponent[index + 1 :]
                            if candidate in cache:
                                predecessor = candidate, index
                                break
                    if predecessor is None:
                        action = self._monomial_matrix(exponent)
                    else:
                        previous, index = predecessor
                        action = cache[previous] * self.variable_multiplication_matrices[index]
                        self._operation_statistics["basis_stream_cache_hits"] += 1
                    cache[exponent] = action
                cache.move_to_end(exponent)
                if len(cache) > 8:
                    cache.popitem(last=False)
                self._operation_statistics["basis_stream_peak_entries"] = max(
                    self._operation_statistics["basis_stream_peak_entries"], len(cache)
                )
                yield action

    def multiplication_matrix(self, element: sp.Expr) -> sp.Matrix:
        """Multiplication by an element without caching all basis actions."""
        result = sp.zeros(self.dimension, self.dimension)
        for coefficient, exponent in zip(
            self.coordinate_vector(element), self.standard_exponents, strict=True
        ):
            if coefficient != 0:
                result += coefficient * self._monomial_matrix(exponent)
        return result

    @cached_property
    def trace_vector(self) -> sp.Matrix:
        """Exact traces, streaming one basis action at a time (quadratic storage)."""
        started = perf_counter()
        result = sp.Matrix([action.trace() for action in self._iter_basis_actions()])
        self._operation_statistics["trace_vector_seconds"] = perf_counter() - started
        return result

    @cached_property
    def trace_pairing(self) -> sp.Matrix:
        """Exact symmetric trace pairing, without retaining cubic basis actions."""
        started = perf_counter()
        pairing = sp.zeros(self.dimension, self.dimension)
        traces = self.trace_vector
        for left, action in enumerate(self._iter_basis_actions()):
            for right in range(left, self.dimension):
                value = (action[:, right].T * traces)[0]
                pairing[left, right] = pairing[right, left] = value
        self._operation_statistics["trace_pairing_seconds"] = perf_counter() - started
        return pairing

    @cached_property
    def geometric_solution_count(self) -> int:
        """Number of geometric points for a characteristic-zero finite quotient."""

        if not self.dimension:
            return 0
        pairing = self.trace_pairing
        started = perf_counter()
        # Local profiling favors exact domain elimination for QQ. Algebraic
        # conversion was slower, so retain the existing path for that domain.
        if self.domain == sp.QQ:
            rank = pairing.to_DM(domain=sp.QQ).rank()
            backend = "rational_domain"
        else:
            rank = pairing.rank()
            backend = "expression"
        self._operation_statistics.update(
            geometric_rank_seconds=perf_counter() - started, rank_backend=backend
        )
        return int(rank)

    def linear_combination_matrix(self, coefficients: Sequence[int | sp.Expr]) -> sp.Matrix:
        """Return multiplication by ``sum(c_i*x_i)`` without extra reductions."""

        coefficient_tuple = tuple(coefficients)
        if len(coefficient_tuple) != len(self.variables):
            raise QuotientAlgebraError("one coefficient is required per quotient variable")
        result = sp.SparseMatrix(self.dimension, self.dimension, {})
        for coefficient, matrix in zip(
            coefficient_tuple, self.variable_multiplication_matrices, strict=True
        ):
            result += sp.sympify(coefficient) * matrix
        return result

    def separator_matrix_columns(
        self, coefficients: Sequence[int | sp.Expr]
    ) -> tuple[tuple[sp.Expr, ...], ...]:
        """Return columns of a linear separator multiplication matrix.

        This direct form deliberately uses exactly one Gröbner reduction per
        staircase monomial.  The RUR path already owns the coordinate
        multiplication matrices and uses :meth:`linear_combination_matrix`;
        the numerical action path should not pay ``n*D`` reductions merely to
        form one candidate separator.
        """

        coefficient_tuple = tuple(coefficients)
        if len(coefficient_tuple) != len(self.variables):
            raise QuotientAlgebraError("one coefficient is required per quotient variable")
        linear_form = sum(
            sp.sympify(coefficient) * variable
            for coefficient, variable in zip(coefficient_tuple, self.variables, strict=True)
        )
        return tuple(
            tuple(self.coordinate_vector(linear_form * monomial))
            for monomial in self.standard_monomials
        )

    def separating_element(
        self,
        parameter: sp.Symbol,
        *,
        max_attempts: int = 64,
    ) -> SeparatingElement:
        """Choose a separating linear form and return exact RUR trace data."""

        if parameter in self.variables:
            raise QuotientAlgebraError("parameter must be distinct from quotient variables")
        if self.dimension == 0:
            raise QuotientAlgebraError("the unit ideal has no separating element")
        if max_attempts < 1:
            raise QuotientAlgebraError("max_attempts must be positive")
        expected_distinct_roots = self.geometric_solution_count
        for attempt in range(1, max_attempts + 1):
            coefficients = tuple(attempt**index for index in range(len(self.variables)))
            linear_form = sum(
                coefficient * variable
                for coefficient, variable in zip(coefficients, self.variables, strict=True)
            )
            multiplication = self.linear_combination_matrix(coefficients)
            # Full cyclic closure certifies equality of minimal and characteristic
            # polynomials. Short closure must retain the characteristic fallback:
            # dropping multiplicities would corrupt the trace RUR numerators.
            from .rational_univariate.linear_algebra import (
                packed_krylov_minimal_polynomial,
                preferred_matrix_storage,
            )

            if self.dimension >= 12 and preferred_matrix_storage(multiplication) == "sparse":
                try:
                    characteristic = sp.Poly(
                        packed_krylov_minimal_polynomial(multiplication, parameter).as_expr(),
                        parameter,
                        domain=self.domain,
                    )
                except ValueError:
                    characteristic = sp.Poly(
                        multiplication.charpoly(parameter).as_expr(), parameter, domain=self.domain
                    )
            else:
                characteristic = sp.Poly(
                    multiplication.charpoly(parameter).as_expr(), parameter, domain=self.domain
                )
            if characteristic.is_zero:
                continue
            squarefree = characteristic.sqf_part().monic()
            if squarefree.degree() != expected_distinct_roots:
                continue
            derivative = characteristic.diff()
            gcd = sp.gcd(characteristic, derivative)
            denominator = derivative.quo(gcd)
            powers = [sp.eye(self.dimension).col(0)]
            for _ in range(1, squarefree.degree()):
                powers.append(multiplication * powers[-1])
            return SeparatingElement(
                expression=linear_form,
                coefficients=coefficients,
                defining_polynomial=squarefree,
                coordinate_denominator=denominator,
                trace_vector=self.trace_vector,
                power_vectors=tuple(powers),
                geometric_solution_count=expected_distinct_roots,
            )
        raise QuotientAlgebraError("could not find a separating linear form within max_attempts")


__all__ = [
    "QuotientAlgebra",
    "QuotientAlgebraError",
    "SeparatingElement",
    "standard_exponent_count",
    "standard_exponents",
]
