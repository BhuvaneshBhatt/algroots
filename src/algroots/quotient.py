"""Exact finite quotient algebras for zero-dimensional polynomial ideals.

This module is the canonical home for the equality-only algebra shared by the
RUR, action-matrix, and exact border-basis backends.  It contains
no semialgebraic Boolean, sign, inequality, or quantifier logic.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from functools import cached_property
from itertools import product
from time import perf_counter

import sympy as sp
from sympy.polys.domains import Domain
from sympy.polys.matrices import DomainMatrix
from sympy.polys.matrices.exceptions import DMNonInvertibleMatrixError
from sympy.polys.orderings import grevlex

from ._validation import validate_integer
from .domains import domain_matrix, field_element, groebner, infer_coefficient_field, polynomial
from .errors import QuotientAlgebraError
from .finite_fields import FiniteFieldExtension, is_finite_field


def _require_quotient_domain(domain: Domain) -> Domain:
    """Validate exact number, finite, or rational-function coefficient fields."""

    if (not isinstance(domain, Domain) and hasattr(domain, "as_domain")) or (
        getattr(domain, "is_FractionField", False)
        and getattr(domain.domain, "is_AlgebraicFunctionField", False)
    ):
        from .domains import coefficient_domain

        domain = coefficient_domain(domain)
    if not isinstance(domain, Domain):
        raise QuotientAlgebraError("domain must be an exact SymPy coefficient domain")

    from .algebraic_function_fields import AlgebraicFunctionField

    if isinstance(domain, AlgebraicFunctionField):
        return domain
    if is_finite_field(domain):
        return domain
    gaussian_domains = tuple(
        item for item in (getattr(sp, "ZZ_I", None), getattr(sp, "QQ_I", None)) if item is not None
    )
    if domain in gaussian_domains:
        # Normalize SymPy's Gaussian domains to the same AlgebraicField family
        # used for all other exact algebraic coefficients.  This keeps result
        # contracts uniform and lets callers rely on ``is_AlgebraicField``.
        return sp.QQ.algebraic_field(sp.I)
    if getattr(domain, "is_FractionField", False):
        if any(not isinstance(p, sp.Symbol) for p in domain.symbols):
            raise QuotientAlgebraError("coefficient parameters must be Symbols")
        ground = domain.domain
        if (
            ground in (sp.ZZ, sp.QQ)
            or getattr(ground, "is_AlgebraicField", False)
            or ground in gaussian_domains
            or is_finite_field(ground)
        ):
            return _require_quotient_domain(ground).frac_field(*domain.symbols)
    if domain not in (sp.ZZ, sp.QQ) and not getattr(domain, "is_AlgebraicField", False):
        raise QuotientAlgebraError(
            "quotient algebra requires rational, exact algebraic-number, or explicit "
            "rational-function coefficients, "
            f"not coefficient domain {domain}"
        )
    return domain if domain.is_Field else domain.get_field()


def _require_exact_domain(domain: Domain) -> Domain:
    """Validate constants for characteristic-zero function-field and numerical algorithms."""
    domain = _require_quotient_domain(domain)
    if domain.characteristic() != 0:
        raise QuotientAlgebraError("this operation requires characteristic-zero coefficients")
    return domain


def _exact_polynomials(
    expressions: Iterable[sp.Expr], variables: Sequence[sp.Symbol], domain: Domain | None = None
) -> tuple[tuple[sp.Poly, ...], Domain]:
    raw = tuple(
        expr.as_expr() if isinstance(expr, sp.Poly) else sp.sympify(expr) for expr in expressions
    )
    if not raw:
        return (), _require_quotient_domain(sp.QQ if domain is None else domain)
    if any(expr.has(sp.Float) for expr in raw):
        raise QuotientAlgebraError(
            "quotient algebra requires exact coefficients, not floating-point inputs"
        )
    if domain is not None:
        domain = _require_quotient_domain(domain)
        _check_coefficient_symbols(domain, variables)
    try:
        if domain is None:
            if any(
                constant.has(*variables)
                for expression in raw
                for constant in expression.atoms(sp.CRootOf, sp.AlgebraicNumber)
            ):
                domain = _require_quotient_domain(infer_coefficient_field(raw, variables))
                polys = [polynomial(p, variables, domain) for p in raw]
            else:
                polys, options = sp.parallel_poly_from_expr(raw, *variables, extension=True)
                domain = _require_quotient_domain(options["domain"])
        else:
            polys = [polynomial(p, variables, domain) for p in raw]
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
    converted: list[sp.Poly] = []
    for poly in polys:
        try:
            exact = polynomial(poly, variables, domain)
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


def _check_coefficient_symbols(domain: Domain, variables: Sequence[sp.Symbol]) -> None:
    parameters = domain.symbols if getattr(domain, "is_FractionField", False) else ()
    parameters = (
        *parameters,
        *getattr(domain, "parameters", ()),
        *getattr(domain, "tower_symbols", ()),
    )
    ground = domain.domain if domain.is_FractionField else domain
    if getattr(domain, "is_AlgebraicFunctionField", False):
        parameters = (*parameters, domain.symbol)
    if isinstance(ground, FiniteFieldExtension):
        parameters = (*parameters, ground.symbol)
    if any(v == p or v.name == p.name for v in variables for p in parameters):
        raise QuotientAlgebraError("coefficient parameters and quotient variables must be distinct")


def _quotient_variables(variables: Sequence[sp.Symbol]) -> tuple[sp.Symbol, ...]:
    variables = tuple(variables)
    if not variables or any(not isinstance(v, sp.Symbol) for v in variables):
        raise QuotientAlgebraError(
            "at least one quotient variable is required; variables must be Symbols"
        )
    if len(set(variables)) != len(variables):
        raise QuotientAlgebraError("quotient variables must be distinct")
    return variables


def _leading_exponent_grevlex(poly: sp.Poly) -> tuple[int, ...]:
    terms = poly.terms(order=grevlex)
    if not terms:
        raise QuotientAlgebraError("zero polynomial has no leading monomial")
    return tuple(int(value) for value in terms[0][0])


def _leading_exponent(poly: sp.Poly, order) -> tuple[int, ...]:
    try:
        monomial = poly.LM(order=order)
    except (sp.PolynomialError, sp.polys.polyerrors.CoercionFailed, ValueError, TypeError) as exc:
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
    """Construct the monomial with the supplied coordinate exponents."""
    monomial = sp.Integer(1)
    for variable, power in zip(variables, exponent, strict=True):
        monomial *= variable ** int(power)
    return monomial


def _normal_form_for_basis(
    groebner_basis: sp.polys.polytools.GroebnerBasis, expression: sp.Expr
) -> sp.Expr:
    try:
        value = polynomial(expression, groebner_basis.gens, groebner_basis.domain)
        _, remainder = groebner_basis.reduce(value)
    except (sp.PolynomialError, sp.polys.polyerrors.CoercionFailed, ValueError, TypeError) as exc:
        raise QuotientAlgebraError(f"Groebner reduction failed: {exc}") from exc
    return sp.expand(remainder)


def _coefficient_vector(
    expression: sp.Expr,
    variables: Sequence[sp.Symbol],
    basis_exponents: Sequence[Sequence[int]],
    domain: Domain,
) -> sp.Matrix:
    try:
        poly = polynomial(expression, variables, domain)
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
    defining_polynomials: tuple[sp.Expr, ...] | None = None

    @classmethod
    def from_polynomials(
        cls,
        polynomials: Iterable[sp.Expr],
        variables: Sequence[sp.Symbol],
        *,
        domain: Domain | None = None,
        order: str = "grevlex",
        max_dimension: int | None = None,
        normal_form_cache_size: int = 128,
        groebner_backend="native",
        groebner_limits=None,
    ) -> QuotientAlgebra:
        """Construct a finite algebra using an explicitly selected, verified basis."""
        validate_integer(normal_form_cache_size, "normal_form_cache_size", minimum=0)
        if max_dimension is not None:
            validate_integer(max_dimension, "max_dimension")
        variable_tuple = _quotient_variables(variables)
        poly_objects, domain = _exact_polynomials(polynomials, variable_tuple, domain)
        if not poly_objects:
            raise QuotientAlgebraError("at least one polynomial generator is required")
        basis = groebner(
            poly_objects,
            variable_tuple,
            domain,
            order,
            backend=groebner_backend,
            limits=groebner_limits,
        )
        quotient = cls.from_groebner_basis(
            basis,
            variable_tuple,
            max_dimension=max_dimension,
            normal_form_cache_size=normal_form_cache_size,
        )
        return replace(quotient, defining_polynomials=tuple(p.as_expr() for p in poly_objects))

    @classmethod
    def from_certificate(cls, certificate, **options):
        """Construct a quotient by replaying reusable exact basis identities."""
        from .basis_certificates import GroebnerCertificate
        from .verified_families import VerifiedGroebnerFamily

        if not isinstance(certificate, (GroebnerCertificate, VerifiedGroebnerFamily)):
            raise TypeError("certificate must be a GroebnerCertificate")
        quotient = cls.from_groebner_basis(certificate.verified_basis(), **options)
        return replace(quotient, defining_polynomials=certificate.source)

    @classmethod
    def from_groebner_basis(
        cls,
        groebner_basis: sp.polys.polytools.GroebnerBasis,
        variables: Sequence[sp.Symbol] | None = None,
        *,
        max_dimension: int | None = None,
        normal_form_cache_size: int = 128,
    ) -> QuotientAlgebra:
        """Construct a finite algebra from an existing exact Gröbner basis."""
        validate_integer(normal_form_cache_size, "normal_form_cache_size", minimum=0)
        if max_dimension is not None:
            validate_integer(max_dimension, "max_dimension")
        variable_tuple = _quotient_variables(
            groebner_basis.gens if variables is None else variables
        )
        if tuple(groebner_basis.gens) != variable_tuple:
            raise QuotientAlgebraError(
                "Groebner basis generators must match the quotient variable order"
            )
        domain = _require_quotient_domain(groebner_basis.domain)
        _check_coefficient_symbols(domain, variable_tuple)
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
        """The ordered monomials representing quotient coordinates."""
        return tuple(
            monomial_from_exponent(self.variables, exponent) for exponent in self.standard_exponents
        )

    @cached_property
    def _basis_index(self) -> dict[tuple[int, ...], int]:
        return {exponent: index for index, exponent in enumerate(self.standard_exponents)}

    def normal_form(self, expression: sp.Expr) -> sp.Expr:
        """Return the exact Gröbner normal form of ``expression`` modulo the ideal."""

        expression = sp.expand(expression)
        if expression.has(sp.Float):
            raise QuotientAlgebraError("quotient elements must have exact coefficients")
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
        """Return reduction counts, cache occupancy and the configured capacity."""
        cache, stats = self._normal_form_cache
        return dict(stats, entries=len(cache), capacity=self.normal_form_cache_size)

    def clear_normal_form_cache(self):
        """Discard cached normal forms and reset their counters."""
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
        if not polynomial(remainder - reconstructed, self.variables, self.domain).is_zero:
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
        for variable_index, variable in enumerate(self.variables):
            entries = {}
            for column, (monomial, exponent) in enumerate(
                zip(self.standard_monomials, self.standard_exponents, strict=True)
            ):
                product = tuple(e + (i == variable_index) for i, e in enumerate(exponent))
                if product in self._basis_index:
                    entries[self._basis_index[product], column] = sp.S.One
                    continue
                remainder = polynomial(
                    self.normal_form(variable * monomial), self.variables, self.domain
                )
                for exponent, value in remainder.rep.to_dict().items():
                    if exponent not in self._basis_index:
                        raise QuotientAlgebraError(
                            "remainder contains a monomial outside the quotient basis"
                        )
                    entries[self._basis_index[exponent], column] = self.domain.to_sympy(value)
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
        identity = sp.SparseMatrix.eye(self.dimension)
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
        matrix = sp.SparseMatrix.eye(self.dimension)
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
            action = sp.SparseMatrix.eye(self.dimension)
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
            cache[zero] = sp.SparseMatrix.eye(self.dimension)
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

    @cached_property
    def _variable_domain_matrices(self) -> tuple[DomainMatrix, ...]:
        return tuple(domain_matrix(m, self.domain) for m in self.variable_multiplication_matrices)

    def _iter_basis_domain_actions(self):
        """Stream canonical coefficient-domain actions through a bounded cache."""
        active = {
            i for exponent in self.standard_exponents for i, power in enumerate(exponent) if power
        }
        if len(active) <= 1:
            index = next(iter(active), 0)
            action = DomainMatrix.eye((self.dimension, self.dimension), self.domain)
            power = 0
            for exponent in self.standard_exponents:
                while power < exponent[index]:
                    action = action * self._variable_domain_matrices[index]
                    power += 1
                self._operation_statistics["basis_stream_peak_entries"] = max(
                    self._operation_statistics["basis_stream_peak_entries"], 1
                )
                yield action
            return
        cache = OrderedDict()
        zero = (0,) * len(self.variables)
        cache[zero] = DomainMatrix.eye((self.dimension, self.dimension), self.domain)
        for exponent in self.standard_exponents:
            if exponent in cache:
                action = cache[exponent]
            else:
                action = None
                for i, power in enumerate(exponent):
                    if power:
                        predecessor = exponent[:i] + (power - 1,) + exponent[i + 1 :]
                        if predecessor in cache:
                            action = cache[predecessor] * self._variable_domain_matrices[i]
                            self._operation_statistics["basis_stream_cache_hits"] += 1
                            break
                if action is None:
                    action = DomainMatrix.eye((self.dimension, self.dimension), self.domain)
                    for power, matrix in zip(exponent, self._variable_domain_matrices, strict=True):
                        if power:
                            action = action * matrix.pow(int(power))
                cache[exponent] = action
            cache.move_to_end(exponent)
            if len(cache) > 8:
                cache.popitem(last=False)
            self._operation_statistics["basis_stream_peak_entries"] = max(
                self._operation_statistics["basis_stream_peak_entries"], len(cache)
            )
            yield action

    def _linear_combination_domain_matrix(self, coefficients):
        result = DomainMatrix.zeros((self.dimension, self.dimension), self.domain)
        for coefficient, matrix in zip(coefficients, self._variable_domain_matrices, strict=True):
            scalar = field_element(coefficient, self.domain)
            if scalar:
                result = result + matrix.scalarmul(scalar)
        return result

    def _matrix_expression(self, matrix):
        entries = matrix.to_dod()
        if sum(map(len, entries.values())) * 5 <= matrix.shape[0] * matrix.shape[1]:
            return sp.SparseMatrix(
                *matrix.shape,
                {
                    (i, j): self.domain.to_sympy(c)
                    for i, row in entries.items()
                    for j, c in row.items()
                },
            )
        return matrix.to_Matrix()

    def multiplication_matrix(self, element: sp.Expr, *, representation: str = "sympy"):
        """Multiplication over the declared coefficient field.

        ``representation='domain'`` returns a DomainMatrix. Fraction-field
        products always use domain arithmetic, including the default Matrix view.
        Elements are polynomial in the quotient variables; use ``inverse`` for units.
        """
        if representation not in ("sympy", "sparse", "domain"):
            raise ValueError("representation must be sympy, sparse or domain")
        if (
            representation in ("domain", "sparse")
            or getattr(self.domain, "is_FractionField", False)
            or getattr(self.domain, "is_FiniteExtension", False)
            or self.domain.characteristic()
        ):
            result = DomainMatrix.zeros((self.dimension, self.dimension), self.domain)
            for coefficient, exponent in zip(
                self.coordinate_vector(element), self.standard_exponents, strict=True
            ):
                scalar = field_element(coefficient, self.domain)
                if not scalar:
                    continue
                action = DomainMatrix.eye((self.dimension, self.dimension), self.domain)
                for power, matrix in zip(exponent, self._variable_domain_matrices, strict=True):
                    if power:
                        action = action * matrix.pow(int(power))
                result = result + action.scalarmul(scalar)
            if representation == "domain":
                return result
            entries = result.to_dod()
            if (
                representation == "sparse"
                or sum(map(len, entries.values())) * 10 < self.dimension**2
            ):
                return sp.SparseMatrix(
                    self.dimension,
                    self.dimension,
                    {
                        (i, j): self.domain.to_sympy(c)
                        for i, row in entries.items()
                        for j, c in row.items()
                    },
                )
            return result.to_Matrix()
        result = sp.SparseMatrix(self.dimension, self.dimension, {})
        for coefficient, exponent in zip(
            self.coordinate_vector(element), self.standard_exponents, strict=True
        ):
            if coefficient != 0:
                result += coefficient * self._monomial_matrix(exponent)
        return result

    def _polynomial_parameter(self, parameter: sp.Symbol) -> None:
        if not isinstance(parameter, sp.Symbol):
            raise QuotientAlgebraError("polynomial parameter must be a Symbol")
        if parameter in self.variables:
            raise QuotientAlgebraError("parameter must be distinct from quotient variables")
        _check_coefficient_symbols(self.domain, (parameter,))

    def characteristic_polynomial(self, element: sp.Expr, parameter: sp.Symbol) -> sp.Poly:
        """Characteristic polynomial of multiplication, preserving multiplicities."""
        self._polynomial_parameter(parameter)
        matrix = self.multiplication_matrix(element, representation="domain")
        from .rational_univariate.linear_algebra import sparse_characteristic_coefficients

        coefficients = (
            sparse_characteristic_coefficients(matrix)
            if self.dimension >= 12 and matrix.nnz() * 5 <= self.dimension**2
            else matrix.charpoly()
        )
        return sp.Poly.from_list(coefficients, parameter, domain=self.domain)

    def minimal_polynomial(self, element: sp.Expr, parameter: sp.Symbol) -> sp.Poly:
        """Monic least-degree annihilator over ``self.domain``, including nilpotents.

        The first exact dependence of 1, element, ... certifies minimality.
        Unlike a characteristic-polynomial replacement, short closure is valid.
        """
        from .rational_univariate.linear_algebra import packed_krylov_coefficients

        self._polynomial_parameter(parameter)
        matrix = self.multiplication_matrix(element, representation="domain")
        coefficients = packed_krylov_coefficients(matrix, self.domain, allow_short_closure=True)
        return sp.Poly.from_list(list(reversed(coefficients)), parameter, domain=self.domain)

    def trace(self, element: sp.Expr) -> sp.Expr:
        """Trace of multiplication over the declared coefficient field."""
        rows = self.multiplication_matrix(element, representation="domain").to_dod()
        value = sum(
            (rows.get(i, {}).get(i, self.domain.zero) for i in range(self.dimension)),
            self.domain.zero,
        )
        return self.domain.to_sympy(value)

    def norm(self, element: sp.Expr) -> sp.Expr:
        """Determinant of multiplication over the declared coefficient field."""
        matrix = self.multiplication_matrix(element, representation="domain")
        if self.dimension >= 12 and matrix.nnz() * 5 <= self.dimension**2:
            from .rational_univariate.linear_algebra import sparse_characteristic_coefficients

            constant = sparse_characteristic_coefficients(matrix)[-1]
            value = -constant if self.dimension % 2 else constant
        else:
            value = matrix.det()
        return self.domain.to_sympy(value)

    def inverse(self, element: sp.Expr) -> sp.Expr:
        """Inverse of a unit; nonunits in reducible/nonreduced algebras raise."""
        if not self.dimension:
            raise QuotientAlgebraError("the zero algebra has no unit inverse")
        matrix = self.multiplication_matrix(element, representation="domain")
        try:
            from .rational_univariate.linear_algebra import sparse_solve

            unit = DomainMatrix({0: {0: self.domain.one}}, (self.dimension, 1), self.domain)
            column = [row[0] for row in sparse_solve(matrix, unit).to_list()]
        except DMNonInvertibleMatrixError as exc:
            raise QuotientAlgebraError("element is not a unit in the quotient algebra") from exc
        expression = sum(
            self.domain.to_sympy(c) * m
            for c, m in zip(column, self.standard_monomials, strict=True)
        )
        return self.normal_form(expression)

    @cached_property
    def is_field(self) -> bool:
        """Certify whether this nonzero finite algebra is a field."""
        if not self.dimension or self.geometric_solution_count != self.dimension:
            return False
        if self.domain.characteristic():
            from .finite_decomposition import frobenius_matrix

            fixed = frobenius_matrix(self) - DomainMatrix.eye(
                (self.dimension, self.dimension), self.domain
            )
            return self.dimension - fixed.rank() == 1
        separator = self.separating_element(sp.Dummy("field_generator"))
        if getattr(self.domain, "is_AlgebraicFunctionField", False):
            factors = self.domain.factor_univariate(separator.defining_polynomial)[1]
            return len(factors) == 1 and factors[0][1] == 1
        return bool(separator.defining_polynomial.is_irreducible)

    def subfield(self, generators: Sequence[sp.Expr], *, over=None, generator_symbols=None):
        """Generate a subfield over this quotient's existing coefficient field.

        Returns a FiniteSubfield with constructive membership and embeddings.
        Use the separate function-field layer to generate over the constants.
        """
        from .subfields import FiniteSubfield

        _require_exact_domain(self.domain)

        return FiniteSubfield.generate(
            self, generators, over=over, generator_symbols=generator_symbols
        )

    def specialization_guard(self, *elements: sp.Expr) -> sp.Expr:
        """Denominator guard for this generic basis and the supplied elements.

        This excludes poles, not all dimension/discriminant changes. ``specialize``
        rebuilds from the original equations instead of assuming a stable fiber.
        """
        expressions = [p.as_expr() for p in self.groebner_basis.polys]
        expressions.extend(elements)
        denominators = []
        for expr in expressions:
            poly = polynomial(expr, self.variables, self.domain)
            for coefficient in poly.coeffs():
                denominator = sp.fraction(sp.cancel(coefficient))[1]
                if denominator != 1:
                    denominators.append(sp.Ne(denominator, 0))
        return sp.And(*denominators)

    def specialize(
        self,
        assignments: dict[sp.Symbol, sp.Expr],
        *,
        max_dimension: int | None = None,
        groebner_backend="native",
        groebner_limits=None,
    ):
        """Rebuild the specialized algebra; never reuse generic rank or inverses.

        Assignments must be constants in the existing ground field. A basis-only
        construction cannot recover discarded exceptional fibers and rejects
        nonempty specializations; construct from defining polynomials instead.
        Positive-dimensional specialized fibers retain the ordinary constructor error.
        """
        parameters = (
            tuple(self.domain.symbols) if getattr(self.domain, "is_FractionField", False) else ()
        )
        if not set(assignments) <= set(parameters):
            raise QuotientAlgebraError("specialization keys must be coefficient parameters")
        ground = self.domain.domain if parameters else self.domain
        if any(sp.sympify(v).has(sp.Float) for v in assignments.values()):
            raise QuotientAlgebraError("specialization values must be exact")
        try:
            substitutions = {p: ground.to_sympy(ground.convert(v)) for p, v in assignments.items()}
        except (sp.polys.polyerrors.CoercionFailed, ValueError, TypeError) as exc:
            raise QuotientAlgebraError(
                "specialization values must belong to the constant field"
            ) from exc
        equations = self.defining_polynomials
        if equations is None:
            if substitutions:
                raise QuotientAlgebraError(
                    "specialization requires original defining polynomials; cannot recover "
                    "exceptional fibers from a generic Groebner basis"
                )
            equations = tuple(p.as_expr() for p in self.groebner_basis.polys)
        specialized = tuple(p.subs(substitutions, simultaneous=True) for p in equations)
        if any(p.has(sp.zoo, sp.nan, sp.oo, -sp.oo) for p in specialized):
            raise QuotientAlgebraError("defining coefficients have a pole at the specialization")
        remaining = tuple(p for p in parameters if p not in substitutions)
        domain = ground.frac_field(*remaining) if remaining else ground
        # Zero generators impose no constraint; an entirely zero fiber is not finite.
        specialized = tuple(p for p in specialized if p != 0)
        if not specialized:
            raise QuotientAlgebraError("specialized ideal is not zero-dimensional")
        return type(self).from_polynomials(
            specialized,
            self.variables,
            domain=domain,
            order=str(self.groebner_basis.order),
            groebner_backend=groebner_backend,
            groebner_limits=groebner_limits,
            max_dimension=max_dimension,
            normal_form_cache_size=self.normal_form_cache_size,
        )

    @cached_property
    def _trace_domain_vector(self):
        entries = {}
        for i, action in enumerate(self._iter_basis_domain_actions()):
            value = sum(
                (row.get(j, self.domain.zero) for j, row in action.to_dod().items()),
                self.domain.zero,
            )
            if value:
                entries[i] = {0: value}
        return DomainMatrix(entries, (self.dimension, 1), self.domain)

    @cached_property
    def trace_vector(self) -> sp.Matrix:
        """Exact sparse trace functional in the standard-monomial basis."""
        started = perf_counter()
        result = self._matrix_expression(self._trace_domain_vector)
        self._operation_statistics["trace_vector_seconds"] = perf_counter() - started
        return result

    @cached_property
    def _trace_domain_pairing(self):
        entries = {}
        traces = self._trace_domain_vector
        for left, action in enumerate(self._iter_basis_domain_actions()):
            for right, row in (action.transpose() * traces).to_dod().items():
                value = row.get(0, self.domain.zero)
                if right >= left and value:
                    entries.setdefault(left, {})[right] = value
                    entries.setdefault(right, {})[left] = value
        matrix = DomainMatrix(entries, (self.dimension, self.dimension), self.domain)
        return matrix.to_dense() if matrix.nnz() * 5 > self.dimension**2 else matrix

    @cached_property
    def trace_pairing(self) -> sp.Matrix:
        """Exact trace pairing, preserving sparse support through field arithmetic."""
        started = perf_counter()
        result = self._matrix_expression(self._trace_domain_pairing)
        self._operation_statistics["trace_pairing_seconds"] = perf_counter() - started
        return result

    @cached_property
    def geometric_solution_count(self) -> int:
        """Number of geometric points over characteristic-zero or finite coefficient fields."""

        if not self.dimension:
            return 0
        if self.domain.characteristic():
            from .finite_decomposition import nilradical_basis

            return self.dimension - len(nilradical_basis(self))
        started = perf_counter()
        rank = self._trace_domain_pairing.rank()
        backend = "coefficient_domain"
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
        if self.domain.characteristic():
            return domain_matrix(result, self.domain).to_Matrix()
        return result

    def separator_matrix_columns(
        self, coefficients: Sequence[int | sp.Expr]
    ) -> tuple[tuple[sp.Expr, ...], ...]:
        """Return columns of a linear separator multiplication matrix.

        This direct form uses one Gröbner reduction per
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

        self._polynomial_parameter(parameter)
        if self.domain.characteristic():
            raise NotImplementedError(
                "finite-field products use Frobenius factors instead of linear separators"
            )
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
                packed_krylov_coefficients,
                preferred_matrix_storage,
            )

            if self.dimension >= 12 and preferred_matrix_storage(multiplication) == "sparse":
                try:
                    characteristic = sp.Poly.from_list(
                        list(
                            reversed(
                                packed_krylov_coefficients(
                                    domain_matrix(multiplication, self.domain), self.domain
                                )
                            )
                        ),
                        parameter,
                        domain=self.domain,
                    )
                except ValueError:
                    characteristic = self.characteristic_polynomial(linear_form, parameter)
            else:
                characteristic = self.characteristic_polynomial(linear_form, parameter)
            if characteristic.is_zero:
                continue
            squarefree = characteristic.sqf_part().monic()
            if squarefree.degree() != expected_distinct_roots:
                continue
            derivative = characteristic.diff()
            gcd = sp.gcd(characteristic, derivative)
            denominator = derivative.quo(gcd)
            native_action = self._linear_combination_domain_matrix(coefficients)
            vector = DomainMatrix({0: {0: self.domain.one}}, (self.dimension, 1), self.domain)
            powers = [self._matrix_expression(vector)]
            for _ in range(1, squarefree.degree()):
                vector = native_action * vector
                powers.append(self._matrix_expression(vector))
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
