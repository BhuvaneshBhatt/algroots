"""Universal monic-factor algebras and automatic coefficient-field discovery.

A degree-m factor of a separable degree-n polynomial is determined by its m
coefficients. The remainder equations describe exactly binomial(n,m) geometric
points. A cyclic presentation and its irreducible factors enumerate their
Galois orbits, providing the required coefficient extensions without building
an entire splitting field. All base changes and coverage claims are replayable.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb

import sympy as sp
from sympy.polys.polyerrors import BasePolynomialError

from .curves import CurveError
from .errors import QuotientAlgebraError
from .function_fields import _constant_domain
from .quotient import QuotientAlgebra
from .subfields import field_normal_form


def _monic(relation):
    x, z = relation.gens
    return sp.Poly(relation.as_expr(), x, domain=relation.domain.frac_field(z)).monic()


def _fresh(label, forbidden):
    names = {s.name for s in forbidden}
    while label in names:
        label = "_" + label
    return sp.Dummy(label)


def _equations(relation, degree, coefficients):
    x, z = relation.gens
    base = relation.domain.frac_field(z)
    ring = base.poly_ring(*coefficients)
    generic = x**degree + sum(c * x ** (degree - i - 1) for i, c in enumerate(coefficients))
    # auto=True tries to construct a nested fraction field; the divisor is
    # monic, so exact polynomial-ring division suffices and avoids that coercion.
    remainder = sp.Poly(_monic(relation).as_expr(), x, domain=ring).rem(
        sp.Poly(generic, x, domain=ring),
        auto=False,
    )
    return tuple(remainder.all_coeffs())


def _evaluate_vector(polynomial, matrix, one):
    result = one.scalarmul(matrix.domain.zero)
    for coefficient in polynomial.all_coeffs():
        result = matrix * result + one.scalarmul(matrix.domain.convert(coefficient))
    return result


def _divide(relation, factor, ambient):
    x = relation.gens[0]
    left = list(_monic(relation).all_coeffs())
    right = sp.Poly(factor, x).all_coeffs()
    quotient = []
    while len(left) >= len(right):
        lead = field_normal_form(ambient, left[0])
        quotient.append(lead)
        for i, c in enumerate(right):
            left[i] = field_normal_form(ambient, left[i] - lead * c)
        left.pop(0)
    expression = sum(c * x ** (len(quotient) - i - 1) for i, c in enumerate(quotient))
    return sp.expand(expression), tuple(field_normal_form(ambient, c) for c in left)


@dataclass(frozen=True)
class DiscoveredAlgebraicFactor:
    """One monic-factor orbit, in its explicitly constructed coefficient field."""

    relation: sp.Poly
    degree: int
    ambient: QuotientAlgebra
    coefficients: tuple[sp.Expr, ...]
    cofactor: sp.Expr

    @property
    def factor(self):
        x = self.relation.gens[0]
        return x**self.degree + sum(
            c * x ** (self.degree - i - 1) for i, c in enumerate(self.coefficients)
        )

    @property
    def extension_degree(self):
        return self.ambient.dimension

    def verify(self):
        from .functional_decomposition import _input_relation, _integer, _symbols

        try:
            x, z = self.relation.gens
            _integer(self.degree, "degree", 1)
            if self.degree > self.relation.degree(x) or len(self.coefficients) != self.degree:
                return False
            if (
                self.ambient.domain != self.relation.domain.frac_field(z)
                or not self.ambient.is_field
            ):
                return False
            _symbols((x, z, *self.ambient.variables))
            if (
                _input_relation(self.relation.as_expr(), (x, z), self.relation.domain)
                != self.relation
            ):
                return False
            if self.coefficients != tuple(
                field_normal_form(self.ambient, c) for c in self.coefficients
            ):
                return False
            quotient, remainder = _divide(self.relation, self.factor, self.ambient)
            return not any(remainder) and quotient == self.cofactor
        except (ValueError, TypeError, AttributeError, BasePolynomialError, ZeroDivisionError):
            return False


@dataclass(frozen=True)
class AlgebraicFactorDegree:
    """An exhaustive, cyclic presentation of all factors of one fixed degree."""

    relation: sp.Poly
    degree: int
    quotient: QuotientAlgebra
    parameter: sp.Symbol
    weights: tuple[int, ...]
    defining_polynomial: sp.Poly
    coordinate_polynomials: tuple[sp.Poly, ...]
    component_polynomials: tuple[sp.Poly, ...]
    orbits: tuple[DiscoveredAlgebraicFactor, ...]

    @property
    def geometric_factor_count(self):
        return comb(self.relation.degree(self.relation.gens[0]), self.degree)

    def verify(self):
        """Replay universal equations, inverse maps and irreducible orbit coverage."""
        from .functional_decomposition import _integer, _symbols

        try:
            x, z = self.relation.gens
            A = self.quotient
            base = self.relation.domain.frac_field(z)
            _integer(self.degree, "degree", 1)
            if not 1 <= self.degree <= self.relation.degree(x) or len(A.variables) != self.degree:
                return False
            _symbols((x, z, self.parameter, *A.variables))
            expected = self.geometric_factor_count
            if A.domain != base or A.dimension != expected:
                return False
            rebuilt = QuotientAlgebra.from_polynomials(
                _equations(self.relation, self.degree, A.variables),
                A.variables,
                domain=base,
            )
            if (
                A.groebner_basis != rebuilt.groebner_basis
                or A.standard_exponents != rebuilt.standard_exponents
            ):
                return False
            if len(self.weights) != self.degree or any(
                isinstance(w, bool) or not isinstance(w, int) for w in self.weights
            ):
                return False
            p = self.defining_polynomial
            if (
                p.gens != (self.parameter,)
                or p.domain != base
                or p.degree() != expected
                or p.LC() != 1
                or p.gcd(p.diff()).degree() != 0
            ):
                return False
            if len(self.coordinate_polynomials) != self.degree:
                return False
            if any(
                q.gens != p.gens or q.domain != base or q.degree() >= expected
                for q in self.coordinate_polynomials
            ):
                return False
            linear = sum(w * c for w, c in zip(self.weights, A.variables, strict=True))
            matrix = A.multiplication_matrix(linear, representation="domain")
            one = A.coordinate_vector(1).to_DM(domain=base)
            if not _evaluate_vector(p, matrix, one).is_zero_matrix:
                return False
            if any(
                _evaluate_vector(q, matrix, one).to_Matrix() != A.coordinate_vector(c)
                for q, c in zip(self.coordinate_polynomials, A.variables, strict=True)
            ):
                return False
            images = tuple(q.as_expr() for q in self.coordinate_polynomials)
            if (
                not sp.Poly(
                    sum(w * q for w, q in zip(self.weights, images, strict=True)) - self.parameter,
                    self.parameter,
                    domain=base,
                )
                .rem(p)
                .is_zero
            ):
                return False
            # p(L)=0 and q_i(L)=c_i make K(z)[T]/p -> A surjective.
            # Equal dimensions prove it is an isomorphism; this also certifies
            # all defining equations in the inverse coordinate map without
            # expanding large rational substitutions into those equations.
            factors = self.component_polynomials
            if not factors or len(factors) != len(self.orbits):
                return False
            if any(
                f.gens != p.gens or f.domain != base or f.LC() != 1 or not f.is_irreducible
                for f in factors
            ):
                return False
            if sp.Poly(sp.prod(f.as_expr() for f in factors), self.parameter, domain=base) != p:
                return False
            for f, orbit in zip(factors, self.orbits, strict=True):
                C = orbit.ambient
                if (
                    orbit.relation != self.relation
                    or orbit.degree != self.degree
                    or C.variables != (self.parameter,)
                    or C.domain != base
                ):
                    return False
                expected_field = QuotientAlgebra.from_polynomials(
                    [f.as_expr()], (self.parameter,), domain=base
                )
                if (
                    C.groebner_basis != expected_field.groebner_basis
                    or C.standard_exponents != expected_field.standard_exponents
                ):
                    return False
                if (
                    orbit.coefficients != tuple(C.normal_form(q) for q in images)
                    or not orbit.verify()
                ):
                    return False
            return True
        except (ValueError, TypeError, AttributeError, BasePolynomialError, ZeroDivisionError):
            return False


@dataclass(frozen=True)
class AlgebraicDiscoveryIssue:
    """An uncompleted factor degree, never a negative decomposition proof."""

    degree: int
    reason: str


@dataclass(frozen=True)
class AlgebraicFactorDiscovery:
    """Factor extensions for requested degrees, grouped by Galois orbit."""

    relation: sp.Poly
    requested_degrees: tuple[int, ...]
    levels: tuple[AlgebraicFactorDegree, ...]
    issues: tuple[AlgebraicDiscoveryIssue, ...]
    max_factor_dimension: int
    max_separating_attempts: int

    @property
    def orbits(self):
        return tuple(orbit for level in self.levels for orbit in level.orbits)

    @property
    def complete(self):
        """Coverage of requested factor degrees, including every conjugate orbit."""
        return (
            not self.issues
            and tuple(level.degree for level in self.levels) == self.requested_degrees
        )

    def verify(self):
        from .functional_decomposition import _input_relation, _integer

        try:
            x, z = self.relation.gens
            if (
                _input_relation(self.relation.as_expr(), (x, z), self.relation.domain)
                != self.relation
            ):
                return False
            n = self.relation.degree(x)
            if _degrees(self.requested_degrees, n) != self.requested_degrees:
                return False
            _integer(self.max_factor_dimension, "max_factor_dimension", 1)
            _integer(self.max_separating_attempts, "max_separating_attempts", 1)
            done = tuple(level.degree for level in self.levels)
            missing = tuple(i.degree for i in self.issues)
            if len(set((*done, *missing))) != len(done) + len(missing) or set(
                (*done, *missing)
            ) != set(self.requested_degrees):
                return False
            if done != tuple(d for d in self.requested_degrees if d in done) or missing != tuple(
                d for d in self.requested_degrees if d in missing
            ):
                return False
            if any(
                level.relation != self.relation
                or level.geometric_factor_count > self.max_factor_dimension
                or not any(
                    level.weights == tuple(j**i for i in range(level.degree))
                    for j in range(1, self.max_separating_attempts + 1)
                )
                or not level.verify()
                for level in self.levels
            ):
                return False
            for issue in self.issues:
                _integer(issue.degree, "issue degree", 1)
                if issue.reason == "factor_dimension_limit":
                    if comb(n, issue.degree) <= self.max_factor_dimension:
                        return False
                elif issue.reason != "separator_limit":
                    return False
                # A separator-limit issue certifies only uncompleted coverage;
                # no failure of a chosen search heuristic is a mathematical proof.
            return True
        except (ValueError, TypeError, AttributeError, BasePolynomialError, ZeroDivisionError):
            return False


def _degrees(degrees, n):
    from .functional_decomposition import FunctionalDecompositionError, _integer

    values = tuple(range(1, n + 1)) if degrees is None else tuple(degrees)
    for d in values:
        _integer(d, "factor degree", 1)
        if d > n:
            raise FunctionalDecompositionError("factor degree exceeds the polynomial degree")
    if tuple(sorted(set(values))) != values:
        raise FunctionalDecompositionError("factor degrees must be distinct and increasing")
    return values


def _level(relation, degree, max_dimension, max_attempts):
    x, z = relation.gens
    base = relation.domain.frac_field(z)
    coefficients = tuple(_fresh(f"factor_coefficient_{i}", (x, z)) for i in range(degree))
    parameter = _fresh("factor_extension", (x, z, *coefficients))
    A = QuotientAlgebra.from_polynomials(
        _equations(relation, degree, coefficients),
        coefficients,
        domain=base,
        max_dimension=max_dimension,
    )
    dimension = comb(relation.degree(x), degree)
    if A.dimension != dimension:
        raise ArithmeticError("universal monic-factor algebra has an incorrect dimension")
    for attempt in range(1, max_attempts + 1):
        weights = tuple(attempt**i for i in range(degree))
        linear = sum(w * c for w, c in zip(weights, coefficients, strict=True))
        p = A.minimal_polynomial(linear, parameter)
        if p.degree() == dimension and p.gcd(p.diff()).degree() == 0:
            break
    else:
        return None
    matrix = A.multiplication_matrix(linear, representation="domain")
    vector = A.coordinate_vector(1).to_DM(domain=base)
    powers = []
    for _ in range(dimension):
        powers.append(vector)
        vector = matrix * vector
    change = powers[0].hstack(*powers[1:])
    targets = tuple(A.coordinate_vector(c).to_DM(domain=base) for c in coefficients)
    coordinates = (change.inv() * targets[0].hstack(*targets[1:])).to_list()
    images = tuple(
        sp.Poly.from_list(
            [base.to_sympy(coordinates[j][i]) for j in range(dimension - 1, -1, -1)],
            parameter,
            domain=base,
        )
        for i in range(degree)
    )
    _, factorization = p.factor_list()
    if any(m != 1 for _, m in factorization):
        raise ArithmeticError("cyclic factor-algebra polynomial did not split squarefreely")
    factors = tuple(
        sorted(
            (f.monic() for f, _ in factorization),
            key=lambda f: (f.degree(), sp.default_sort_key(f.as_expr())),
        )
    )
    if sp.Poly(sp.prod(f.as_expr() for f in factors), parameter, domain=base) != p:
        raise ArithmeticError("normalized extension factors did not reconstruct their polynomial")
    orbits = []
    for f in factors:
        B = QuotientAlgebra.from_polynomials([f.as_expr()], (parameter,), domain=base)
        values = tuple(B.normal_form(q.as_expr()) for q in images)
        factor = x**degree + sum(c * x ** (degree - i - 1) for i, c in enumerate(values))
        cofactor, remainder = _divide(relation, factor, B)
        if any(remainder):
            raise ArithmeticError("discovered factor failed exact polynomial division")
        orbits.append(DiscoveredAlgebraicFactor(relation, degree, B, values, cofactor))
    return AlgebraicFactorDegree(
        relation, degree, A, parameter, weights, p, images, factors, tuple(orbits)
    )


def discover_algebraic_factors(
    relation,
    variables,
    *,
    degrees=None,
    constants=sp.QQ,
    max_degree=6,
    max_factor_dimension=32,
    max_separating_attempts=64,
):
    """Construct coefficient extensions and all monic factor orbits of requested degrees.

    Completeness is over constants(input), up to its Galois action, not numerical
    root selection. Limits bound input degree, factor-algebra dimension and the
    search for a separating form; they are not wall-clock limits.
    """
    # The public facade re-exports these result types. Lazy helper imports keep
    # both import orders valid, including direct internal-module inspection.
    from .functional_decomposition import (
        DecompositionLimitError,
        _input_relation,
        _integer,
        _symbols,
    )

    x, z = tuple(variables)
    _symbols((x, z))
    constants = _constant_domain(constants)
    _integer(max_degree, "max_degree", 1)
    _integer(max_factor_dimension, "max_factor_dimension", 1)
    _integer(max_separating_attempts, "max_separating_attempts", 1)
    F = _input_relation(relation, (x, z), constants)
    n = F.degree(x)
    if n > max_degree:
        raise DecompositionLimitError(f"output degree {n} exceeds max_degree={max_degree}")
    requested = _degrees(degrees, n)
    levels, issues = [], []
    for degree in requested:
        if comb(n, degree) > max_factor_dimension:
            issues.append(AlgebraicDiscoveryIssue(degree, "factor_dimension_limit"))
            continue
        level = _level(F, degree, max_factor_dimension, max_separating_attempts)
        if level is None:
            issues.append(AlgebraicDiscoveryIssue(degree, "separator_limit"))
        else:
            levels.append(level)
    return AlgebraicFactorDiscovery(
        F, requested, tuple(levels), tuple(issues), max_factor_dimension, max_separating_attempts
    )


@dataclass(frozen=True)
class AlgebraicCandidateAttempt:
    """A reconstruction result or an explicitly unresolved candidate diagnostic."""

    index: int
    result: object | None
    error: str | None = None


@dataclass(frozen=True)
class AlgebraicDecompositionSearch:
    """Automatic minimal coefficient-field search, with exact coverage accounting."""

    discovery: AlgebraicFactorDiscovery
    parameter: sp.Symbol
    attempts: tuple[AlgebraicCandidateAttempt, ...]

    @property
    def relation(self):
        return self.discovery.relation

    @property
    def candidate_count(self):
        return len(self.discovery.orbits)

    @property
    def examined(self):
        return len(self.attempts)

    @property
    def decompositions(self):
        return tuple(
            a.result.decomposition
            for a in self.attempts
            if a.result is not None and a.result.decomposition is not None
        )

    @property
    def complete(self):
        n = self.relation.degree(self.relation.gens[0])
        return (
            self.discovery.complete
            and self.discovery.requested_degrees == tuple(range(2, n + 1))
            and self.examined == self.candidate_count
            and all(a.result is not None for a in self.attempts)
        )

    @property
    def status(self):
        if self.decompositions:
            return "decomposable"
        if not self.complete:
            return "incomplete"
        if self.relation.degree(self.relation.gens[0]) < 2:
            return "degree_below_two"
        return "indecomposable"

    def verify(self):
        """Replay orbit coverage and every claimed answer/rejection.

        Unresolved diagnostic text is not a proof of unsupportedness; it always
        prevents completeness and an indecomposability claim.
        """
        from .functional_decomposition import _candidate, _symbols

        try:
            if not self.discovery.verify() or self.examined > self.candidate_count:
                return False
            _symbols((*self.relation.gens, self.parameter))
            if tuple(a.index for a in self.attempts) != tuple(range(self.examined)):
                return False
            for attempt, orbit in zip(self.attempts, self.discovery.orbits, strict=False):
                if attempt.result is None:
                    if not isinstance(attempt.error, str) or not attempt.error:
                        return False
                    continue
                r = attempt.result
                if (
                    attempt.error is not None
                    or r.relation != self.relation
                    or r.ambient is not orbit.ambient
                    or r.candidate != _candidate(orbit.factor, self.relation.gens[0], orbit.ambient)
                    or r.parameter != self.parameter
                    or not r.verify()
                ):
                    return False
            return True
        except (ValueError, TypeError, AttributeError, BasePolynomialError, ZeroDivisionError):
            return False


def search_algebraic_decomposition(
    relation,
    variables,
    *,
    parameter,
    constants,
    factor_degrees,
    max_degree,
    max_factor_dimension,
    max_separating_attempts,
    max_candidates,
    exhaustive,
    max_steps,
    max_dimension,
):
    from .functional_decomposition import (
        FunctionalDecompositionError,
        _input_relation,
        _integer,
        _symbols,
        algebraic_decompose,
    )

    _integer(max_candidates, "max_candidates", 0)
    _integer(max_steps, "max_steps", 1)
    _integer(max_dimension, "max_dimension", 1)
    if not isinstance(exhaustive, bool):
        raise FunctionalDecompositionError("exhaustive must be a Boolean")
    x, z = tuple(variables)
    _symbols((x, z, parameter))
    F = _input_relation(relation, (x, z), constants)
    degrees = tuple(range(2, F.degree(x) + 1)) if factor_degrees is None else tuple(factor_degrees)
    if any(d == 1 for d in degrees):
        raise FunctionalDecompositionError(
            "degree-one factors only give trivial minimal outer components"
        )
    discovery = discover_algebraic_factors(
        F,
        (x, z),
        degrees=degrees,
        constants=constants,
        max_degree=max_degree,
        max_factor_dimension=max_factor_dimension,
        max_separating_attempts=max_separating_attempts,
    )
    attempts = []
    for index, orbit in enumerate(discovery.orbits):
        if index >= max_candidates:
            break
        try:
            result = algebraic_decompose(
                F,
                (x, z),
                candidate=orbit.factor,
                ambient=orbit.ambient,
                parameter=parameter,
                constants=constants,
                max_steps=max_steps,
                max_dimension=max_dimension,
            )
        except (CurveError, QuotientAlgebraError, NotImplementedError) as exc:
            attempts.append(AlgebraicCandidateAttempt(index, None, f"{type(exc).__name__}: {exc}"))
            continue
        attempts.append(AlgebraicCandidateAttempt(index, result))
        if result.decomposition is not None and not exhaustive:
            break
    return AlgebraicDecompositionSearch(discovery, parameter, tuple(attempts))
