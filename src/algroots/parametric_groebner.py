"""Exact specialization guards and checked generic parameter-field bases."""

from dataclasses import dataclass

import sympy as sp
from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.polyoptions import build_options
from sympy.polys.rings import PolyRing

from ._validation import normalize_equation, validate_variables
from .basis import groebner_basis
from .basis_certificates import GroebnerCertificate, _identities, _polynomial, _proof_data
from .domains import (
    coefficient_domain,
    field_element,
    infer_coefficient_field,
    polynomial,
    polynomial_domain,
)
from .groebner_errors import GroebnerLimitError, GroebnerVerificationError
from .native_f4 import NativeF4Limits


def _guard(polynomials, parameters, ground, max_terms):
    """Include coefficient poles of every identity used in the specialization proof."""
    denominators = set()
    for poly in polynomials:
        for coefficient in poly.values():
            denominator = coefficient.denom.as_expr()
            if denominator.free_symbols.intersection(parameters):
                denominators.add(denominator)
    guard = sp.Poly(1, *parameters, domain=ground)
    for denominator in sorted(denominators, key=sp.default_sort_key):
        guard *= polynomial(denominator, parameters, ground)
        if len(guard.terms()) > max_terms:
            raise GroebnerLimitError("specialization guard term budget exceeded")
    return guard.sqf_part().monic().as_expr()


@dataclass(frozen=True)
class GroebnerSpecialization:
    """A generic basis and exact conditions under which it specializes correctly.

    Generator-combination identities, reverse membership, and standard S-pair
    representations establish specialization on the open guard. Exceptional
    points are recomputed from the original generators. Conditions describe
    coefficient poles and basis stability, not root-count or sign invariance.
    """

    source: tuple
    variables: tuple
    parameters: tuple
    domain: object
    basis: object
    memberships: tuple
    generator_reductions: tuple
    pair_reductions: tuple
    guard: object
    input_guard: object
    backend: str
    limits: object
    proof_limits: NativeF4Limits

    def __reduce__(self):
        certificate = GroebnerCertificate(
            self.source,
            self.basis,
            self.memberships,
            self.generator_reductions,
            self.pair_reductions,
            self.proof_limits,
        )
        return _restore_specialization, (
            certificate,
            self.parameters,
            self.guard,
            self.input_guard,
            self.backend,
            self.limits,
            self.proof_limits,
        )

    @property
    def generic_condition(self):
        """Parameter condition on which the stored basis remains a reduced basis."""
        return sp.Ne(self.guard, 0)

    @property
    def exceptional_condition(self):
        """Defined input points requiring recomputation or separate real elimination."""
        return sp.And(sp.Ne(self.input_guard, 0), sp.Eq(self.guard, 0))

    def _data(self):
        ring = PolyRing(self.variables, polynomial_domain(self.domain), self.basis.order)
        source = tuple(_polynomial(p, ring) for p in self.source)
        basis = tuple(_polynomial(p, ring) for p in self.basis.polys)
        memberships = tuple(tuple(_polynomial(p, ring) for p in row) for row in self.memberships)
        reductions = tuple(
            tuple(_polynomial(p, ring) for p in row) for row in self.generator_reductions
        )
        pairs = tuple(
            (i, j, tuple(_polynomial(p, ring) for p in row)) for i, j, row in self.pair_reductions
        )
        return ring, source, basis, memberships, reductions, pairs

    def verify(self):
        """Replay all identities and derive the guard without trusting stored conditions."""
        try:
            if (
                not self.domain.is_FractionField
                or tuple(self.domain.symbols) != self.parameters
                or self.basis.domain != self.domain
                or self.basis.gens != self.variables
            ):
                return False
            if self.domain.domain != sp.QQ and not self.domain.domain.is_AlgebraicField:
                return False
            if str(self.basis.order) not in ("lex", "grlex", "grevlex"):
                return False
            ring, source, basis, memberships, reductions, pairs = self._data()
            if not _identities(source, basis, ring, memberships, reductions, pairs):
                return False
            all_polys = (
                source
                + basis
                + tuple(p for row in memberships + reductions for p in row)
                + tuple(p for _, _, row in pairs for p in row)
            )
            if sum(len(p) for p in all_polys) > self.proof_limits.max_proof_nodes:
                return False
            ground = self.domain.domain
            return (
                _guard(source, self.parameters, ground, self.proof_limits.max_terms)
                == self.input_guard
                and _guard(all_polys, self.parameters, ground, self.proof_limits.max_terms)
                == self.guard
            )
        except (
            ValueError,
            TypeError,
            AttributeError,
            ArithmeticError,
            sp.PolynomialError,
            CoercionFailed,
        ):
            return False

    def as_formula(self):
        """Return equivalent real equality branches, preserving the exceptional locus."""
        if not self.verify():
            raise GroebnerVerificationError("specialization proof failed verification")
        original = sp.And(*(sp.Eq(sp.together(p).as_numer_denom()[0], 0) for p in self.source))
        generic = sp.And(
            *(sp.Eq(sp.together(p.as_expr()).as_numer_denom()[0], 0) for p in self.basis.polys)
        )
        return sp.Or(
            sp.And(self.generic_condition, generic), sp.And(self.exceptional_condition, original)
        )

    def prepare(self):
        """Return a detached verified basis family for repeated parameter values."""
        if not self.verify():
            raise GroebnerVerificationError("specialization proof failed verification")
        return GroebnerCertificate(
            self.source,
            self.basis,
            self.memberships,
            self.generator_reductions,
            self.pair_reductions,
            self.proof_limits,
        ).prepare()

    def specialize(self, values):
        """Specialize exactly, recomputing original generators on exceptional points."""
        if not self.verify():
            raise GroebnerVerificationError("specialization proof failed verification")
        if set(values) != set(self.parameters):
            raise ValueError("specialization requires a value for every parameter")
        assignment = {key: sp.sympify(value) for key, value in values.items()}
        if any(
            value.has(sp.Float) or value.is_algebraic is not True for value in assignment.values()
        ):
            raise ValueError("specialization values must be exact algebraic constants")
        ground = self.domain.domain
        extensions = tuple(ground.orig_ext) if ground.is_AlgebraicField else ()
        extensions += tuple(value for value in assignment.values() if not value.is_Rational)
        target = sp.QQ.algebraic_field(*extensions) if extensions else sp.QQ

        def substitute(expression):
            return expression.subs(assignment, simultaneous=True)

        if not field_element(substitute(self.input_guard), target):
            raise ValueError("specialization lies outside the input coefficient domain")
        if not field_element(substitute(self.guard), target):
            return groebner_basis(
                tuple(substitute(p) for p in self.source),
                self.variables,
                domain=target,
                order=str(self.basis.order),
                backend=self.backend,
                limits=self.limits,
            )
        polynomials = [
            polynomial(substitute(p.as_expr()), self.variables, target) for p in self.basis.polys
        ]
        options = build_options(
            self.variables, {"domain": target, "order": str(self.basis.order), "polys": False}
        )
        return sp.GroebnerBasis._new(polynomials, options)


def _restore_specialization(certificate, parameters, guard, input_guard, backend, limits, budget):
    return GroebnerSpecialization(
        certificate.source,
        certificate.basis.gens,
        parameters,
        certificate.basis.domain,
        certificate.basis,
        certificate.memberships,
        certificate.generator_reductions,
        certificate.pair_reductions,
        guard,
        input_guard,
        backend,
        limits,
        budget,
    )


def parametric_groebner_basis(
    polynomials,
    variables,
    parameters,
    *,
    domain=None,
    order="grevlex",
    backend="native",
    limits=None,
    proof_limits=None,
    execution=None,
):
    """Return a checked generic basis with specialization conditions and witnesses.

    ``domain`` is QQ, a number field, or that field extended by the listed
    parameters. Generic field arithmetic does not remove exceptional fibers:
    ``as_formula()`` retains their original equations, and ``specialize``
    recomputes them. ``proof_limits`` uses NativeF4Limits' pair, basis, term,
    and witness-node budgets independently of the selected candidate backend.
    """
    if execution is not None:
        from .execution import run_supervised

        return run_supervised(
            parametric_groebner_basis,
            polynomials,
            variables,
            parameters,
            domain=domain,
            order=order,
            backend=backend,
            limits=limits,
            proof_limits=proof_limits,
            execution=execution,
        )
    variables = validate_variables(variables)
    parameters = validate_variables(parameters)
    if set(variables) & set(parameters):
        raise ValueError("parameters and polynomial variables must be disjoint")
    source = tuple(normalize_equation(p) for p in polynomials)
    if domain is None:
        ground = infer_coefficient_field(source, variables + parameters)
    else:
        domain = coefficient_domain(domain)
        if domain.is_FractionField:
            if tuple(domain.symbols) != parameters:
                raise ValueError("domain parameters must match the supplied ordered parameters")
            ground = domain.domain
        else:
            ground = domain
    if ground != sp.QQ and not ground.is_AlgebraicField:
        raise ValueError("specialization guards require QQ or an algebraic number constant field")
    domain = ground.frac_field(*parameters)
    source = tuple(polynomial(p, variables, domain).as_expr() for p in source)
    budget = (
        proof_limits
        if proof_limits is not None
        else limits
        if isinstance(limits, NativeF4Limits)
        else NativeF4Limits()
    )
    if not isinstance(budget, NativeF4Limits):
        raise TypeError("proof_limits must be NativeF4Limits")
    basis = groebner_basis(
        source, variables, domain=domain, order=order, backend=backend, limits=limits
    )
    ring = PolyRing(variables, polynomial_domain(domain), order)
    original = tuple(_polynomial(p, ring) for p in source)
    candidate = tuple(_polynomial(p, ring) for p in basis.polys)
    memberships, reductions, pairs = _proof_data(source, basis, budget)
    all_polys = (
        original
        + candidate
        + tuple(p for row in memberships + tuple(reductions) for p in row)
        + tuple(p for _, _, row in pairs for p in row)
    )
    if sum(len(p) for p in all_polys) > budget.max_proof_nodes:
        raise GroebnerLimitError("specialization witness-node budget exceeded")
    guard = _guard(all_polys, parameters, ground, budget.max_terms)
    input_guard = _guard(original, parameters, ground, budget.max_terms)
    result = GroebnerSpecialization(
        source,
        variables,
        parameters,
        domain,
        basis,
        tuple(tuple(p.as_expr() for p in row) for row in memberships),
        tuple(tuple(p.as_expr() for p in row) for row in reductions),
        tuple((i, j, tuple(p.as_expr() for p in row)) for i, j, row in pairs),
        guard,
        input_guard,
        backend,
        limits,
        budget,
    )
    if not result.verify():
        raise GroebnerVerificationError("specialization proof failed independent replay")
    return result
