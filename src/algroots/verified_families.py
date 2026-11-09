"""Detached verified snapshots for repeated exact specialization."""

from dataclasses import replace

import sympy as sp
from sympy.polys.polyoptions import build_options
from sympy.polys.rings import PolyRing

from .basis_certificates import GroebnerCertificate, _polynomial, _restore_certificate
from .domains import polynomial, polynomial_domain


def _coefficient_expressions(polynomials, domain):
    return tuple(
        sorted(
            {domain.to_sympy(c) for p in polynomials for c in p.values()}, key=sp.default_sort_key
        )
    )


def _proof_polynomials(certificate):
    ring = PolyRing(
        certificate.basis.gens, polynomial_domain(certificate.basis.domain), certificate.basis.order
    )
    expressions = list(certificate.source) + list(certificate.basis.polys)
    expressions.extend(
        p for row in certificate.memberships + certificate.generator_reductions for p in row
    )
    expressions.extend(p for _, _, row in certificate.pair_reductions for p in row)
    return ring, tuple(_polynomial(p, ring) for p in expressions)


def _basis(expressions, variables, domain, order):
    return sp.GroebnerBasis._new(
        [polynomial(p, variables, domain) for p in expressions],
        build_options(variables, {"domain": domain, "order": order, "polys": False}),
    )


def _assignment(values, parameters, ground):
    if set(values) != set(parameters):
        raise ValueError("specialization requires a value for every parameter")
    values = {p: sp.sympify(v) for p, v in values.items()}
    if any(v.has(sp.Float) or v.is_algebraic is not True for v in values.values()):
        raise ValueError("specialization values must be exact algebraic constants")
    constants = tuple(ground.orig_ext) if ground.is_AlgebraicField else ()
    constants += tuple(v for v in values.values() if not v.is_Rational)
    return values, sp.QQ.algebraic_field(*constants) if constants else sp.QQ


class VerifiedGroebnerFamily:
    """A detached proof checked once for repeated basis reuse and specialization.

    Construction copies mathematical state and replays the proof. Subsequent
    uses rely on its immutable expressions and the specialization theorem;
    they do not trust mutable source certificates or serialized success flags.
    Unpickling constructs and verifies a new snapshot.
    """

    __slots__ = (
        "_state",
        "_source",
        "_variables",
        "_domain",
        "_order",
        "_basis",
        "_coefficients",
        "_guard",
        "_input_guard",
    )

    def __init__(self, certificate):
        if not isinstance(certificate, GroebnerCertificate):
            raise TypeError("a GroebnerCertificate is required")
        state = certificate.__reduce__()[1]
        snapshot = _restore_certificate(*state)
        snapshot.verified_basis()
        ring, polynomials = _proof_polynomials(snapshot)
        parameters = (
            tuple(snapshot.basis.domain.symbols) if snapshot.basis.domain.is_FractionField else ()
        )
        guard = input_guard = sp.S.One
        if parameters:
            from .parametric_groebner import _guard

            ground = snapshot.basis.domain.domain
            guard = _guard(polynomials, parameters, ground, snapshot.proof_limits.max_terms)
            input_guard = _guard(
                polynomials[: len(snapshot.source)],
                parameters,
                ground,
                snapshot.proof_limits.max_terms,
            )
        for key, value in dict(
            _state=state,
            _source=snapshot.source,
            _variables=snapshot.basis.gens,
            _domain=snapshot.basis.domain,
            _order=str(snapshot.basis.order),
            _basis=tuple(p.as_expr() for p in snapshot.basis.polys),
            _coefficients=_coefficient_expressions(polynomials, ring.domain),
            _guard=guard,
            _input_guard=input_guard,
        ).items():
            object.__setattr__(self, key, value)

    def __setattr__(self, name, value):
        raise AttributeError("verified families are immutable")

    def __reduce__(self):
        return _restore_verified_basis, (self._state,)

    @property
    def source(self):
        """The ordered immutable source expressions."""
        return self._source

    @property
    def guard(self):
        """Sufficient regularity guard for ordinary rational parameter fields."""
        return self._guard

    def verified_basis(self):
        """Materialize an independent mathematical basis without replaying identities."""
        from ._coefficient_state import restore_domain

        return _basis(self._basis, self._variables, restore_domain(self._state[3]), self._order)

    def _mapped(self, expression, embedding, variables):
        p = polynomial(expression, variables, self._domain)
        return sp.Poly.from_dict(
            {
                m: embedding.domain.convert(embedding._evaluate(c))
                for m, c in p.rep.to_dict().items()
            },
            *variables,
            domain=embedding.domain,
        )

    def _field_regular(self, embedding, coefficients=None):
        from .algebraic_function_fields import FieldSpecialization

        if (
            not isinstance(embedding, FieldSpecialization)
            or embedding.source != self._domain
            or not embedding.verify()
        ):
            raise ValueError("a matching verified field specialization is required")
        try:
            for c in self._coefficients if coefficients is None else coefficients:
                embedding._evaluate(c)
            return True
        except (ValueError, ZeroDivisionError):
            return False

    def specialize(self, values, *, backend="native", limits=None):
        """Reuse a regular fiber or recompute the original ideal at a bad proof point."""
        from .algebraic_function_fields import FieldSpecialization
        from .basis import _validate_backend, groebner_basis

        _validate_backend(backend, limits)
        if isinstance(values, FieldSpecialization):
            regular = self._field_regular(values)
            source = tuple(self._mapped(p, values, self._variables).as_expr() for p in self._source)
            if regular:
                return _basis(
                    tuple(self._mapped(p, values, self._variables) for p in self._basis),
                    self._variables,
                    values.domain,
                    self._order,
                )
            return groebner_basis(
                source,
                self._variables,
                domain=values.domain,
                order=self._order,
                backend=backend,
                limits=limits,
            )
        if getattr(self._domain, "is_AlgebraicFunctionField", False):
            raise ValueError("use a field specialization with explicit tower roots")
        parameters = tuple(self._domain.symbols) if self._domain.is_FractionField else ()
        if not parameters:
            if values:
                raise ValueError("this coefficient field has no specialization parameters")
            return self.verified_basis()
        ground = self._domain.domain
        if ground.characteristic():
            raise ValueError("parameter specialization requires characteristic-zero constants")
        assignment, target = _assignment(values, parameters, ground)

        def subs(p):
            return p.subs(assignment, simultaneous=True)

        if not target.convert(subs(self._input_guard)):
            raise ValueError("specialization lies outside the input coefficient domain")
        if target.convert(subs(self._guard)):
            return _basis(tuple(subs(p) for p in self._basis), self._variables, target, self._order)
        return groebner_basis(
            tuple(subs(p) for p in self._source),
            self._variables,
            domain=target,
            order=self._order,
            backend=backend,
            limits=limits,
        )


def _restore_verified_basis(state):
    return VerifiedGroebnerFamily(_restore_certificate(*state))


class VerifiedRationalUnivariateFamily:
    """An independently checked immutable RUR family for repeated fibers.

    Generic ideal and nilpotence proofs are replayed once. Specialization then
    checks their coefficient domain and the discriminant, and rebuilds original
    equations where those sufficient conditions fail.
    """

    __slots__ = ("_snapshot", "_basis", "_coefficients", "_regularity")

    def __init__(self, certificate):
        from .rur_certificates import (
            RationalUnivariateCertificate,
            _restore_rur_certificate,
            _rur_evidence,
        )

        if not isinstance(certificate, RationalUnivariateCertificate):
            raise TypeError("a RationalUnivariateCertificate is required")
        state = list(certificate.__reduce__()[1])
        state[0] = _restore_certificate(*certificate.basis_certificate.__reduce__()[1])
        snapshot = _restore_rur_certificate(*state)
        snapshot.verified_representation()
        basis = VerifiedGroebnerFamily(snapshot.basis_certificate)
        ring, polynomials = _proof_polynomials(snapshot.basis_certificate)
        evidence, regularity = _rur_evidence(snapshot.basis_certificate, snapshot.representation)
        object.__setattr__(self, "_snapshot", snapshot)
        object.__setattr__(self, "_basis", basis)
        object.__setattr__(
            self,
            "_coefficients",
            _coefficient_expressions(polynomials + tuple(evidence), ring.domain),
        )
        object.__setattr__(self, "_regularity", regularity)

    def __setattr__(self, name, value):
        raise AttributeError("verified families are immutable")

    def __reduce__(self):
        return VerifiedRationalUnivariateFamily, (self._snapshot,)

    def verified_representation(self):
        """Return an independent representation with no shared coordinate cache."""
        from .rur_certificates import _restore_rur_certificate

        state = list(self._snapshot.__reduce__()[1])
        state[0] = _restore_certificate(*self._basis._state)
        return _restore_rur_certificate(*state).representation

    def specialize(self, values):
        """Specialize rational parameters using already checked generic evidence."""
        from .rur_certificates import RationalUnivariateCertificate

        # The unbound helper's mathematical specialization body is also used
        # by ordinary certificates after full replay.
        return RationalUnivariateCertificate._specialize_values(self._snapshot, values)

    def specialize_field(self, embedding):
        """Specialize selected tower roots, recomputing collision or proof-pole fibers."""
        from .quotient import QuotientAlgebra
        from .rational_univariate.construction import rur_from_quotient

        regular = self._basis._field_regular(embedding, self._coefficients)
        r = self._snapshot.representation
        source = tuple(
            self._basis._mapped(p, embedding, r.variables).as_expr() for p in self._basis.source
        )
        if regular:
            try:
                regular = bool(embedding.domain.convert(embedding._evaluate(self._regularity)))
            except (ValueError, ZeroDivisionError):
                regular = False
        if regular:

            def mapped(p):
                return self._basis._mapped(p, embedding, (r.parameter,))

            return replace(
                r,
                defining_polynomial=mapped(r.defining_polynomial),
                coordinate_denominator=mapped(r.coordinate_denominator),
                coordinate_numerators=tuple(mapped(p) for p in r.coordinate_numerators),
                separating_linear_form=self._basis._mapped(
                    r.separating_linear_form, embedding, r.variables
                ).as_expr(),
            )
        quotient = QuotientAlgebra.from_polynomials(
            source,
            r.variables,
            domain=embedding.domain,
            groebner_backend=self._snapshot.backend,
            groebner_limits=self._snapshot.limits,
        )
        return rur_from_quotient(quotient, r.parameter)
