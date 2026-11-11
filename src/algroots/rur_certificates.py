"""Reusable exact certificates for generic RURs and their regular fibers."""

from dataclasses import dataclass, replace

import sympy as sp
from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.rings import PolyRing

from .basis_certificates import GroebnerCertificate, _polynomial, certify_groebner_basis
from .domains import coefficient_domain, polynomial, polynomial_domain
from .groebner_errors import GroebnerLimitError, GroebnerVerificationError
from .native_f4 import NativeF4Limits
from .parametric_groebner import _guard, parametric_groebner_basis
from .quotient import QuotientAlgebra
from .rational_univariate.construction import rur_from_quotient


def _power_remainder(value, exponent, basis, evidence):
    """Reduce each product, keeping denominators of every division identity."""
    result = value.ring.one
    while exponent:
        if exponent & 1:
            quotients, result = (result * value).div(list(basis))
            evidence.extend(quotients)
            evidence.append(result)
        exponent //= 2
        if exponent:
            quotients, value = (value * value).div(list(basis))
            evidence.extend(quotients)
            evidence.append(value)
    return result


def _rur_evidence(certificate, representation):
    """Prove a bijection of geometric points using exact nilpotence identities.

    In a D-dimensional algebra every nilpotent has exponent at most D. Thus
    testing q(L)^D and (x_i-C_i(L))^D proves the map covers the whole reduced
    quotient, including nonradical inputs. The reverse map is checked modulo
    the squarefree q. No trace rank or numerical root count is trusted.
    """
    basis = certificate.basis
    variables, domain = basis.gens, basis.domain
    r = representation
    if (
        r.variables != variables
        or r.parameter in variables
        or r.defining_polynomial.domain != domain
        or len(r.coordinate_numerators) != len(variables)
    ):
        raise GroebnerVerificationError("RUR metadata does not match the certified algebra")
    quotient = QuotientAlgebra.from_groebner_basis(basis)
    if r.dimension != quotient.dimension or r.standard_exponents != quotient.standard_exponents:
        raise GroebnerVerificationError("RUR staircase or algebra dimension is incorrect")
    q = r.defining_polynomial
    if (
        q.gen != r.parameter
        or r.coordinate_denominator.gens != q.gens
        or any(
            p.gens != q.gens or p.domain != domain
            for p in (*r.coordinate_numerators, r.coordinate_denominator)
        )
    ):
        raise GroebnerVerificationError("RUR polynomial fields or variables do not match")
    if not quotient.dimension:
        if (
            q.as_expr() != 1
            or r.solution_count != 0
            or r.coordinate_denominator.as_expr() != 1
            or any(not p.is_zero for p in r.coordinate_numerators)
        ):
            raise GroebnerVerificationError("empty algebra requires the unit RUR")
        return [], sp.S.One
    if q.degree() < 1 or q.LC() != 1 or not q.is_sqf or r.solution_count != q.degree():
        raise GroebnerVerificationError("RUR defining polynomial is not monic and squarefree")
    t = r.parameter
    coefficient_parameters = tuple(domain.symbols) if domain.is_FractionField else ()
    if any(
        t.name == p.name
        for p in (
            *variables,
            *coefficient_parameters,
            *getattr(domain, "parameters", ()),
            *getattr(domain, "tower_symbols", ()),
        )
    ):
        raise GroebnerVerificationError(
            "RUR variable collides with a coordinate or coefficient generator"
        )
    arithmetic = polynomial_domain(domain)
    modulus = q.set_domain(arithmetic)
    denominator_inverse = r.coordinate_denominator.set_domain(arithmetic).invert(modulus)
    coordinates = tuple(
        (p.set_domain(arithmetic) * denominator_inverse).rem(modulus).set_domain(domain)
        for p in r.coordinate_numerators
    )
    cached_coordinates = r.__dict__.get("_normalized_coordinate_polynomials")
    if cached_coordinates is not None and cached_coordinates != coordinates:
        raise GroebnerVerificationError(
            "cached RUR coordinates disagree with the polynomial identities"
        )
    tring = PolyRing((t,), polynomial_domain(domain), "lex")
    q_native = _polynomial(q, tring)
    coordinate_map = dict(zip(variables, (p.as_expr() for p in coordinates), strict=True))
    for source in certificate.source:
        if _polynomial(source.subs(coordinate_map, simultaneous=True), tring).rem(q_native):
            raise GroebnerVerificationError("RUR coordinates fail an input equation")
    if _polynomial(r.separating_linear_form.subs(coordinate_map, simultaneous=True) - t, tring).rem(
        q_native
    ):
        raise GroebnerVerificationError("RUR parameter fails the inverse separator identity")
    ring = PolyRing(variables, polynomial_domain(domain), basis.order)
    native_basis = tuple(_polynomial(p, ring) for p in basis.polys)
    separator = polynomial(r.separating_linear_form, variables, domain)
    if separator.total_degree() > 1:
        raise GroebnerVerificationError("RUR separator must be linear")
    residuals = [q.as_expr().subs(t, separator.as_expr())]
    residuals.extend(
        v - c.as_expr().subs(t, separator.as_expr())
        for v, c in zip(variables, coordinates, strict=True)
    )
    evidence = []
    for residual in residuals:
        value = _polynomial(residual, ring)
        evidence.append(value)
        if _power_remainder(value, quotient.dimension, native_basis, evidence):
            raise GroebnerVerificationError("RUR does not cover every geometric quotient point")
    # Include both the normalized coordinates and the original denominator.
    # Otherwise a representation with a pole could be reused on a bad fiber.
    for poly in (q, r.coordinate_denominator, *r.coordinate_numerators, *coordinates):
        for coefficient in poly.rep.to_dict().values():
            evidence.append(ring.ground_new(domain.convert(coefficient)))
    # The resultant also preserves the denominator's invertibility: cancelled
    # coordinate factors can otherwise hide undefined fractions at some roots.
    return evidence, q.discriminant() * q.resultant(r.coordinate_denominator)


@dataclass(frozen=True)
class RationalUnivariateCertificate:
    """Exact RUR proof with a sufficient guard for algebra dimension and point count.

    The guard preserves coefficient identities, the basis staircase, and
    squarefreeness of the RUR polynomial. It says nothing about real signs.
    Exceptional fibers rebuild the original equations; input poles are errors.
    """

    basis_certificate: GroebnerCertificate
    representation: object
    parameters: tuple
    guard: sp.Expr
    input_guard: sp.Expr
    backend: str
    limits: object
    proof_limits: NativeF4Limits

    def __reduce__(self):
        r = self.representation
        state = (
            r.parameter,
            r.defining_polynomial.as_expr(),
            r.coordinate_denominator.as_expr(),
            tuple(p.as_expr() for p in r.coordinate_numerators),
            r.separating_linear_form,
            r.standard_exponents,
            r.quotient_dimension,
            r.geometric_solution_count,
        )
        return _restore_rur_certificate, (
            self.basis_certificate,
            state,
            self.parameters,
            self.guard,
            self.input_guard,
            self.backend,
            self.limits,
            self.proof_limits,
        )

    def _conditions(self):
        evidence, regularity = _rur_evidence(self.basis_certificate, self.representation)
        if sum(map(len, evidence)) > self.proof_limits.max_proof_nodes:
            raise GroebnerLimitError("RUR certificate witness-node budget exceeded")
        if not self.parameters:
            return sp.S.One, sp.S.One
        basis = self.basis_certificate.basis
        domain = basis.domain
        if not domain.is_FractionField or tuple(domain.symbols) != self.parameters:
            raise ValueError("RUR specialization parameters do not match the coefficient field")
        ground = domain.domain
        ring = PolyRing(basis.gens, polynomial_domain(domain), basis.order)
        original = tuple(_polynomial(p, ring) for p in self.basis_certificate.source)
        all_polys = list(original) + [_polynomial(p, ring) for p in basis.polys] + evidence
        for row in self.basis_certificate.memberships + self.basis_certificate.generator_reductions:
            all_polys.extend(_polynomial(p, ring) for p in row)
        for _, _, row in self.basis_certificate.pair_reductions:
            all_polys.extend(_polynomial(p, ring) for p in row)
        guard = _guard(all_polys, self.parameters, ground, self.proof_limits.max_terms)
        numerator, denominator = sp.together(regularity).as_numer_denom()
        # A squarefree generic polynomial may acquire repeated roots even when
        # the Gröbner staircase stays fixed (for example x^2-t at t=0).
        guard_poly = polynomial(guard * numerator * denominator, self.parameters, ground)
        if len(guard_poly.terms()) > self.proof_limits.max_terms:
            raise GroebnerLimitError("RUR specialization guard term budget exceeded")
        return guard_poly.sqf_part().monic().as_expr(), _guard(
            original, self.parameters, ground, self.proof_limits.max_terms
        )

    def verify(self):
        """Replay ideal equality, both geometric maps and independently derived guards."""
        try:
            return self.basis_certificate.verify() and self._conditions() == (
                self.guard,
                self.input_guard,
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

    def verified_representation(self):
        """Return the mathematical RUR after exact certificate replay."""
        if not self.verify():
            raise GroebnerVerificationError("RUR certificate failed verification")
        return self.representation

    def prepare(self):
        """Verify a detached snapshot once for repeated exact fiber specialization."""
        from .verified_families import VerifiedRationalUnivariateFamily

        return VerifiedRationalUnivariateFamily(self)

    def specialize_field(self, embedding):
        """Specialize a tower RUR through exact selected roots, rebuilding bad fibers."""
        from .algebraic_function_fields import FieldSpecialization

        self.verified_representation()
        if (
            not isinstance(embedding, FieldSpecialization)
            or embedding.source != self.basis_certificate.basis.domain
            or not embedding.verify()
        ):
            raise ValueError("a matching verified field specialization is required")
        r, target = self.representation, embedding.domain

        def mapped(expression, variables):
            p = polynomial(expression, variables, embedding.source)
            return sp.Poly.from_dict(
                {m: target.convert(embedding._evaluate(c)) for m, c in p.rep.to_dict().items()},
                *variables,
                domain=target,
            )

        source = tuple(mapped(p, r.variables).as_expr() for p in self.basis_certificate.source)
        try:
            proof = self.basis_certificate.specialization_certificate(embedding)
            candidate = replace(
                r,
                defining_polynomial=mapped(r.defining_polynomial, (r.parameter,)),
                coordinate_denominator=mapped(r.coordinate_denominator, (r.parameter,)),
                coordinate_numerators=tuple(
                    mapped(p, (r.parameter,)) for p in r.coordinate_numerators
                ),
                separating_linear_form=mapped(r.separating_linear_form, r.variables).as_expr(),
            )
            checked = RationalUnivariateCertificate(
                proof,
                candidate,
                (),
                sp.S.One,
                sp.S.One,
                self.backend,
                self.limits,
                self.proof_limits,
            )
            if checked.verify():
                return candidate
        except (ValueError, ZeroDivisionError, GroebnerVerificationError):
            pass
        quotient = QuotientAlgebra.from_polynomials(
            source,
            r.variables,
            domain=target,
            groebner_backend=self.backend,
            groebner_limits=self.limits,
        )
        return rur_from_quotient(quotient, r.parameter)

    def specialize(self, values):
        """Reuse regular fibers exactly; rebuild original equations at exceptional points."""
        self.verified_representation()
        return self._specialize_values(values)

    def _specialize_values(self, values):
        if getattr(
            self.representation.defining_polynomial.domain, "is_AlgebraicFunctionField", False
        ):
            raise ValueError("use specialize_field with explicit tower roots")
        if set(values) != set(self.parameters):
            raise ValueError("specialization requires a value for every parameter")
        values = {key: sp.sympify(value) for key, value in values.items()}
        if any(v.has(sp.Float) or v.is_algebraic is not True for v in values.values()):
            raise ValueError("specialization requires exact algebraic constants")
        r = self.representation
        domain = r.defining_polynomial.domain
        ground = domain.domain if domain.is_FractionField else domain
        extensions = tuple(ground.orig_ext) if ground.is_AlgebraicField else ()
        extensions += tuple(v for v in values.values() if not v.is_Rational)
        target = sp.QQ.algebraic_field(*extensions) if extensions else sp.QQ

        def subs(expr):
            return expr.subs(values, simultaneous=True)

        if not target.convert(subs(self.input_guard)):
            raise ValueError("specialization lies outside the input coefficient domain")
        if not target.convert(subs(self.guard)):
            quotient = QuotientAlgebra.from_polynomials(
                tuple(subs(p) for p in self.basis_certificate.source),
                r.variables,
                domain=target,
                groebner_backend=self.backend,
                groebner_limits=self.limits,
            )
            return rur_from_quotient(quotient, r.parameter)

        def poly(p):
            return polynomial(subs(p.as_expr()), (r.parameter,), target)

        return replace(
            r,
            defining_polynomial=poly(r.defining_polynomial),
            coordinate_denominator=poly(r.coordinate_denominator),
            coordinate_numerators=tuple(poly(p) for p in r.coordinate_numerators),
            separating_linear_form=subs(r.separating_linear_form),
        )


def _restore_rur_certificate(
    certificate, state, parameters, guard, input_guard, backend, limits, budget
):
    from .rational_univariate.representation import RationalUnivariateRepresentation

    t, q, den, numerators, separator, exponents, dimension, count = state
    domain = certificate.basis.domain
    representation = RationalUnivariateRepresentation(
        certificate.basis.gens,
        t,
        polynomial(q, (t,), domain),
        polynomial(den, (t,), domain),
        tuple(polynomial(p, (t,), domain) for p in numerators),
        separator,
        exponents,
        dimension,
        count,
    )
    return RationalUnivariateCertificate(
        certificate, representation, parameters, guard, input_guard, backend, limits, budget
    )


def certify_rational_univariate_representation(
    polynomials,
    variables,
    *,
    parameters=(),
    domain=None,
    candidate=None,
    basis_certificate=None,
    parameter=None,
    backend="native",
    limits=None,
    proof_limits=None,
):
    """Construct a reusable exact RUR proof, optionally checking a supplied candidate.

    ``parameters`` requests regular-fiber guards and exceptional recomputation.
    A supplied basis certificate is replayed and must describe the same system.
    """
    from ._validation import normalize_equation, validate_variables

    variables = validate_variables(variables)
    parameters = validate_variables(parameters) if parameters else ()
    source = tuple(normalize_equation(p) for p in polynomials)
    budget = NativeF4Limits() if proof_limits is None else proof_limits
    if not isinstance(budget, NativeF4Limits):
        raise TypeError("proof_limits must be NativeF4Limits")
    if basis_certificate is None:
        if parameters:
            generic = parametric_groebner_basis(
                source,
                variables,
                parameters,
                domain=domain,
                backend=backend,
                limits=limits,
                proof_limits=budget,
            )
            basis_certificate = GroebnerCertificate(
                generic.source,
                generic.basis,
                generic.memberships,
                generic.generator_reductions,
                generic.pair_reductions,
                budget,
            )
        else:
            basis_certificate = certify_groebner_basis(
                source,
                variables,
                domain=domain,
                backend=backend,
                limits=limits,
                proof_limits=budget,
            )
    basis = basis_certificate.verified_basis()
    requested_domain = coefficient_domain(domain) if domain is not None else None
    if requested_domain is not None and parameters and not requested_domain.is_FractionField:
        requested_domain = requested_domain.frac_field(*parameters)
    if (
        basis.gens != variables
        or (requested_domain is not None and basis.domain != requested_domain)
        or tuple(polynomial(p, variables, basis.domain) for p in source)
        != tuple(polynomial(p, variables, basis.domain) for p in basis_certificate.source)
    ):
        raise ValueError("basis certificate does not match the requested system")
    if not parameters and basis.domain.is_FractionField:
        parameters = tuple(basis.domain.symbols)
    if candidate is None:
        candidate = rur_from_quotient(
            QuotientAlgebra.from_certificate(basis_certificate), parameter
        )
    result = RationalUnivariateCertificate(
        basis_certificate, candidate, parameters, sp.S.One, sp.S.One, backend, limits, budget
    )
    guard, input_guard = result._conditions()
    result = replace(result, guard=guard, input_guard=input_guard)
    result.verified_representation()
    return result
