"""Reusable exact ideal-equality and Buchberger certificates."""

from dataclasses import dataclass
from itertools import combinations

import sympy as sp
from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.rings import PolyRing

from .domains import coefficient_domain, polynomial, polynomial_domain
from .groebner_errors import GroebnerLimitError, GroebnerVerificationError
from .groebner_proofs import construct_membership_witnesses
from .native_f4 import NativeF4Limits


def _polynomial(expression, ring):
    return ring.from_dict(polynomial(expression, ring.symbols, ring.domain).rep.to_dict())


def _spoly(left, right):
    a, b = tuple(left.LM), tuple(right.LM)
    common = tuple(max(x, y) for x, y in zip(a, b, strict=True))
    ring = left.ring
    first = ring.from_dict({tuple(m - e for m, e in zip(common, a, strict=True)): ring.domain.one})
    second = ring.from_dict({tuple(m - e for m, e in zip(common, b, strict=True)): ring.domain.one})
    return first * left - second * right, common


def _divide(target, basis):
    return (
        target.div(list(basis)) if target else ([target.ring.zero] * len(basis), target.ring.zero)
    )


def _identities(source, basis, ring, memberships, reductions, pairs):
    """Check both ideal inclusions and strict standard S-pair representations."""
    if any(not g or ring.domain.one != g.LC for g in basis):
        return False
    for index, g in enumerate(basis):
        if any(
            all(a <= b for a, b in zip(other.LM, monomial, strict=True))
            for j, other in enumerate(basis)
            if j != index
            for monomial in g
        ):
            return False
    if len(memberships) != len(basis) or len(reductions) != len(source):
        return False
    for targets, rows, generators in ((basis, memberships, source), (source, reductions, basis)):
        for target, row in zip(targets, rows, strict=True):
            if (
                len(row) != len(generators)
                or sum((a * b for a, b in zip(row, generators, strict=True)), ring.zero) != target
            ):
                return False
    expected = set(combinations(range(len(basis)), 2))
    if {(i, j) for i, j, _ in pairs} != expected or len(pairs) != len(expected):
        return False
    for i, j, row in pairs:
        target, common = _spoly(basis[i], basis[j])
        if (
            len(row) != len(basis)
            or sum((a * b for a, b in zip(row, basis, strict=True)), ring.zero) != target
        ):
            return False
        if any(
            q and ring.order((q * g).LM) >= ring.order(common)
            for q, g in zip(row, basis, strict=True)
        ):
            return False
    return True


def _proof_data(source, basis, budget):
    ring = PolyRing(basis.gens, polynomial_domain(basis.domain), basis.order)
    original = tuple(_polynomial(p, ring) for p in source)
    candidate = tuple(_polynomial(p, ring) for p in basis.polys)
    if max(len(original), len(candidate)) > budget.max_basis:
        raise GroebnerLimitError("basis certificate basis-count budget exceeded")
    if len(candidate) * (len(candidate) - 1) // 2 > budget.max_pairs:
        raise GroebnerLimitError("basis certificate pair-count budget exceeded")
    if sum(map(len, original + candidate)) > budget.max_proof_nodes:
        raise GroebnerLimitError("basis certificate input-node budget exceeded")
    memberships = construct_membership_witnesses(
        original, candidate, ring, max_basis=budget.max_basis, max_pairs=budget.max_pairs
    )
    reductions, pairs = [], []
    for target in original:
        quotients, remainder = _divide(target, candidate)
        if remainder:
            raise GroebnerVerificationError("input generator failed exact basis reduction")
        reductions.append(tuple(quotients))
    for i, j in combinations(range(len(candidate)), 2):
        quotients, remainder = _divide(_spoly(candidate[i], candidate[j])[0], candidate)
        if remainder:
            raise GroebnerVerificationError("S-pair failed exact basis reduction")
        pairs.append((i, j, tuple(quotients)))
    all_polys = (
        original
        + candidate
        + tuple(p for row in memberships + tuple(reductions) for p in row)
        + tuple(p for _, _, row in pairs for p in row)
    )
    if sum(map(len, all_polys)) > budget.max_proof_nodes:
        raise GroebnerLimitError("basis certificate witness-node budget exceeded")
    return memberships, tuple(reductions), tuple(pairs)


@dataclass(frozen=True)
class GroebnerCertificate:
    """Exact proof reusable without recomputing a reference Gröbner basis.

    Every use replays generator identities and Buchberger's criterion. No
    cached boolean, hash, modular image, or producer result is trusted. Proof
    construction has independent budgets and may cost more than computation.
    """

    source: tuple
    basis: object
    memberships: tuple
    generator_reductions: tuple
    pair_reductions: tuple
    proof_limits: NativeF4Limits

    def __reduce__(self):
        from ._coefficient_state import domain_state

        return _restore_certificate, (
            self.source,
            tuple(p.as_expr() for p in self.basis.polys),
            self.basis.gens,
            domain_state(self.basis.domain),
            str(self.basis.order),
            self.memberships,
            self.generator_reductions,
            self.pair_reductions,
            self.proof_limits,
        )

    def verify(self):
        """Check the reduced basis, both ideal inclusions and all standard pairs."""
        try:
            basis = self.basis
            if (
                not basis.domain.is_Field
                or not basis.domain.is_Exact
                or str(basis.order) not in ("lex", "grlex", "grevlex")
            ):
                return False
            ring = PolyRing(basis.gens, polynomial_domain(basis.domain), basis.order)
            source = tuple(_polynomial(p, ring) for p in self.source)
            candidate = tuple(_polynomial(p, ring) for p in basis.polys)
            memberships = tuple(
                tuple(_polynomial(p, ring) for p in row) for row in self.memberships
            )
            reductions = tuple(
                tuple(_polynomial(p, ring) for p in row) for row in self.generator_reductions
            )
            pairs = tuple(
                (i, j, tuple(_polynomial(p, ring) for p in row))
                for i, j, row in self.pair_reductions
            )
            count = (
                sum(map(len, source + candidate))
                + sum(len(p) for row in memberships + reductions for p in row)
                + sum(len(p) for _, _, row in pairs for p in row)
            )
            return count <= self.proof_limits.max_proof_nodes and _identities(
                source, candidate, ring, memberships, reductions, pairs
            )
        except (
            ValueError,
            TypeError,
            AttributeError,
            IndexError,
            ArithmeticError,
            sp.PolynomialError,
            CoercionFailed,
        ):
            return False

    def prepare(self):
        """Verify a detached snapshot once for repeated mathematical reuse."""
        from .verified_families import VerifiedGroebnerFamily

        return VerifiedGroebnerFamily(self)

    def specialization_certificate(self, embedding):
        """Map every proof coefficient through a checked algebraic-field specialization.

        Poles in any identity are rejected. Successful replay proves the
        specialized reduced basis without a fresh Gröbner computation.
        """
        from sympy.polys.polyoptions import build_options

        from .algebraic_function_fields import FieldSpecialization

        self.verified_basis()
        if (
            not isinstance(embedding, FieldSpecialization)
            or embedding.source != self.basis.domain
            or not embedding.verify()
        ):
            raise ValueError("a matching verified field specialization is required")
        target, variables = embedding.domain, self.basis.gens

        def mapped(expression):
            p = polynomial(expression, variables, self.basis.domain)
            return sp.Poly.from_dict(
                {m: target.convert(embedding._evaluate(c)) for m, c in p.rep.to_dict().items()},
                *variables,
                domain=target,
            )

        options = build_options(
            variables, {"domain": target, "order": str(self.basis.order), "polys": False}
        )
        basis = sp.GroebnerBasis._new([mapped(p) for p in self.basis.polys], options)
        result = GroebnerCertificate(
            tuple(mapped(p).as_expr() for p in self.source),
            basis,
            tuple(tuple(mapped(p).as_expr() for p in row) for row in self.memberships),
            tuple(tuple(mapped(p).as_expr() for p in row) for row in self.generator_reductions),
            tuple(
                (i, j, tuple(mapped(p).as_expr() for p in row))
                for i, j, row in self.pair_reductions
            ),
            self.proof_limits,
        )
        result.verified_basis()
        return result

    def specialize(self, embedding):
        """Return a basis by specializing and replaying every stored identity."""
        return self.specialization_certificate(embedding).basis

    def verified_basis(self):
        """Return the mathematical basis after independent proof replay."""
        if not self.verify():
            raise GroebnerVerificationError("basis certificate failed verification")
        return self.basis


def _restore_certificate(
    source, polynomials, variables, domain, order, memberships, reductions, pairs, budget
):
    from sympy.polys.polyoptions import build_options

    from ._coefficient_state import restore_domain

    domain = restore_domain(domain)
    basis = sp.GroebnerBasis._new(
        [polynomial(p, variables, domain) for p in polynomials],
        build_options(variables, {"domain": domain, "order": order, "polys": False}),
    )
    return GroebnerCertificate(source, basis, memberships, reductions, pairs, budget)


def certify_groebner_basis(
    polynomials,
    variables,
    *,
    candidate=None,
    domain=None,
    order="grevlex",
    backend="native",
    limits=None,
    proof_limits=None,
):
    """Produce exact reusable witnesses for a supplied or newly computed basis.

    A supplied ``candidate`` is checked through identities, without computing
    another basis. Its field, variables and order must match this request.
    """
    from ._validation import normalize_equation, validate_variables
    from .basis import groebner_basis

    variables = validate_variables(variables)
    source = tuple(normalize_equation(p) for p in polynomials)
    budget = NativeF4Limits() if proof_limits is None else proof_limits
    if not isinstance(budget, NativeF4Limits):
        raise TypeError("proof_limits must be NativeF4Limits")
    if candidate is None:
        candidate = groebner_basis(
            source, variables, domain=domain, order=order, backend=backend, limits=limits
        )
    if (
        not isinstance(candidate, sp.GroebnerBasis)
        or candidate.gens != variables
        or str(candidate.order) != order
        or (domain is not None and candidate.domain != coefficient_domain(domain))
    ):
        raise ValueError("candidate field, variables and order must match the request")
    source = tuple(polynomial(p, variables, candidate.domain).as_expr() for p in source)
    memberships, reductions, pairs = _proof_data(source, candidate, budget)
    result = GroebnerCertificate(
        source,
        candidate,
        tuple(tuple(p.as_expr() for p in row) for row in memberships),
        tuple(tuple(p.as_expr() for p in row) for row in reductions),
        tuple((i, j, tuple(p.as_expr() for p in row)) for i, j, row in pairs),
        budget,
    )
    result.verified_basis()
    return result
