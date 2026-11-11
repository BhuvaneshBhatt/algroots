"""Verified Gröbner bases in an explicit coefficient field and term order."""

from time import perf_counter

import sympy as sp

from ._validation import normalize_equation, validate_variables
from .groebner_errors import GroebnerVerificationError
from .profiling import current_record, profiled_groebner


@profiled_groebner
def groebner_basis(
    polynomials,
    variables,
    *,
    domain=None,
    order="grevlex",
    backend="native",
    limits=None,
    execution=None,
    certificate=None,
    verification="reference",
    proof_limits=None,
):
    """Return a reduced monic SymPy ``GroebnerBasis`` over an exact field.

    ``variables`` fixes coordinate order; ``order`` is lex, grlex or grevlex.
    Rational input is computed over QQ. Supply ``domain=GF(p)`` for a prime
    field. Validated GF(p^d) domains use the FLINT finite-extension kernel.
    Native modular images also support number fields and rational
    function fields over QQ or number fields, with FLINT extension support
    installed for algebraic constants and parameter-dependent algebraic towers.
    ``backend="native"`` is the default and requests the compiled F4 kernel.
    Use ``backend="sympy"`` for the portable implementation. The native algorithm is
    ``f4_tiled_finish``. Native operation limits use ``NativeF4Limits``.

    Accelerated results use independent exact SymPy comparison by default.
    ``verification="certificate"`` instead checks generator-combination and
    standard S-pair identities, with optional ``proof_limits``. A reusable
    ``certificate`` checks an existing proof for this exact system and avoids
    basis construction. Raw certificates replay their identities; prepared
    snapshots retain a detached proof checked at construction. Both ideal
    inclusions and the reduced-basis property remain required. Reference
    verification, when selected, is included in total time.
    No partial or unverified result is returned on failure.

    ``groebner_backends()`` describes supported domains and installation steps.
    ``execution=ExecutionLimits(timeout=...)`` runs the complete verified call
    in a spawned worker; use a guarded script for multiprocessing.
    """
    if execution is not None:
        from .execution import run_supervised

        return run_supervised(
            groebner_basis,
            polynomials,
            variables,
            domain=domain,
            order=order,
            backend=backend,
            limits=limits,
            execution=execution,
            certificate=certificate,
            verification=verification,
            proof_limits=proof_limits,
        )
    _validate_backend(backend, limits)
    from .native_f4 import NativeF4Limits

    if verification not in ("reference", "certificate"):
        raise ValueError("verification must be reference or certificate")
    if proof_limits is not None and not isinstance(proof_limits, NativeF4Limits):
        raise TypeError("proof_limits must be NativeF4Limits")
    if proof_limits is not None and verification != "certificate":
        raise ValueError("proof_limits requires verification='certificate'")
    variables = validate_variables(variables)
    expressions = tuple(normalize_equation(p) for p in polynomials)
    if order not in ("lex", "grlex", "grevlex"):
        raise ValueError("order must be lex, grlex or grevlex")
    if domain is None:
        if any(
            constant.has(*variables)
            for expression in expressions
            for constant in expression.atoms(sp.CRootOf, sp.AlgebraicNumber)
        ):
            from .domains import infer_coefficient_field

            domain = infer_coefficient_field(expressions, variables)
        elif expressions:
            parsed, options = sp.parallel_poly_from_expr(expressions, *variables, extension=True)
            domain = options.domain.get_field()
            expressions = tuple(poly.set_domain(domain) for poly in parsed)
        else:
            domain = sp.QQ
    else:
        from .domains import coefficient_domain

        domain = coefficient_domain(domain)
        if not domain.is_Field:
            raise ValueError("domain must be an exact field; use QQ for rational coefficients")
    from .finite_fields import FiniteFieldExtension

    if getattr(domain, "is_FiniteExtension", False):
        from .algebraic_function_fields import AlgebraicFunctionField

        if not isinstance(domain, (AlgebraicFunctionField, FiniteFieldExtension)):
            raise ValueError(
                "use a validated AlgebraicFunctionField or finite_field extension domain"
            )
    if not domain.is_Exact:
        raise ValueError(
            "domain must be exact; use QQ, GF(p) for prime p, or an exact number field"
        )
    if domain.is_FiniteField and not sp.isprime(domain.characteristic()):
        raise ValueError("finite coefficient fields require a prime modulus")
    if getattr(domain, "is_AlgebraicFunctionField", False) or isinstance(
        domain, FiniteFieldExtension
    ):
        from .quotient import _check_coefficient_symbols

        _check_coefficient_symbols(domain, variables)
    if backend != "sympy" and certificate is None:
        from .backends import _require_backend

        _require_backend(backend, domain, len(variables))
    if certificate is not None:
        from .basis_certificates import GroebnerCertificate
        from .domains import polynomial
        from .verified_families import VerifiedGroebnerFamily

        if not isinstance(certificate, (GroebnerCertificate, VerifiedGroebnerFamily)):
            raise TypeError("certificate must be a GroebnerCertificate")
        basis = certificate.verified_basis()
        if (
            basis.gens != variables
            or basis.domain != domain
            or str(basis.order) != order
            or tuple(polynomial(p, variables, domain) for p in expressions)
            != tuple(polynomial(p, variables, domain) for p in certificate.source)
        ):
            raise GroebnerVerificationError("basis certificate does not match the requested system")
        return basis
    return _verified_basis(
        expressions, variables, domain, order, backend, limits, verification, proof_limits
    )


def _verified_basis(
    expressions,
    variables,
    domain,
    order,
    backend,
    limits,
    verification="reference",
    proof_limits=None,
):
    """Preserve native embeddings and verify candidates in their original field."""
    from sympy.polys.groebnertools import groebner as reference_groebner
    from sympy.polys.polyoptions import build_options
    from sympy.polys.rings import PolyRing

    from .domains import polynomial, polynomial_domain
    from .groebner import compute_groebner_basis

    profile = current_record()
    started = perf_counter() if profile is not None else 0.0
    if profile is not None:
        profile["domain"] = str(domain)
    ring = PolyRing(variables, polynomial_domain(domain), order)
    inputs = tuple(
        ring.from_dict(polynomial(p, variables, ring.domain).rep.to_dict()) for p in expressions
    )
    if profile is not None:
        profile["conversion_seconds"] = perf_counter() - started
        started = perf_counter()
    candidate = tuple(
        compute_groebner_basis(inputs, ring, backend=backend, limits=limits, certify=False)
    )
    if profile is not None:
        profile["candidate_seconds"] = perf_counter() - started
        started = perf_counter()
    if backend != "sympy" and verification == "reference":
        reference = tuple(reference_groebner([p for p in inputs if p], ring))
        # Reduced monic bases are unique. Comparing exact coefficients also
        # rejects a strict larger ideal such as an erroneous unit ideal.
        if len(candidate) != len(reference) or {frozenset(p.items()) for p in candidate} != {
            frozenset(p.items()) for p in reference
        }:
            raise GroebnerVerificationError(f"{backend} basis failed exact verification")
    if profile is not None:
        profile["verification_seconds"] = perf_counter() - started if backend != "sympy" else 0.0
        started = perf_counter()
    candidate = tuple(sorted(candidate, key=lambda p: ring.order(p.LM), reverse=True))
    options = build_options(variables, {"domain": domain, "order": order, "polys": False})
    polys = [
        sp.Poly.from_dict(
            {m: domain.convert(c, ring.domain) for m, c in p.items()}, *variables, domain=domain
        )
        for p in candidate
    ]
    result = sp.GroebnerBasis._new(polys, options)
    if profile is not None:
        profile["output_seconds"] = perf_counter() - started
    if verification == "certificate":
        from .basis_certificates import certify_groebner_basis

        started = perf_counter() if profile is not None else 0.0
        try:
            certify_groebner_basis(
                tuple(p.as_expr() for p in inputs),
                variables,
                candidate=result,
                domain=domain,
                order=order,
                proof_limits=proof_limits,
            )
        except ValueError as exc:
            raise GroebnerVerificationError(
                f"{backend} basis failed exact certificate verification"
            ) from exc
        if profile is not None:
            profile["verification_seconds"] = perf_counter() - started
    return result


def _validate_backend(backend, limits):
    """Validate engine selection even when presolve needs no basis."""
    from .native_f4 import NativeF4Limits

    if not isinstance(backend, str):
        raise TypeError("backend must be sympy or native")
    if backend not in ("sympy", "native"):
        raise ValueError("backend must be sympy or native")
    if limits is not None:
        if backend == "sympy":
            raise ValueError("native limits do not apply to SymPy")
        if not isinstance(limits, NativeF4Limits):
            raise TypeError("native limits must be NativeF4Limits")
