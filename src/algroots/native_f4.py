"""Tiled native F4 over prime fields and verified modular QQ reconstruction.

Every prime image computes its own coefficients and pivots. A reconstructed
candidate is unverified until an independent exact verifier accepts it. Use
``groebner_basis`` for verified mathematical results or ``certify=True`` with
this module's expert reconstruction interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import sympy as sp
from sympy.polys.rings import PolyElement, PolyRing

from .groebner_errors import GroebnerBackendUnavailable, GroebnerLimitError
from .profiling import record_native_stats

_MAX_NATIVE_PRIME = 1000000007


class NativeF4Unavailable(GroebnerBackendUnavailable):
    """The optional compiled full-loop F4 extension was not built."""


class NativeF4LimitError(GroebnerLimitError):
    """A computation exceeded its explicit resource or reconstruction budget."""


@dataclass(frozen=True)
class NativeF4Limits:
    """Operation and reconstruction budgets for tiled native F4.

    Field degree bounds the constant power basis. Parameter degree bounds each
    reconstructed numerator and denominator; point and term budgets apply per
    prime. Proof construction has a separate node budget. Kernel limits do not
    impose a deadline on independent reference verification. Prime-field F4
    batches at least 128 pairs per degree; extension-field arithmetic uses
    batch_size directly.
    """

    max_pairs: int = 20000
    max_terms: int = 100000
    max_basis: int = 256
    max_proof_nodes: int = 1000000
    max_reduction_steps: int = 250000
    batch_size: int = 8
    max_primes: int = 96
    first_prime: int = 1009
    min_primes: int = 2
    max_modulus_bits: int = 1024
    collect_batch_metrics: bool = False
    max_field_degree: int = 16
    max_parameter_degree: int = 16
    max_evaluation_points: int = 256
    max_interpolation_terms: int = 512
    max_field_matrix_cells: int = 4194304
    parameter_interpolation: str = "auto"
    max_sparse_monomials: int = 100000

    def __post_init__(self):
        if (
            isinstance(self.first_prime, bool)
            or not isinstance(self.first_prime, int)
            or not 2 <= self.first_prime <= _MAX_NATIVE_PRIME
            or not sp.isprime(self.first_prime)
        ):
            raise ValueError("first_prime must be prime and at most 1000000007")
        for name in (
            "max_pairs",
            "max_terms",
            "max_basis",
            "max_proof_nodes",
            "max_reduction_steps",
            "batch_size",
            "max_primes",
            "min_primes",
            "max_modulus_bits",
            "max_field_degree",
            "max_evaluation_points",
            "max_interpolation_terms",
            "max_field_matrix_cells",
            "max_sparse_monomials",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.max_parameter_degree, bool)
            or not isinstance(self.max_parameter_degree, int)
            or self.max_parameter_degree < 0
        ):
            raise ValueError("max_parameter_degree must be a nonnegative integer")
        if self.parameter_interpolation not in ("auto", "dense", "sparse"):
            raise ValueError("parameter_interpolation must be auto, dense, or sparse")
        if not isinstance(self.collect_batch_metrics, bool):
            raise ValueError("collect_batch_metrics must be boolean")
        if self.min_primes > self.max_primes:
            raise ValueError("min_primes cannot exceed max_primes")


@dataclass(frozen=True)
class NativeF4Result:
    """Native basis with reconstruction evidence and certification status."""

    basis: tuple
    primes: tuple
    verification: str
    profile: dict
    # True only if an independent exact QQ verifier has accepted the result.
    certified: bool = False


def compiled_available():
    """Return whether the optional compiled F4 extension is importable."""
    try:
        from . import _native_f4
    except ImportError:
        return False
    return hasattr(_native_f4, "compute")


def native_prime_basis(polynomials, ring, prime, *, limits=None):
    """Compute a reduced GF(p) basis and collect native operation statistics."""
    if limits is None:
        limits = NativeF4Limits()
    if not isinstance(limits, NativeF4Limits):
        raise TypeError("limits must be NativeF4Limits")
    if not isinstance(ring, PolyRing):
        raise TypeError("expected a SymPy PolyRing")
    if isinstance(prime, bool) or not isinstance(prime, int) or not 2 <= prime <= _MAX_NATIVE_PRIME:
        raise ValueError("prime must be an integer between 2 and 1000000007")
    if not 1 <= len(ring.symbols) <= 64:
        raise ValueError("native kernels require between 1 and 64 variables")
    if int(ring.domain.characteristic()) != prime or not sp.isprime(prime):
        raise ValueError("ring must have precisely the supplied prime characteristic")
    if str(ring.order) not in ("lex", "grlex", "grevlex"):
        raise ValueError("unsupported monomial order")
    try:
        from . import _native_f4
    except ImportError as exc:
        raise NativeF4Unavailable(
            "build with ALGROOTS_BUILD_NATIVE_F4=1 python setup.py build_ext --inplace"
        ) from exc
    inputs = [
        p
        if isinstance(p, PolyElement) and p.ring is ring
        else ring.from_expr(p.as_expr() if hasattr(p, "as_expr") else p)
        for p in polynomials
    ]
    sparse = [{tuple(m): int(c) % prime for m, c in p.terms()} for p in inputs]
    kernel_started = perf_counter() if limits.collect_batch_metrics else 0.0
    try:
        bases, stats = _native_f4.compute(
            sparse,
            prime,
            len(ring.symbols),
            str(ring.order),
            max_pairs=limits.max_pairs,
            max_terms=limits.max_terms,
            max_basis=limits.max_basis,
            batch_size=limits.batch_size,
            max_reduction_steps=limits.max_reduction_steps,
            collect_batch_metrics=limits.collect_batch_metrics,
        )
    except _native_f4.LimitError as exc:
        raise NativeF4LimitError(str(exc)) from exc
    if limits.collect_batch_metrics:
        stats["native_call_seconds"] = perf_counter() - kernel_started
    return tuple(ring.from_dict(p) for p in bases), dict(stats)


def _primitive_integer_inputs(inputs, ring):
    from .modular_arithmetic import _primitive_integer_polynomials_and_scales

    values, _ = _primitive_integer_polynomials_and_scales(inputs, ring)
    return values


def _mod_prime_rational(candidate, prime, ring):
    from_ring = PolyRing(ring.symbols, sp.GF(prime), str(ring.order))
    reduced = []
    for p in candidate:
        d = {}
        for m, c in p.terms():
            numerator, denominator = sp.fraction(sp.Rational(c))
            den = int(denominator) % prime
            if den == 0:
                return None
            coeff = int(numerator) % prime * pow(den, -1, prime) % prime
            if coeff:
                d[tuple(m)] = coeff
        reduced.append(from_ring.from_dict(d))
    return tuple(reduced)


def native_modular_basis(polynomials, ring, *, limits=None, certify=False):
    """Compute a reduced QQ candidate with a native, modular F4 control loop.

    ``certify=False`` returns an explicitly *unverified* QQ candidate when CRT
    reconstruction stabilizes and a fresh prime independently matches it.
    No proof matrices or QQ reference basis are computed in this mode.

    ``certify=True`` uses an independently checked exact proof from alggeom;
    certificate production is performed only upon request and may be slow.
    Failure to produce a proof raises, rather than returning a certified flag.
    """
    if limits is None:
        limits = NativeF4Limits()
    if not isinstance(limits, NativeF4Limits):
        raise TypeError("limits must be NativeF4Limits")
    if not isinstance(ring, PolyRing) or ring.domain != sp.QQ:
        raise ValueError("native modular QQ lifting requires an explicit QQ polynomial ring")
    if str(ring.order) not in ("lex", "grlex", "grevlex"):
        raise ValueError("unsupported monomial order")
    original = tuple(
        ring.from_expr(p.as_expr() if hasattr(p, "as_expr") else p) for p in polynomials
    )
    if not any(original):
        return NativeF4Result((), (), "exact-zero-ideal", {}, True)
    integer_polys = _primitive_integer_inputs(original, ring)
    from .modular_arithmetic import rational_reconstruction

    buckets = {}
    profile = {
        "kind": "rational_modular",
        "coefficient_domain": "QQ",
        "images": [],
        "reconstruction_seconds": 0.0,
        "validation_seconds": 0.0,
        "prime_seconds": 0.0,
        "crt_seconds": 0.0,
        "discarded_primes": 0,
    }
    prime = limits.first_prime - 1
    for _ in range(limits.max_primes):
        prime = int(sp.nextprime(prime))
        if prime > _MAX_NATIVE_PRIME:
            raise NativeF4LimitError("no supported native reconstruction primes remain")
        finite_ring = PolyRing(ring.symbols, sp.GF(prime), str(ring.order))
        sparse = [
            finite_ring.from_dict({tuple(m): int(c) % prime for m, c in f.items()})
            for f in integer_polys
        ]
        start = perf_counter()
        basis, stats = native_prime_basis(sparse, finite_ring, prime, limits=limits)
        profile["prime_seconds"] += perf_counter() - start
        if limits.collect_batch_metrics:
            profile["images"].append({"prime": prime, "profile": stats})
        normalized = tuple({tuple(m): int(c) % prime for m, c in f.terms()} for f in basis)
        signature = tuple(tuple(f.LM) for f in basis)
        # Do not mix incompatible leading ideals or coefficient supports.
        support = tuple(tuple(sorted(d)) for d in normalized)
        key = signature, support
        modulus, residues, accepted = buckets.get(key, (1, tuple({} for _ in basis), ()))
        start = perf_counter()
        combined = []
        inv_modulus = pow(modulus % prime, -1, prime)
        for prior, current in zip(residues, normalized, strict=True):
            merged = {}
            for monomial in current:
                old = prior.get(monomial, 0)
                coefficient = current[monomial]
                updated = old + modulus * ((coefficient - old) * inv_modulus % prime)
                merged[monomial] = updated
            combined.append(merged)
        residues = tuple(combined)
        modulus *= prime
        accepted = (*accepted, prime)
        buckets[key] = modulus, residues, accepted
        profile["crt_seconds"] += perf_counter() - start
        if modulus.bit_length() > limits.max_modulus_bits:
            buckets.pop(key, None)
            profile["discarded_primes"] += 1
            continue
        if len(accepted) < limits.min_primes:
            continue
        reconstruction_started = perf_counter()
        converted = []
        for d in residues:
            output = {}
            for monomial, residue in d.items():
                value = rational_reconstruction(residue, modulus)
                if value is None:
                    break
                if value:
                    output[monomial] = value
            else:
                converted.append(ring.from_dict(output))
                continue
            break
        profile["reconstruction_seconds"] += perf_counter() - reconstruction_started
        if len(converted) != len(basis):
            continue
        candidate = tuple(converted)
        # Exact finite-field agreement with a fresh independent prime is a
        # sanity check only; it is *not* a proof over QQ.
        prime_validation = int(sp.nextprime(prime))
        if prime_validation > _MAX_NATIVE_PRIME:
            raise NativeF4LimitError("no supported native validation prime remains")
        vr = PolyRing(ring.symbols, sp.GF(prime_validation), str(ring.order))
        vinputs = [
            vr.from_dict({tuple(m): int(c) % prime_validation for m, c in f.items()})
            for f in integer_polys
        ]
        validation_started = perf_counter()
        validation, validation_stats = native_prime_basis(
            vinputs, vr, prime_validation, limits=limits
        )
        profile["validation_seconds"] += perf_counter() - validation_started
        if limits.collect_batch_metrics:
            profile["images"].append(
                {"prime": prime_validation, "validation": True, "profile": validation_stats}
            )
        reduced = _mod_prime_rational(candidate, prime_validation, ring)
        if reduced is None or tuple(dict(x.terms()) for x in reduced) != tuple(
            dict(x.terms()) for x in validation
        ):
            continue
        profile["validation_prime"] = prime_validation
        if not certify:
            record_native_stats(profile)
            return NativeF4Result(candidate, accepted, "unchecked-modular-lift", profile)
        try:
            from alggeom.groebner_certificates import (
                construct_membership_witnesses,
                verify_reduced_groebner_basis,
            )
        except ImportError as exc:
            raise ImportError("certify=True requires alggeom's independent verifier") from exc
        witnesses = construct_membership_witnesses(
            original, candidate, ring, max_basis=limits.max_basis, max_pairs=limits.max_pairs
        )
        if not verify_reduced_groebner_basis(
            original, candidate, ring, membership_witnesses=witnesses
        ):
            raise ArithmeticError("independent rational Gröbner verification failed")
        record_native_stats(profile)
        return NativeF4Result(candidate, accepted, "independently-certified", profile, True)
    raise NativeF4LimitError("no stable rational reconstruction within the prime budget")


__all__ = [
    "NativeF4Unavailable",
    "NativeF4LimitError",
    "NativeF4Limits",
    "NativeF4Result",
    "compiled_available",
    "native_prime_basis",
    "native_modular_basis",
]
