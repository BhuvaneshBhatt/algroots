"""Modular syzygy-module Gröbner bases with independent exact QQ certification.

CRT and rational reconstruction provide *candidates*, not proof. The
verifier derives a complete generating set of the input syzygy module from
an independently certified ideal GB, then establishes equality and the
module Buchberger criterion directly over QQ. It never calls a QQ module
Groebner-basis routine to accept a candidate.
"""

from dataclasses import dataclass
from math import gcd, lcm

import sympy as sp

from .modular_arithmetic import _crt_polynomial, rational_reconstruction
from .syzygy_normal_form import (
    SyzygyLimits,
    SyzygyNormalFormLimitError,
    _divides,
    _lead,
    _module_groebner,
    _normal_form,
    _s_pair,
    syzygy_generators,
)


class ModularSyzygyLimitError(ArithmeticError):
    """No independently certified module GB within modular resource budgets."""


@dataclass(frozen=True)
class ModularSyzygyLimits:
    """Budgets for modular construction and exact verification of syzygy bases."""

    max_primes: int = 32
    first_prime: int = 1009
    min_primes: int = 2
    max_modulus_bits: int = 1024
    max_basis: int = 128
    max_pairs: int = 2000
    max_terms: int = 30000
    max_steps: int = 100000
    finite_field_kernel: str = "buchberger"
    f4_matrix_backend: str = "auto"
    f4_batch_size: int = 8
    f4_reducer_policy: str = "full"

    def __post_init__(self):
        if (
            self.max_primes < 1
            or self.first_prime < 2
            or self.max_modulus_bits < 2
            or not 1 <= self.min_primes <= self.max_primes
            or min(self.max_basis, self.max_pairs, self.max_terms, self.max_steps) < 1
        ):
            raise ValueError("invalid modular syzygy resource limits")
        if self.finite_field_kernel not in ("buchberger", "f4"):
            raise ValueError("unknown module finite-field kernel")
        if (
            self.f4_matrix_backend not in ("auto", "python", "compiled")
            or self.f4_batch_size < 1
            or self.f4_reducer_policy not in ("full", "frontier")
        ):
            raise ValueError("invalid module F4 matrix configuration")

    def module_limits(self):
        return SyzygyLimits(self.max_basis, self.max_pairs, self.max_terms, self.max_steps)


@dataclass(frozen=True)
class ModularSyzygyResult:
    """A reconstructed syzygy basis with its accepted primes and exact verification evidence."""

    basis: tuple
    primes: tuple
    modulus: int
    discarded_primes: tuple


def _integer_generators(relations, ring):
    output = []
    for row in relations:
        denominators = [int(ring.domain.denom(coeff)) for poly in row for _, coeff in poly.terms()]
        denominator = lcm(*denominators) if denominators else 1
        coordinates = []
        all_coefficients = []
        for poly in row:
            coordinate = {}
            for power, coeff in poly.terms():
                num = int(ring.domain.numer(coeff))
                den = int(ring.domain.denom(coeff))
                integer = num * (denominator // den)
                coordinate[tuple(power)] = integer
                all_coefficients.append(integer)
            coordinates.append(coordinate)
        content = gcd(*all_coefficients) if all_coefficients else 1
        if content == 0:
            continue
        output.append(tuple({k: v // content for k, v in part.items()} for part in coordinates))
    return tuple(output)


def _module_signature(row):
    lead = _lead(row, row[0].ring)
    return lead[0], lead[1]


def verify_syzygy_basis(inputs, ideal_basis, memberships, module_basis, ring, *, limits=None):
    """Independent exact module certificate, including generation completeness.

    Uses Schreyer-lift generators of all syzygies, not a modular or rational
    reference module GB. Therefore an arbitrary submodule of syzygies cannot
    pass by merely satisfying Buchberger's criterion.
    """
    if limits is None:
        limits = SyzygyLimits()
    relations = syzygy_generators(inputs, ideal_basis, memberships, ring)
    module_basis = tuple(tuple(a for a in row) for row in module_basis)
    n = len(inputs)
    if any(len(row) != n or not any(row) for row in module_basis):
        return False
    leads = [_lead(row, ring) for row in module_basis]
    if any(leader[2] != ring.domain.one for leader in leads):
        return False
    for i, row in enumerate(module_basis):
        if sum((a * f for a, f in zip(row, inputs, strict=True)), ring.zero):
            return False
        for position, poly in enumerate(row):
            for powers, _ in poly.terms():
                for j, leader in enumerate(leads):
                    if (
                        leader[0] == position
                        and _divides(leader[1], powers)
                        and not (i == j and position == leader[0] and tuple(powers) == leader[1])
                    ):
                        return False
    for relation in relations:
        if any(_normal_form(relation, module_basis, ring, limits)):
            return False
    for i, left in enumerate(module_basis):
        for right in module_basis[:i]:
            if _lead(left, ring)[0] == _lead(right, ring)[0] and any(
                _normal_form(_s_pair(left, right, ring), module_basis, ring, limits)
            ):
                return False
    return True


def modular_syzygy_groebner_basis(inputs, ideal_basis, memberships, ring, *, limits=None):
    """Modular reconstruction with exact module-GB certification over QQ."""
    from .basis_verification import verify_rational_basis

    if limits is None:
        limits = ModularSyzygyLimits()
    if not isinstance(limits, ModularSyzygyLimits):
        raise TypeError("limits must be ModularSyzygyLimits")
    inputs = tuple(inputs)
    ideal_basis = tuple(ideal_basis)
    memberships = tuple(tuple(r) for r in memberships)
    if ring.domain != sp.QQ:
        raise ValueError("modular syzygy reconstruction requires QQ")
    if not verify_rational_basis(inputs, ideal_basis, ring, membership_witnesses=memberships):
        raise ValueError("input ideal GB and witnesses are not certified")
    relations = syzygy_generators(inputs, ideal_basis, memberships, ring)
    if not relations:
        return ModularSyzygyResult((), (), 1, ())
    ints = _integer_generators(relations, ring)
    buckets = {}
    module_symbolic_cache = {}
    attempted = []
    prime = limits.first_prime - 1
    n = len(inputs)
    for _ in range(limits.max_primes):
        prime = int(sp.nextprime(prime))
        attempted.append(prime)
        ff = ring.clone(domain=sp.GF(prime))
        modular = tuple(
            tuple(
                ff.from_dict({key: coeff % prime for key, coeff in entry.items() if coeff % prime})
                for entry in row
            )
            for row in ints
        )
        if any(not any(row) for row in modular):
            continue
        try:
            if limits.finite_field_kernel == "f4":
                module_basis = _module_groebner(
                    modular,
                    ff,
                    limits.module_limits(),
                    kernel="f4",
                    matrix_backend=limits.f4_matrix_backend,
                    symbolic_cache=module_symbolic_cache,
                    batch_size=limits.f4_batch_size,
                    reducer_policy=limits.f4_reducer_policy,
                )
            else:
                module_basis = _module_groebner(modular, ff, limits.module_limits())
        except SyzygyNormalFormLimitError:
            continue
        signature = tuple(_module_signature(row) for row in module_basis)
        if signature not in buckets:
            buckets[signature] = (1, (), tuple(tuple({} for _ in range(n)) for _ in module_basis))
        modulus, primes, old_rows = buckets[signature]
        current_rows = tuple(
            tuple(
                {tuple(powers): int(value) % prime for powers, value in entry.terms()}
                for entry in row
            )
            for row in module_basis
        )
        accumulated = tuple(
            tuple(_crt_polynomial(a, b, modulus, prime) for a, b in zip(old, new, strict=True))
            for old, new in zip(old_rows, current_rows, strict=True)
        )
        new_modulus = modulus * prime
        if new_modulus.bit_length() > limits.max_modulus_bits:
            del buckets[signature]
            continue
        primes = (*primes, prime)
        buckets[signature] = (new_modulus, primes, accumulated)
        if len(primes) < limits.min_primes:
            continue
        recovered = []
        for row in accumulated:
            result_row = []
            for part in row:
                rebuilt = {}
                for power, residue in part.items():
                    coeff = rational_reconstruction(residue, new_modulus)
                    if coeff is None:
                        break
                    if coeff:
                        rebuilt[power] = coeff
                else:
                    result_row.append(ring.from_dict(rebuilt))
                    continue
                break
            if len(result_row) != n:
                break
            recovered.append(tuple(result_row))
        if len(recovered) != len(signature):
            continue
        candidate = tuple(recovered)
        if tuple(_module_signature(row) for row in candidate) != signature:
            continue
        if verify_syzygy_basis(
            inputs, ideal_basis, memberships, candidate, ring, limits=limits.module_limits()
        ):
            return ModularSyzygyResult(
                candidate, primes, new_modulus, tuple(p for p in attempted if p not in primes)
            )
    raise ModularSyzygyLimitError(f"no certified module basis after {len(attempted)} primes")


__all__ = [
    "ModularSyzygyLimitError",
    "ModularSyzygyLimits",
    "ModularSyzygyResult",
    "modular_syzygy_groebner_basis",
    "verify_syzygy_basis",
]
