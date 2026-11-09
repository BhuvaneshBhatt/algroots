"""Exact canonical polynomial-combination witnesses modulo the *full* syzygy module.

The order is position-over-term (POT) on the input-generator coefficient rows:
component 0 has the highest priority, with the supplied polynomial order within
components.  The normal form is unique for this fixed order and ordered inputs.

A certified Groebner basis G with a matrix A satisfying G=A*F is used to
construct a complete generating set of Syz(F), without recomputing a rational
Groebner basis of the ideal.  Schreyer S-pair relations for G plus the
relations F=B*G yield all syzygies of F.  A separate module Buchberger
computation reduces these generators to a module GB.  This optional step may
be expensive; resource limits fail closed.
"""

from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import zip_longest


class SyzygyNormalFormLimitError(ArithmeticError):
    """The exact module calculation exceeded its configured budget."""


@dataclass(frozen=True)
class SyzygyLimits:
    """Term, pair, basis, and reduction budgets for exact syzygy normal forms."""

    max_basis: int = 128
    max_pairs: int = 2000
    max_terms: int = 30000
    max_steps: int = 100000

    def __post_init__(self):
        if min(self.max_basis, self.max_pairs, self.max_terms, self.max_steps) < 1:
            raise ValueError("syzygy resource limits must be positive")


def _terms(row):
    return sum(len(poly) for poly in row)


def _lead(row, ring):
    """POT leading position, monomial, coefficient, or None for zero row."""
    for position, poly in enumerate(row):
        if poly:
            return position, tuple(poly.LM), poly.LC
    return None


def _divides(left, right):
    return all(a <= b for a, b in zip(left, right, strict=True))


def _monomial(ring, powers, coefficient):
    return ring.from_dict({tuple(powers): coefficient})


def _monic(row, ring):
    lead = _lead(row, ring)
    if lead is None:
        return tuple(row)
    inverse = ring.domain.one / lead[2]
    return tuple(poly * inverse for poly in row)


def _row_sub(row, multiplier, other):
    return tuple(a - multiplier * b for a, b in zip(row, other, strict=True))


def _normal_form(row, basis, ring, limits):
    """Deterministic module long division, reducing with the first applicable row."""
    current = tuple(row)
    rest = [ring.zero for _ in row]
    leads = [_lead(g, ring) for g in basis]
    steps = 0
    while (term := _lead(current, ring)) is not None:
        steps += 1
        if steps > limits.max_steps:
            raise SyzygyNormalFormLimitError("module division step budget exhausted")
        position, powers, coefficient = term
        for divisor, leader in zip(basis, leads, strict=True):
            if leader is None or leader[0] != position or not _divides(leader[1], powers):
                continue
            multiplier = _monomial(
                ring,
                (a - b for a, b in zip(powers, leader[1], strict=True)),
                coefficient / leader[2],
            )
            current = _row_sub(current, multiplier, divisor)
            break
        else:
            monomial = _monomial(ring, powers, coefficient)
            remaining = list(current)
            remaining[position] -= monomial
            current = tuple(remaining)
            rest[position] += monomial
        if _terms(current) + _terms(rest) > limits.max_terms:
            raise SyzygyNormalFormLimitError("module division term budget exhausted")
    return tuple(rest)


def _s_pair(left, right, ring):
    a = _lead(left, ring)
    b = _lead(right, ring)
    if a[0] != b[0]:
        return tuple(ring.zero for _ in left)
    lcm = tuple(max(x, y) for x, y in zip(a[1], b[1], strict=True))
    ma = _monomial(ring, (x - y for x, y in zip(lcm, a[1], strict=True)), ring.domain.one / a[2])
    mb = _monomial(ring, (x - y for x, y in zip(lcm, b[1], strict=True)), ring.domain.one / b[2])
    return _row_sub(tuple(ma * x for x in left), mb, right)


def _module_groebner(
    generators,
    ring,
    limits,
    *,
    kernel="buchberger",
    matrix_backend="auto",
    symbolic_cache=None,
    batch_size=8,
    reducer_policy="full",
):
    """Compute a reduced module GB in POT order. No ideal-GB oracle is called."""
    if kernel not in ("buchberger", "f4"):
        raise ValueError("unknown module kernel")
    basis = []
    pending = []
    for row in generators:
        if not any(row):
            continue
        reduced = _normal_form(row, basis, ring, limits)
        if not any(reduced):
            continue
        reduced = _monic(reduced, ring)
        if len(basis) >= limits.max_basis:
            raise SyzygyNormalFormLimitError("module basis budget exhausted")
        position = len(basis)
        for i, old in enumerate(basis):
            left, right = _lead(old, ring), _lead(reduced, ring)
            if left[0] == right[0]:
                lcm = tuple(max(x, y) for x, y in zip(left[1], right[1], strict=True))
                heappush(pending, (sum(lcm), len(pending), i, position))
        basis.append(reduced)
    pairs_seen = 0
    while pending:
        if kernel == "f4":
            from .module_f4 import module_f4_batch

            degree = pending[0][0]
            batch = []
            while pending and pending[0][0] == degree and len(batch) < batch_size:
                pairs_seen += 1
                if pairs_seen > limits.max_pairs:
                    raise SyzygyNormalFormLimitError("module S-pair budget exhausted")
                _, _, i, j = heappop(pending)
                batch.append((i, j))
            new_rows = module_f4_batch(
                batch,
                basis,
                ring,
                max_terms=limits.max_terms,
                matrix_backend=matrix_backend,
                symbolic_cache=symbolic_cache,
                reducer_policy=reducer_policy,
            )
        else:
            pairs_seen += 1
            if pairs_seen > limits.max_pairs:
                raise SyzygyNormalFormLimitError("module S-pair budget exhausted")
            _, _, i, j = heappop(pending)
            new_rows = (_s_pair(basis[i], basis[j], ring),)
        for candidate in new_rows:
            reduced = _normal_form(candidate, basis, ring, limits)
            if not any(reduced):
                continue
            reduced = _monic(reduced, ring)
            if len(basis) >= limits.max_basis:
                raise SyzygyNormalFormLimitError("module basis budget exhausted")
            position = len(basis)
            for k, old in enumerate(basis):
                left, right = _lead(old, ring), _lead(reduced, ring)
                if left[0] == right[0]:
                    lcm = tuple(max(x, y) for x, y in zip(left[1], right[1], strict=True))
                    heappush(pending, (sum(lcm), pairs_seen + len(pending), k, position))
            basis.append(reduced)
    # A reduced module GB has unique leading terms and unique monic tails.
    # Remove superfluous rows by divisibility before normal-form reduction.
    retained = []
    for i, row in enumerate(basis):
        leader = _lead(row, ring)
        if any(
            _lead(other, ring)[0] == leader[0]
            and _divides(_lead(other, ring)[1], leader[1])
            and (not _divides(leader[1], _lead(other, ring)[1]) or j < i)
            for j, other in enumerate(basis)
            if j != i
        ):
            continue
        retained.append(row)
    output = []
    for i, row in enumerate(retained):
        reduced = _normal_form(row, retained[:i] + retained[i + 1 :], ring, limits)
        if any(reduced):
            output.append(_monic(reduced, ring))
    output.sort(
        key=lambda row: (-_lead(row, ring)[0], ring.order(_lead(row, ring)[1])), reverse=True
    )
    if sum(_terms(row) for row in output) > limits.max_terms:
        raise SyzygyNormalFormLimitError("module basis term budget exhausted")
    return tuple(output)


def syzygy_generators(inputs, basis, memberships, ring):
    """Complete Schreyer-lift generators of Syz(inputs).

    Certifies the supplied ideal basis and traces independently. The
    construction requires no rational ideal or module Groebner computation.
    """
    from .basis_verification import verify_rational_basis

    inputs = tuple(inputs)
    basis = tuple(basis)
    memberships = tuple(tuple(row) for row in memberships)
    if not verify_rational_basis(inputs, basis, ring, membership_witnesses=memberships):
        raise ValueError("input basis and generator witnesses are not certified")
    n = len(inputs)
    relations = []
    # F_j = sum_i B_ji G_i. Thus e_j - sum_i B_ji A_i is a syzygy.
    for j, f in enumerate(inputs):
        quotients, remainder = f.div(basis) if basis else ((), f)
        if remainder:
            raise AssertionError("certified basis failed to divide an input")
        row = [ring.zero for _ in range(n)]
        row[j] = ring.one
        for quotient, witness in zip_longest(quotients, memberships, fillvalue=None):
            if quotient is None:
                quotient = ring.zero
            for k, coeff in enumerate(witness):
                if quotient and coeff:
                    row[k] -= quotient * coeff
        if any(row):
            relations.append(tuple(row))
    # Schreyer S-pair relations generate Syz(G); map them through G=A*F.
    for i, left in enumerate(basis):
        for j in range(i):
            right = basis[j]
            al, ar = tuple(left.LM), tuple(right.LM)
            lcm = tuple(max(a, b) for a, b in zip(al, ar, strict=True))
            ml = _monomial(
                ring, (a - b for a, b in zip(lcm, al, strict=True)), ring.domain.one / left.LC
            )
            mr = _monomial(
                ring, (a - b for a, b in zip(lcm, ar, strict=True)), ring.domain.one / right.LC
            )
            quotients, remainder = (ml * left - mr * right).div(basis)
            if remainder:
                raise AssertionError("certified S-pair did not reduce to zero")
            row = [ml * a - mr * b for a, b in zip(memberships[i], memberships[j], strict=True)]
            for q, witness in zip_longest(quotients, memberships, fillvalue=None):
                if q is None:
                    q = ring.zero
                if q:
                    row = [a - q * b for a, b in zip(row, witness, strict=True)]
            if any(row):
                relations.append(tuple(row))
    # Verify each relation so even a faulty Schreyer lift cannot be trusted.
    for row in relations:
        if sum((a * f for a, f in zip(row, inputs, strict=True)), ring.zero):
            raise AssertionError("invalid syzygy generator")
    return tuple(relations)


def syzygy_groebner_basis(inputs, basis, memberships, ring, *, limits=None):
    """Fully certified exact reduced syzygy-module Groebner basis."""
    if limits is None:
        limits = SyzygyLimits()
    relations = syzygy_generators(inputs, basis, memberships, ring)
    return _module_groebner(relations, ring, limits)


def canonicalize_syzygy_witnesses(witness_rows, inputs, basis, ring, *, limits=None):
    """Unique POT normal forms of proof rows modulo *all* syzygies of inputs.

    Requires independently certified basis/witnesses. Canonicality is with
    respect to the input order, ring monomial order and specified POT module
    order. Fails closed on resource exhaustion. No length-decreasing heuristic.
    """
    if limits is None:
        limits = SyzygyLimits()
    rows = tuple(tuple(row) for row in witness_rows)
    if len(rows) != len(basis):
        raise ValueError("one witness row is needed for every basis element")
    module = syzygy_groebner_basis(inputs, basis, rows, ring, limits=limits)
    reduced = tuple(_normal_form(row, module, ring, limits) for row in rows)
    if any(
        sum((a * f for a, f in zip(row, inputs, strict=True)), ring.zero) != g
        for row, g in zip(reduced, basis, strict=True)
    ):
        raise AssertionError("module reduction changed an exact witness")
    return reduced, module


__all__ = [
    "SyzygyLimits",
    "SyzygyNormalFormLimitError",
    "syzygy_groebner_basis",
    "syzygy_generators",
    "canonicalize_syzygy_witnesses",
]
