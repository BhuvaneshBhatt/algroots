"""Exact streaming verification of sparse polynomial-combination certificates.

Witness coefficients are sparse dictionaries, never polynomial-ring objects.
The verifier merges sorted shifted copies of the original generators and
compares each resulting coefficient with the candidate, without materializing
an intermediate polynomial or a polynomial witness coefficient.

This verifies ideal inclusion only; callers must *also* check reverse inclusion,
Buchberger's criterion and reducedness with an independent algorithm.
"""

from heapq import heappop, heappush

from sympy.polys.polyerrors import CoercionFailed


def sparse_membership_rows(witness_rows, modulus, scales, *, reconstruct):
    """Reconstruct exact rational witness dictionaries from CRT data.

    ``reconstruct`` is the supplied rational-reconstruction function. Failure
    to reconstruct any coefficient rejects the *whole* candidate.
    """
    rows = []
    for row in witness_rows:
        coefficients = []
        for terms, scale in zip(row, scales, strict=True):
            out = {}
            for powers, residue in terms.items():
                value = reconstruct(residue, modulus)
                if value is None:
                    return None
                value *= scale
                if value:
                    out[tuple(powers)] = value
            coefficients.append(out)
        rows.append(tuple(coefficients))
    return tuple(rows)


def expand_sparse_rows(rows, ring):
    """Compatibility conversion to the traditional, expanded witness matrix."""
    return tuple(tuple(ring.from_dict(dict(coefficient)) for coefficient in row) for row in rows)


def _terms(poly):
    return tuple(
        sorted(
            ((tuple(powers), coefficient) for powers, coefficient in poly.terms() if coefficient),
            key=lambda pair: pair[0],
        )
    )


def verify_sparse_memberships(original, candidate, rows, ring, *, max_streams=None):
    """Independently replay exact QQ generator-combination identities.

    For each nonzero monomial in a coefficient q_j, a stream emits the shifted
    support of f_j, ordered lexicographically. A heap merges these streams and
    combines coefficients with identical exponents. Comparison against the
    sparse candidate detects missing and extra terms without constructing
    q_j, q_j*f_j, or their sum as polynomial-ring elements.

    The heap stores O(number of nonzero witness terms) entries; the algorithm
    uses exact rational arithmetic and accepts no probabilistic equality test.
    """
    try:
        if len(rows) != len(candidate):
            return False
        if not getattr(ring.domain, "is_Field", False):
            return False
        num_vars = len(ring.symbols)
        domain = ring.domain
        originals = tuple(_terms(poly) for poly in original)
        for polynomial, row in zip(candidate, rows, strict=True):
            if len(row) != len(original):
                return False
            # Entries: (exponent, stream ID, value).  The stream state is
            # (shift, scalar, generator-terms, next term index).
            heap = []
            states = []
            for j, coefficient in enumerate(row):
                if not hasattr(coefficient, "items"):
                    return False
                support = originals[j]
                for power, scalar in coefficient.items():
                    shift = tuple(power)
                    if len(shift) != num_vars or any(
                        not isinstance(e, int) or e < 0 for e in shift
                    ):
                        return False
                    scalar = domain.convert(scalar)
                    if not scalar or not support:
                        continue
                    if max_streams is not None and len(states) >= max_streams:
                        raise ValueError("sparse membership stream budget exceeded")
                    index = len(states)
                    states.append((shift, scalar, support))
                    first_powers, first_scalar = support[0]
                    heappush(
                        heap,
                        (
                            tuple(a + b for a, b in zip(shift, first_powers, strict=True)),
                            index,
                            0,
                            scalar * first_scalar,
                        ),
                    )
            expected = _terms(polynomial)
            target_index = 0
            while heap:
                power = heap[0][0]
                total = domain.zero
                while heap and heap[0][0] == power:
                    _, index, offset, value = heappop(heap)
                    total += value
                    shift, scalar, support = states[index]
                    offset += 1
                    if offset < len(support):
                        powers, coefficient = support[offset]
                        heappush(
                            heap,
                            (
                                tuple(a + b for a, b in zip(shift, powers, strict=True)),
                                index,
                                offset,
                                scalar * coefficient,
                            ),
                        )
                if total:
                    if target_index >= len(expected) or expected[target_index] != (power, total):
                        return False
                    target_index += 1
            if target_index != len(expected):
                return False
        return True
    except (
        TypeError,
        ValueError,
        AttributeError,
        ArithmeticError,
        ZeroDivisionError,
        OverflowError,
        CoercionFailed,
    ):
        return False


def verify_sparse_memberships_accumulate(
    original, candidate, rows, ring, *, max_accumulated_terms=None
):
    """Exact term-stream replay with sparse cancellation accumulator.

    This variant avoids polynomial-ring objects for witnesses and their
    products, but may keep a sparse dictionary for a full output polynomial.
    Prefer it when direct hash accumulation is cheaper than heap merging.
    """
    try:
        if len(rows) != len(candidate):
            return False
        zero = ring.domain.zero
        length = len(ring.symbols)
        generators = tuple(poly.terms() for poly in original)
        for expected, row in zip(candidate, rows, strict=True):
            if len(row) != len(original):
                return False
            accumulator = {}
            for factor, terms in zip(row, generators, strict=True):
                if not hasattr(factor, "items"):
                    return False
                for shift, scalar in factor.items():
                    shift = tuple(shift)
                    if len(shift) != length or any(not isinstance(a, int) or a < 0 for a in shift):
                        return False
                    scalar = ring.domain.convert(scalar)
                    if scalar:
                        for powers, coefficient in terms:
                            monomial = tuple(a + b for a, b in zip(shift, powers, strict=True))
                            value = accumulator.get(monomial, zero) + scalar * coefficient
                            if value:
                                accumulator[monomial] = value
                            else:
                                accumulator.pop(monomial, None)
                        if (
                            max_accumulated_terms is not None
                            and len(accumulator) > max_accumulated_terms
                        ):
                            return False
            if accumulator != dict(expected.terms()):
                return False
        return True
    except (
        TypeError,
        ValueError,
        AttributeError,
        ArithmeticError,
        ZeroDivisionError,
        CoercionFailed,
    ):
        return False


__all__ = [
    "sparse_membership_rows",
    "expand_sparse_rows",
    "verify_sparse_memberships",
    "verify_sparse_memberships_accumulate",
]
