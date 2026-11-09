"""Bounded exact generator-combination witnesses over coefficient fields."""

from heapq import heappop, heappush

from .domains import polynomial
from .groebner_errors import GroebnerLimitError


class GroebnerProofLimitError(GroebnerLimitError, ArithmeticError):
    """Exact membership witness construction exhausted its structural budget."""


def _poly(expression, ring):
    if getattr(expression, "ring", None) is ring:
        return expression
    return ring.from_dict(polynomial(expression, ring.symbols, ring.domain).rep.to_dict())


def _tracked_buchberger(generators, ring, *, max_basis, max_pairs, targets=()):
    """Priority-ordered Buchberger with exact generator representations.

    If *targets* are given, return as soon as the current (possibly
    incomplete) basis reduces every target to zero. This is sufficient for
    *membership*, and can avoid computing a full Gröbner basis. The independent
    verifier replays all returned polynomial identities. The product criterion
    discards S-pairs of relatively prime leading monomials.
    """
    count = len(generators)
    basis = []
    representations = []
    pairs = []
    serial = 0

    def append_pair(i, j):
        nonlocal serial
        left, right = tuple(basis[i].LM), tuple(basis[j].LM)
        common = tuple(max(a, b) for a, b in zip(left, right, strict=True))
        if all(a + b == c for a, b, c in zip(left, right, common, strict=True)):
            return  # Buchberger's product criterion
        heappush(pairs, (sum(common), serial, i, j))
        serial += 1

    def target_rows():
        if not targets:
            return None
        output = []
        for target in targets:
            quotients, remainder = target.div(basis)
            if remainder:
                return None
            row = [ring.zero for _ in range(count)]
            for quotient, previous in zip(quotients, representations, strict=True):
                if quotient:
                    row = [a + quotient * b for a, b in zip(row, previous, strict=True)]
            output.append(tuple(row))
        return tuple(output)

    for index, poly in enumerate(generators):
        if not poly:
            continue
        leading_inverse = ring.domain.one / poly.LC
        normalized = poly * leading_inverse
        row = [ring.zero for _ in range(count)]
        row[index] = ring.ground_new(leading_inverse)
        # Append the polynomial before constructing priority pairs.
        basis.append(normalized)
        representations.append(tuple(row))
        new_index = len(basis) - 1
        for previous in range(new_index):
            append_pair(previous, new_index)
    if len(basis) > max_basis:
        raise GroebnerProofLimitError("witness Buchberger basis count exceeded its limit")
    if targets:
        rows = target_rows()
        if rows is not None:
            return basis, representations, rows
    processed_pairs = 0
    while pairs:
        if processed_pairs >= max_pairs:
            raise GroebnerProofLimitError("witness Buchberger pair count exceeded its limit")
        processed_pairs += 1
        _, _, i, j = heappop(pairs)
        left, right = basis[i], basis[j]
        m1, m2 = tuple(left.LM), tuple(right.LM)
        multiple = tuple(max(a, b) for a, b in zip(m1, m2, strict=True))
        factor_left = ring.from_dict({tuple(m - k for m, k in zip(multiple, m1, strict=True)): 1})
        factor_right = ring.from_dict({tuple(m - k for m, k in zip(multiple, m2, strict=True)): 1})
        spolynomial = factor_left * left - factor_right * right
        quotients, remainder = spolynomial.div(basis)
        if not remainder:
            continue
        row = [
            factor_left * u - factor_right * v
            for u, v in zip(representations[i], representations[j], strict=True)
        ]
        for quotient, prior_row in zip(quotients, representations, strict=True):
            if quotient:
                row = [u - quotient * v for u, v in zip(row, prior_row, strict=True)]
        inverse = ring.domain.one / remainder.LC
        remainder *= inverse
        row = tuple(poly * inverse for poly in row)
        if len(basis) >= max_basis:
            raise GroebnerProofLimitError("witness Buchberger basis count exceeded its limit")
        basis.append(remainder)
        representations.append(row)
        new_index = len(basis) - 1
        for previous in range(new_index):
            append_pair(previous, new_index)
        if targets:
            rows = target_rows()
            if rows is not None:
                return basis, representations, rows
    if targets:
        rows = target_rows()
        if rows is None:
            raise ValueError("candidate is not in the input ideal")
        return basis, representations, rows
    return basis, representations


def construct_membership_witnesses(generators, candidate, ring, *, max_basis=128, max_pairs=500):
    """Construct exact expression witnesses for a candidate's ideal membership.

    First attempt division by input generators, which is often inexpensive.
    If it fails, run a targeted, tracked Buchberger process with leading-pair
    priority and Buchberger product criterion. Stop as soon as all requested
    membership identities are available, without completing an unnecessary
    second Gröbner basis. Proof production can still be expensive.
    ``max_basis`` and ``max_pairs`` are hard safety budgets, not mathematical
    completeness bounds; a budget failure raises without claiming a proof.
    The resulting witnesses must still be independently checked by ``verify``.
    """
    if not getattr(ring.domain, "is_Field", False):
        raise ValueError("witness production requires a coefficient field")
    if not isinstance(max_basis, int) or isinstance(max_basis, bool) or max_basis < 1:
        raise ValueError("max_basis must be a positive integer")
    if not isinstance(max_pairs, int) or isinstance(max_pairs, bool) or max_pairs < 1:
        raise ValueError("max_pairs must be a positive integer")
    original = tuple(_poly(f, ring) for f in generators)
    target = tuple(_poly(g, ring) for g in candidate)
    if not target:
        if any(original):
            raise ValueError("nonzero ideal cannot have an empty Gröbner basis")
        return ()
    if not any(original):
        raise ValueError("candidate does not belong to the zero ideal")
    initial_rows = []
    for g in target:
        quotients, remainder = g.div(list(original))
        if remainder:
            break
        initial_rows.append(tuple(quotients))
    else:
        return tuple(initial_rows)
    # A membership proof does not require the tracked basis itself to be a
    # Gröbner basis. Stop at the first intermediate basis that reduces *all*
    # targets to zero, and lift their quotient rows in the same pass.
    _, _, rows = _tracked_buchberger(
        original, ring, max_basis=max_basis, max_pairs=max_pairs, targets=target
    )
    return rows
