"""Proof-producing finite-field Buchberger arithmetic for modular reconstruction.

This module deliberately depends only on SymPy's polynomial-ring operations,
not a private native-FLINT interface. Each basis polynomial has a tracked
representation in the *ordered* input generators. The resulting finite-field
identities are NOT certificates over QQ until reconstructed and replayed there.
"""

from heapq import heappop, heappush


class ModularProofLimitError(ArithmeticError):
    """A finite-field proof trace exceeded an explicit resource budget."""


def _monomial(ring, powers):
    return ring.from_dict({tuple(powers): ring.domain.one})


def _normalize(polynomial, row, ring):
    inverse = ring.domain.one / polynomial.LC
    return polynomial * inverse, tuple(value * inverse for value in row)


def _reduce_with_trace(polynomial, row, basis, representations, ring):
    if not basis or not polynomial:
        return polynomial, row
    quotients, remainder = polynomial.div(basis)
    result = list(row)
    for quotient, previous in zip(quotients, representations, strict=True):
        if quotient:
            result = [a - quotient * b for a, b in zip(result, previous, strict=True)]
    return remainder, tuple(result)


def _autoreduce_tracked(basis, representations, ring, *, max_passes):
    """Reduce a finite-field basis while preserving its exact representations."""
    pending = list(zip(basis, representations, strict=True))
    for _ in range(max_passes):
        output = []
        for index, (poly, row) in enumerate(pending):
            others = pending[:index] + pending[index + 1 :]
            remainder, new_row = _reduce_with_trace(
                poly, row, [g for g, _ in others], [r for _, r in others], ring
            )
            if remainder:
                remainder, new_row = _normalize(remainder, new_row, ring)
                if all(remainder != previous for previous, _ in output):
                    output.append((remainder, new_row))
        output.sort(key=lambda item: ring.order(tuple(item[0].LM)), reverse=True)
        if [g for g, _ in output] == [g for g, _ in pending]:
            return tuple(g for g, _ in output), tuple(r for _, r in output)
        pending = output
    raise ModularProofLimitError("tracked finite-field autoreduction exceeded its limit")


def modular_basis_with_witnesses(
    integer_polynomials,
    ring,
    prime,
    *,
    max_basis=128,
    max_pairs=1000,
    max_terms=20000,
    lex_via_fglm=False,
):
    """Compute reduced GF(p) basis and polynomial-combination rows.

    ``integer_polynomials`` is an ordered tuple of exponent->integer dicts.
    Each output row has the same number of entries and proves an identity
    modulo ``prime``. Inputs are assumed primitive, with nonzero images.
    """
    import sympy as sp

    if lex_via_fglm:
        if str(ring.order) != "lex":
            raise ValueError("modular FGLM requires a lexicographic target")
        from sympy.polys.fglmtools import matrix_fglm

        grevlex_ring = ring.clone(order="grevlex")
        grevlex_basis, grevlex_rows = modular_basis_with_witnesses(
            integer_polynomials,
            grevlex_ring,
            prime,
            max_basis=max_basis,
            max_pairs=max_pairs,
            max_terms=max_terms,
        )
        grevlex_field = grevlex_ring.clone(domain=sp.GF(prime))
        field = ring.clone(domain=sp.GF(prime))
        grevlex_polys = tuple(grevlex_field.from_dict(g) for g in grevlex_basis)
        is_zero_dimensional = (
            len(grevlex_polys) == 1 and grevlex_polys[0] == grevlex_field.one
        ) or all(
            any(
                lm[index] > 0 and all(degree == 0 for j, degree in enumerate(lm) if j != index)
                for lm in (tuple(g.LM) for g in grevlex_polys)
            )
            for index in range(len(ring.symbols))
        )
        if is_zero_dimensional:
            if grevlex_polys == (grevlex_field.one,):
                target_polys = (field.one,)
            else:
                target_polys = tuple(matrix_fglm(grevlex_polys, grevlex_field, field.order))
            result_rows = []
            for target in target_polys:
                quotients, remainder = target.set_ring(grevlex_field).div(grevlex_polys)
                if remainder:
                    raise ModularProofLimitError(
                        "FGLM conversion failed to lift a membership identity"
                    )
                row = [grevlex_field.zero for _ in integer_polynomials]
                for quotient, previous in zip(quotients, grevlex_rows, strict=True):
                    for index, factor in enumerate(previous):
                        if factor:
                            row[index] += quotient * grevlex_field.from_dict(factor)
                result_rows.append(tuple(row))
            if sum(len(value) for row in result_rows for value in row) > max_terms:
                raise ModularProofLimitError("FGLM witness lifting exceeded term budget")

            def residues(poly):
                return {
                    tuple(power): int(coefficient) % prime
                    for power, coefficient in poly.terms()
                    if int(coefficient) % prime
                }

            return (
                tuple(residues(poly) for poly in target_polys),
                tuple(tuple(residues(poly) for poly in row) for row in result_rows),
            )
        # An ideal with positive-dimensional grevlex leading ideal cannot be
        # converted by FGLM. The direct lex trace still works for such ideals.

    field = ring.clone(domain=sp.GF(prime))
    inputs = tuple(
        field.from_dict(
            {
                powers: coefficient % prime
                for powers, coefficient in poly.items()
                if coefficient % prime
            }
        )
        for poly in integer_polynomials
    )
    if any(not poly for poly in inputs):
        raise ModularProofLimitError("a primitive input vanished in this characteristic")
    n = len(inputs)
    if not n:
        return (), ()
    basis, representations = [], []
    pending, serial = [], 0
    zero = field.zero

    def push_pair(i, j):
        nonlocal serial
        first, second = tuple(basis[i].LM), tuple(basis[j].LM)
        least = tuple(max(a, b) for a, b in zip(first, second, strict=True))
        if all(a + b == c for a, b, c in zip(first, second, least, strict=True)):
            return
        heappush(pending, (sum(least), serial, i, j))
        serial += 1

    def append(poly, row):
        if len(basis) >= max_basis:
            raise ModularProofLimitError("finite-field proof basis size exceeded limit")
        if sum(len(factor) for factor in row) > max_terms:
            raise ModularProofLimitError("finite-field proof witness size exceeded limit")
        basis.append(poly)
        representations.append(row)
        newest = len(basis) - 1
        for previous in range(newest):
            push_pair(previous, newest)

    for index, poly in enumerate(inputs):
        row = [zero for _ in range(n)]
        row[index] = field.one
        remainder, reduced = _reduce_with_trace(poly, tuple(row), basis, representations, field)
        if remainder:
            normalized, normalized_row = _normalize(remainder, reduced, field)
            append(normalized, normalized_row)
    processed = 0
    while pending:
        if processed >= max_pairs:
            raise ModularProofLimitError("finite-field proof pair budget exceeded")
        processed += 1
        _, _, i, j = heappop(pending)
        left, right = basis[i], basis[j]
        first, second = tuple(left.LM), tuple(right.LM)
        least = tuple(max(a, b) for a, b in zip(first, second, strict=True))
        mul_left = _monomial(field, (c - a for c, a in zip(least, first, strict=True)))
        mul_right = _monomial(field, (c - b for c, b in zip(least, second, strict=True)))
        s_poly = mul_left * left - mul_right * right
        s_row = tuple(
            mul_left * a - mul_right * b
            for a, b in zip(representations[i], representations[j], strict=True)
        )
        remainder, row = _reduce_with_trace(s_poly, s_row, basis, representations, field)
        if remainder:
            normalized, normalized_row = _normalize(remainder, row, field)
            append(normalized, normalized_row)
        if sum(len(poly) for poly in basis) > max_terms:
            raise ModularProofLimitError("finite-field proof term budget exceeded")
    reduced, witnesses = _autoreduce_tracked(
        basis, representations, field, max_passes=max_basis + 1
    )

    def residues(poly):
        return {
            tuple(power): int(coefficient) % prime
            for power, coefficient in poly.terms()
            if int(coefficient) % prime
        }

    return (
        tuple(residues(poly) for poly in reduced),
        tuple(tuple(residues(poly) for poly in row) for row in witnesses),
    )


__all__ = ["ModularProofLimitError", "modular_basis_with_witnesses"]
