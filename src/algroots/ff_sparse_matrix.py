"""Optional native GF(p) sparse matrices and compiled polynomial-multiple assembly.

Arithmetic results are always untrusted candidates for subsequent exact QQ proof
verification. Replayed row operations are constructed within each prime.
"""

try:
    from ._ff_sparse import batched_spairs as _compiled_spairs
    from ._ff_sparse import reduce_rows as _compiled_reduce
    from ._ff_sparse import shifted_rows as _compiled_shift
except ImportError:
    _compiled_reduce = None
    _compiled_shift = None
    _compiled_spairs = None


def compiled_available():
    """Return whether the optional compiled arithmetic extension is importable."""
    return _compiled_reduce is not None


def _python_reduce(rows, ncols, prime):
    pivots = {}
    output = []
    for i, original in enumerate(rows):
        current = {int(k): int(v) % prime for k, v in original.items() if int(v) % prime}
        if any(k < 0 or k >= ncols for k in current):
            raise ValueError("sparse row column out of bounds")
        operations = []
        while True:
            eligible = [column for column in current if column in pivots]
            if not eligible:
                break
            column = min(eligible)
            coefficient = current[column]
            pivot_index, row = pivots[column]
            operations.append((pivot_index, coefficient))
            for index, value in row.items():
                updated = (current.get(index, 0) - coefficient * value) % prime
                if updated:
                    current[index] = updated
                else:
                    current.pop(index, None)
        if current:
            leader = min(current)
            inverse = pow(current[leader], -1, prime)
            current = {k: v * inverse % prime for k, v in current.items()}
            pivots[leader] = (i, current)
        else:
            inverse = 1
        output.append((current, operations, inverse))
    return output


def reduce_rows(rows, ncols, prime, *, backend="auto", active_columns="adaptive"):
    """Reduce with pivot traces; native mode is dense, sparse, or adaptive.

    The caller must provide a *prime* modulus. All native values are still
    independently verified by the rational Gröbner certificate layer.
    """
    if backend not in ("auto", "compiled", "python"):
        raise ValueError("matrix backend must be auto, compiled or python")
    if active_columns not in ("adaptive", "dense", "sparse"):
        raise ValueError("active-column mode must be adaptive, dense or sparse")
    if ncols < 0 or prime < 2:
        raise ValueError("invalid finite-field matrix dimensions")
    if backend == "compiled" and (_compiled_reduce is None or prime > 2**30):
        raise RuntimeError("compiled GF(p) reducer unavailable for requested modulus")
    if backend != "python" and _compiled_reduce is not None and prime <= 2**30:
        return _compiled_reduce(rows, ncols, prime, active_columns)
    return _python_reduce(rows, ncols, prime)


def assemble_shifted_rows(basis_terms, reducer_specs, positions, prime, *, backend="auto"):
    """Map monomial multiples directly into sparse matrix columns.

    ``basis_terms`` contains tuples of ``(exponent_tuple, integer_residue)``.
    Every resulting row is recomputed in the current prime, with exact
    symbolic coverage supplied by the caller's validated ``positions`` map.
    """
    if backend not in ("auto", "compiled", "python"):
        raise ValueError("matrix backend must be auto, compiled or python")
    if backend == "compiled" and (_compiled_shift is None or prime > 2**30):
        raise RuntimeError("compiled shifted-row assembler unavailable")
    if backend != "python" and _compiled_shift is not None and prime <= 2**30:
        return _compiled_shift(basis_terms, reducer_specs, positions, prime)
    output = []
    for shift, index in reducer_specs:
        row = {}
        for powers, value in basis_terms[index]:
            exponent = tuple(a + b for a, b in zip(powers, shift, strict=True))
            col = positions[exponent]
            residue = int(value) % prime
            if residue:
                row[col] = residue
        output.append(row)
    return output


__all__ = ["reduce_rows", "compiled_available", "assemble_shifted_rows"]


def batch_spolynomials(basis_terms, descriptors, prime, *, backend="auto"):
    """Compiled batched shifted-support arithmetic, with exact Python fallback.

    The descriptors contain (left-index, right-index, left-shift,
    right-shift, inverse-leading-coefficient-left, inverse-right).
    """
    if backend not in ("auto", "python", "compiled"):
        raise ValueError("invalid S-polynomial batch backend")
    if backend == "compiled" and (_compiled_spairs is None or prime > 2**30):
        raise RuntimeError("compiled S-polynomial assembly is unavailable")
    if backend != "python" and _compiled_spairs is not None and prime <= 2**30:
        return _compiled_spairs(basis_terms, descriptors, prime)
    output = []
    for i, j, left_shift, right_shift, left_inv, right_inv in descriptors:
        result = {}
        for which, (index, shift, inverse) in enumerate(
            ((i, left_shift, left_inv), (j, right_shift, right_inv))
        ):
            sign = 1 if which == 0 else -1
            for power, coeff in basis_terms[index].items():
                exponent = tuple(a + b for a, b in zip(power, shift, strict=True))
                value = (result.get(exponent, 0) + sign * int(coeff) * inverse) % prime
                if value:
                    result[exponent] = value
                else:
                    result.pop(exponent, None)
        output.append(result)
    return output
