"""F4-style batched POT module reduction over prime fields.

The backend is not a trusted oracle: it computes candidate generators of the
syzygy module. The rational reconstruction layer independently verifies all
syzygy identities, Schreyer completeness, and the module Buchberger criterion.
"""

from .ff_sparse_matrix import reduce_rows


def _matching_reducers(leaders, queries, *, monomial_backend="auto"):
    """First eligible module leading monomial, grouped by component.

    The fixed basis order is preserved. Divisibility is computed on batched
    exponent vectors, without changing any module-order comparisons.
    """
    from .monomial_ops import first_divisors

    by_position = {}
    for i, item in enumerate(leaders):
        if item is not None:
            by_position.setdefault(item[0], []).append((i, item[1]))
    groups = {}
    for index, (position, power) in enumerate(queries):
        groups.setdefault(position, []).append((index, power))
    chosen = [-1] * len(queries)
    for position, requests in groups.items():
        available = by_position.get(position, ())
        if not available:
            continue
        ranks = first_divisors(
            [exp for _, exp in available], [exp for _, exp in requests], backend=monomial_backend
        )
        for (original, _), rank in zip(requests, ranks, strict=True):
            if rank >= 0:
                chosen[original] = available[rank][0]
    return tuple(chosen)


def _module_columns(basis, s_rows, ring, *, limit, reducer_policy="full", monomial_backend="auto"):
    from .syzygy_normal_form import _lead

    leading = [_lead(row, ring) for row in basis]
    powers = set()
    for row in s_rows:
        for position, poly in enumerate(row):
            powers.update((position, tuple(exponent)) for exponent, _ in poly.terms())
    pending = set(powers)
    visited = set()
    specifications = []

    def key(item):
        return (-item[0], ring.order(item[1]))

    set(powers)
    while pending:
        frontier = tuple(monomial for monomial in pending if monomial not in visited)
        pending.clear()
        visited.update(frontier)
        divisors = _matching_reducers(leading, frontier, monomial_backend=monomial_backend)
        for monomial, reducer in zip(frontier, divisors, strict=True):
            if reducer < 0:
                continue
            position, exponent = monomial
            shift = tuple(a - b for a, b in zip(exponent, leading[reducer][1], strict=True))
            specifications.append((shift, reducer))
            for coordinate, entry in enumerate(basis[reducer]):
                for power, _ in entry.terms():
                    item = (coordinate, tuple(a + b for a, b in zip(power, shift, strict=True)))
                    if item not in powers:
                        powers.add(item)
                        if reducer_policy == "full":
                            pending.add(item)
            if len(powers) > limit or len(specifications) > limit:
                from .syzygy_normal_form import SyzygyNormalFormLimitError

                raise SyzygyNormalFormLimitError("module F4 symbolic budget exhausted")
    specifications.sort(
        key=lambda item: key(
            (
                leading[item[1]][0],
                tuple(a + b for a, b in zip(item[0], leading[item[1]][1], strict=True)),
            )
        ),
        reverse=True,
    )
    return tuple(sorted(powers, key=key, reverse=True)), tuple(specifications)


def _valid_module_layout(layout, basis, s_rows, ring, *, reducer_policy, monomial_backend="auto"):
    """Check a cached layout by set identities, without repeating closure search.

    Only supports and module-leading monomials are reused. A missing
    reducer or missing column forces a fresh layout for this characteristic.
    """
    from .syzygy_normal_form import _lead

    try:
        columns, specs = layout

        def order_key(item):
            return (-item[0], ring.order(item[1]))

        if len(set(columns)) != len(columns):
            return False
        if tuple(sorted(columns, key=order_key, reverse=True)) != columns:
            return False
        initial = {
            (position, tuple(power))
            for row in s_rows
            for position, entry in enumerate(row)
            for power, _ in entry.terms()
        }
        leaders = tuple(_lead(row, ring) for row in basis)
        coverage = set(initial)
        assigned = {}
        for shift, index in specs:
            if not (0 <= index < len(basis)):
                return False
            position, lm, _ = leaders[index]
            monomial = (position, tuple(a + b for a, b in zip(lm, shift, strict=True)))
            if monomial in assigned or not all(x >= 0 for x in shift):
                return False
            assigned[monomial] = (shift, index)
            for coordinate, entry in enumerate(basis[index]):
                coverage.update(
                    (coordinate, tuple(a + b for a, b in zip(power, shift, strict=True)))
                    for power, _ in entry.terms()
                )
        if coverage != set(columns):
            return False
        needs = set(columns) if reducer_policy == "full" else initial
        if reducer_policy == "frontier" and set(assigned) - initial:
            return False
        queries = tuple(needs)
        ranks = _matching_reducers(leaders, queries, monomial_backend=monomial_backend)
        for (position, powers), index in zip(queries, ranks, strict=True):
            expected = (
                None
                if index < 0
                else (tuple(a - b for a, b in zip(powers, leaders[index][1], strict=True)), index)
            )
            if assigned.get((position, powers)) != expected:
                return False
        return True
    except (TypeError, ValueError, IndexError, KeyError, AttributeError):
        return False


def _as_field_row(row, lookup, prime):
    return {
        lookup[(position, tuple(powers))]: int(coeff) % prime
        for position, poly in enumerate(row)
        for powers, coeff in poly.terms()
        if int(coeff) % prime
    }


def module_f4_batch(
    pairs,
    basis,
    ring,
    *,
    max_terms=30000,
    matrix_backend="auto",
    symbolic_cache=None,
    reducer_policy="full",
    monomial_backend="auto",
):
    """Compute module S-pair candidates through sparse finite-field elimination.

    Reuses symbolic monomial columns across primes only when basis and S-pair
    supports match exactly; does not reuse any modular coefficients or ranks.
    """
    from .syzygy_normal_form import SyzygyNormalFormLimitError, _s_pair

    prime = int(ring.domain.characteristic())
    if prime < 2:
        raise ValueError("module F4 requires a prime coefficient field")
    if reducer_policy not in ("full", "frontier"):
        raise ValueError("invalid module F4 reducer policy")
    s_rows = tuple(_s_pair(basis[i], basis[j], ring) for i, j in pairs)
    supports = tuple(
        tuple(tuple(tuple(exp) for exp, _ in p.terms()) for p in row) for row in (*basis, *s_rows)
    )
    key = ("module-f4", str(ring.order), reducer_policy, supports)
    layout = None
    if symbolic_cache is not None:
        layout = symbolic_cache.get(key)
    if layout is None:
        layout = _module_columns(
            basis,
            s_rows,
            ring,
            limit=max_terms,
            reducer_policy=reducer_policy,
            monomial_backend=monomial_backend,
        )
        if symbolic_cache is not None and len(symbolic_cache) < 256:
            symbolic_cache[key] = layout
    else:
        if not _valid_module_layout(
            layout,
            basis,
            s_rows,
            ring,
            reducer_policy=reducer_policy,
            monomial_backend=monomial_backend,
        ):
            layout = _module_columns(
                basis,
                s_rows,
                ring,
                limit=max_terms,
                reducer_policy=reducer_policy,
                monomial_backend=monomial_backend,
            )
            symbolic_cache[key] = layout
    columns, specifications = layout
    if len(columns) > max_terms:
        raise SyzygyNormalFormLimitError("module F4 column budget exhausted")
    from .ff_sparse_matrix import assemble_shifted_rows

    flat_basis_terms = tuple(
        tuple(
            ((position, *tuple(powers)), int(coefficient) % prime)
            for position, entry in enumerate(row)
            for powers, coefficient in entry.terms()
        )
        for row in basis
    )
    flat_specs = tuple(((0, *shift), index) for shift, index in specifications)
    positions = {(position, *powers): column for column, (position, powers) in enumerate(columns)}
    assembled = assemble_shifted_rows(
        flat_basis_terms, flat_specs, positions, prime, backend=matrix_backend
    )
    seen = set()
    rows = []
    for row in assembled:
        fingerprint = tuple(sorted(row.items()))
        if fingerprint not in seen:
            seen.add(fingerprint)
            rows.append(row)
    lookup = {power: index for index, power in enumerate(columns)}
    s_count = len(s_rows)
    rows.extend(_as_field_row(row, lookup, prime) for row in s_rows)
    result = reduce_rows(rows, len(columns), prime, backend=matrix_backend)
    candidates = []
    for row, _, _ in result[len(rows) - s_count :]:
        if not row:
            continue
        coordinates = [{} for _ in basis[0]]
        for col, coeff in row.items():
            position, power = columns[col]
            coordinates[position][power] = coeff
        candidates.append(tuple(ring.from_dict(item) for item in coordinates))
    return tuple(candidates)


__all__ = ["module_f4_batch"]
