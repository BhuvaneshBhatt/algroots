"""Batched exact monomial primitives, optionally accelerated by CPython C.

These routines depend only on nonnegative exponent vectors; they do not use
or trust data from another prime.  The pure-Python backend is always available.
"""

try:
    from ._ff_sparse import monomial_divisors as _native_divisors
    from ._ff_sparse import monomial_pairs as _native_pairs
    from ._ff_sparse import monomial_sort as _native_sort
except ImportError:
    _native_divisors = _native_pairs = _native_sort = None


def native_available():
    """Return whether the optional compiled monomial arithmetic extension is importable."""
    return _native_divisors is not None


def _validate(leaders, terms):
    vectors = tuple(tuple(int(x) for x in exp) for exp in leaders)
    queries = tuple(tuple(int(x) for x in exp) for exp in terms)
    widths = {len(x) for x in (*vectors, *queries)}
    if len(widths) > 1 or any(e < 0 for v in (*vectors, *queries) for e in v):
        raise ValueError("inconsistent or negative monomial exponents")
    return vectors, queries


def first_divisors(leaders, terms, *, backend="auto"):
    """First leading monomial dividing each term; -1 if none."""
    if backend not in ("auto", "python", "compiled"):
        raise ValueError("invalid monomial backend")
    lm, exps = _validate(leaders, terms)
    if backend == "compiled" and _native_divisors is None:
        raise RuntimeError("compiled monomial backend unavailable")
    if backend != "python" and _native_divisors is not None:
        return tuple(_native_divisors(lm, exps))
    return tuple(
        next((i for i, m in enumerate(lm) if all(a <= b for a, b in zip(m, exp, strict=True))), -1)
        for exp in exps
    )


def pair_lcms(leaders, pairs, *, backend="auto"):
    """Return (lcm, relatively-prime) for each pair of leading monomials."""
    if backend not in ("auto", "python", "compiled"):
        raise ValueError("invalid monomial backend")
    lm, _ = _validate(leaders, ())
    ij = tuple((int(i), int(j)) for i, j in pairs)
    if any(i < 0 or j < 0 or i >= len(lm) or j >= len(lm) for i, j in ij):
        raise IndexError("invalid pair")
    if backend == "compiled" and _native_pairs is None:
        raise RuntimeError("compiled monomial backend unavailable")
    if backend != "python" and _native_pairs is not None:
        return tuple((tuple(m), bool(p)) for m, p in _native_pairs(lm, ij))
    return tuple(
        (
            tuple(max(a, b) for a, b in zip(lm[i], lm[j], strict=True)),
            all(not (a and b) for a, b in zip(lm[i], lm[j], strict=True)),
        )
        for i, j in ij
    )


def sort_monomials(terms, order, *, reverse=False, backend="auto"):
    """Exact lex, grlex, or grevlex sorting of nonnegative exponent tuples."""
    if backend not in ("auto", "python", "compiled"):
        raise ValueError("invalid monomial backend")
    order = str(order)
    if order not in ("lex", "grlex", "grevlex"):
        raise ValueError("unsupported monomial order")
    _, exps = _validate((), terms)
    if backend == "compiled" and _native_sort is None:
        raise RuntimeError("compiled monomial backend unavailable")
    if backend != "python" and _native_sort is not None:
        return tuple(tuple(x) for x in _native_sort(exps, order, bool(reverse)))

    def key(v):
        return (
            v
            if order == "lex"
            else (sum(v), v)
            if order == "grlex"
            else (sum(v), tuple(-x for x in reversed(v)))
        )

    return tuple(sorted(exps, key=key, reverse=reverse))
