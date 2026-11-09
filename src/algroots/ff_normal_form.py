"""Opt-in CPython GF(p) normal-form engine with a SymPy-independent interface.

The C engine computes exact arithmetic in prime fields; returned values are
untrusted candidates when used in Gröbner computations. Rational certificates
remain fully independent of this optional extension.
"""

from functools import lru_cache

try:
    from ._ff_sparse import normal_form as _native_normal_form
except ImportError:
    _native_normal_form = None


def compiled_available():
    """Return whether the optional compiled arithmetic extension is importable."""
    return _native_normal_form is not None


@lru_cache(maxsize=128)
def _is_prime(prime):
    from sympy import isprime

    return bool(isprime(prime))


def normal_form(
    polynomial,
    divisors,
    prime,
    order="grevlex",
    *,
    backend="auto",
    max_steps=200000,
    max_terms=200000,
):
    """Divide sparse polynomials modulo a prime, returning (quotients, remainder).

    Inputs and output use ``{tuple(exponents): int}`` with coefficients reduced
    modulo ``prime``. The division is deterministic for the given divisor order.
    ``prime`` is required to be prime (checked here, not by the C extension).
    ``max_steps`` bounds processed leading terms. ``max_terms`` bounds the
    current dividend and each quotient/remainder polynomial. Both backends
    enforce these limits and raise OverflowError instead of a partial result.
    """
    if backend not in ("auto", "python", "compiled"):
        raise ValueError("unknown normal-form backend")
    if order not in ("lex", "grlex", "grevlex"):
        raise ValueError("unsupported term order")
    if not isinstance(prime, int) or not (2 <= prime <= (1 << 30)):
        raise ValueError("unsupported prime")
    if not _is_prime(prime):
        raise ValueError("normal-form modulus must be prime")
    for name, value in (("max_steps", max_steps), ("max_terms", max_terms)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    divisors = tuple(divisors)
    dimensions = [len(exponent) for poly in (polynomial, *divisors) for exponent in poly]
    if not dimensions or dimensions[0] == 0:
        raise ValueError("cannot determine number of variables")
    if any(
        len(exponent) != dimensions[0]
        or any(not isinstance(power, int) or power < 0 or power > 2**31 - 1 for power in exponent)
        for poly in (polynomial, *divisors)
        for exponent in poly
    ):
        raise ValueError("invalid exponent tuples")
    if backend == "compiled" and _native_normal_form is None:
        raise RuntimeError("optional native normal-form extension unavailable")
    if _native_normal_form is not None and backend != "python":
        return _native_normal_form(polynomial, tuple(divisors), prime, order, max_steps, max_terms)
    from sympy.polys.orderings import monomial_key

    key = monomial_key(order)
    inputs = {tuple(k): v % prime for k, v in polynomial.items() if v % prime}
    gs = [{tuple(k): v % prime for k, v in d.items() if v % prime} for d in divisors]
    if any(not g for g in gs):
        raise ZeroDivisionError("zero polynomial divisor")
    leaders = [max(g, key=key) for g in gs]
    inverses = [pow(g[m], -1, prime) for g, m in zip(gs, leaders, strict=True)]
    qs = [{} for _ in gs]
    remainder = {}
    steps = 0

    def accumulate(target, monomial, coefficient):
        value = (target.get(monomial, 0) + coefficient) % prime
        if value:
            target[monomial] = value
        else:
            target.pop(monomial, None)

    while inputs:
        if steps >= max_steps or len(inputs) > max_terms:
            raise OverflowError("GF(p) division resource limit")
        steps += 1
        monomial = max(inputs, key=key)
        coefficient = inputs.pop(monomial)
        for i, leader in enumerate(leaders):
            if not all(a >= b for a, b in zip(monomial, leader, strict=True)):
                continue
            shift = tuple(a - b for a, b in zip(monomial, leader, strict=True))
            scale = coefficient * inverses[i] % prime
            accumulate(qs[i], shift, scale)
            if len(qs[i]) > max_terms:
                raise OverflowError("GF(p) division resource limit")
            for term, value in gs[i].items():
                if term == leader:
                    continue
                shifted = tuple(a + b for a, b in zip(shift, term, strict=True))
                if any(a > 2**31 - 1 for a in shifted):
                    raise OverflowError("invalid monomial shift")
                accumulate(inputs, shifted, -scale * value)
            break
        else:
            remainder[monomial] = coefficient
            if len(remainder) > max_terms:
                raise OverflowError("GF(p) division resource limit")
    return tuple(qs), remainder


__all__ = ["normal_form", "compiled_available"]
