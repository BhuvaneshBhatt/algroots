"""Exact coefficient normalization, CRT and rational reconstruction."""

from math import gcd, isqrt, lcm

import sympy as sp


def rational_reconstruction(residue: int, modulus: int):
    """Recover unique small a/b with b invertible mod M, or return None.

    Uses the symmetric sqrt(M/2) numerator and denominator bound. This is
    *not* a proof of the result's suitability as a Gröbner coefficient.
    """
    if modulus <= 1:
        raise ValueError("modulus must exceed one")
    residue %= modulus
    if residue == 0:
        return sp.S.Zero
    bound = isqrt(modulus // 2)
    previous_remainder, remainder = modulus, residue
    previous_denominator, denominator = 0, 1
    while abs(remainder) > bound and remainder:
        quotient = previous_remainder // remainder
        previous_remainder, remainder = remainder, previous_remainder - quotient * remainder
        previous_denominator, denominator = (
            denominator,
            previous_denominator - quotient * denominator,
        )
    if denominator == 0 or abs(denominator) > bound or abs(remainder) > bound:
        return None
    if denominator < 0:
        denominator, remainder = -denominator, -remainder
    if gcd(abs(remainder), denominator) != 1:
        return None
    if (residue * denominator - remainder) % modulus:
        return None
    return sp.Rational(remainder, denominator)


def _primitive_integer_polynomials_and_scales(inputs, ring):
    integer_polynomials = []
    scales = []
    for poly in inputs:
        terms = poly.terms()
        if not terms:
            continue
        denominator = lcm(*(int(ring.domain.denom(value)) for _, value in terms))
        coefficients = {
            tuple(powers): int(ring.domain.numer(value))
            * (denominator // int(ring.domain.denom(value)))
            for powers, value in terms
        }
        content = gcd(*coefficients.values())
        integer_polynomials.append({key: value // content for key, value in coefficients.items()})
        scales.append(sp.Rational(denominator, content))
    return tuple(integer_polynomials), tuple(scales)


def _crt_polynomial(old, new, old_modulus, prime):
    inverse = pow(old_modulus, -1, prime)
    return {
        monomial: previous + old_modulus * (((new.get(monomial, 0) - previous) * inverse) % prime)
        for monomial, previous in ((key, old.get(key, 0)) for key in old.keys() | new.keys())
    }
