"""Shared exact-input and independent verification helpers for benchmarks."""

import hashlib
from math import isqrt

from gb_case_catalog import GBCase
from gb_published_additions import exact_lift_primes


def fingerprint(polynomials: list[dict]) -> str:
    canonical = sorted(tuple(sorted((tuple(k), int(v)) for k, v in p.items())) for p in polynomials)
    return hashlib.sha256(repr(canonical).encode("ascii")).hexdigest()


def load_native():
    from gb_native import load_native as load_kernel

    return load_kernel()


def verify_sympy(case: GBCase, p: int, order: str, basis: list[dict]) -> bool:
    """Independent engine: SymPy's groebnertools, run AFTER native measurement."""
    import sympy as sp
    from sympy.polys.groebnertools import groebner
    from sympy.polys.rings import PolyRing

    variables = sp.symbols(f"x0:{case.nvars}")
    ring = PolyRing(variables, sp.GF(p), order)
    generators = [
        ring.from_dict({m: c % p for m, c in poly.items() if c % p}) for poly in case.polynomials
    ]
    # A generator may vanish in the selected characteristic.  SymPy's
    # internal Buchberger routine does not discard zero polynomial inputs.
    # Zero generators do not change the ideal, so filter them for the oracle.
    generators = [g for g in generators if g]
    independent = groebner(generators, ring) if generators else []
    reference = [dict((tuple(m), int(c) % p) for m, c in poly.terms()) for poly in independent]
    return fingerprint(reference) == fingerprint(basis)


def is_prime(value: int) -> bool:
    return value >= 2 and all(value % d != 0 for d in range(2, isqrt(value) + 1))


def _validate_exact_lift(case_name: str, prime: int) -> None:
    """Reject reductions that change the original rational problem."""
    if case_name == "caprasse_modified" and prime in (2, 5):
        raise ValueError("Modified Caprasse denominators are not invertible modulo 2 or 5")
    if case_name == "wood_rational" and prime in (2, 5):
        raise ValueError("Wood rational-lift denominators are not invertible modulo 2 or 5")
    if case_name == "posso_butcher" and prime in (2, 3):
        raise ValueError("Butcher input denominators are not invertible modulo 2 or 3")
    if prime in exact_lift_primes(case_name):
        raise ValueError(f"{case_name} source denominators are not invertible modulo {prime}")
