"""Independent, deliberately simple finite-field Gröbner-basis checks.

No native F4 code or FLINT is imported. Independent from SymPy as well,
except for the optional *reverse containment* oracle in full_certificate().

Two-sided ideal equality is NOT established by checking S-polynomials and
reducing input generators alone. A reduced Gröbner basis of a strict *larger*
ideal could pass those necessary checks. For full certification we additionally
establish every output generator lies in the input ideal, using a separately
computed SymPy reference Gröbner basis. Oracle work is excluded from timing.
"""

from __future__ import annotations

from itertools import combinations_with_replacement


def monomial_key(m, order):
    if order == "lex":
        return tuple(m)
    if order == "grlex":
        return (sum(m), *m)
    if order == "grevlex":
        return (sum(m), *(-v for v in reversed(m)))
    raise ValueError(f"Unsupported order {order}")


def divides(a, b):
    return all(x <= y for x, y in zip(a, b, strict=False))


def leading(poly, order):
    if not poly:
        raise ValueError("Zero polynomial has no leading term")
    return max(poly, key=lambda m: monomial_key(m, order))


def normalize(poly, p):
    return {tuple(m): int(c) % p for m, c in poly.items() if int(c) % p}


def normal_form(poly, basis, p, order, *, max_steps=1_000_000):
    """Exact multivariate division in GF(p), independent of native reduction."""
    todo = normalize(poly, p)
    reference = [normalize(b, p) for b in basis if b]
    leaders = [(leading(b, order), b) for b in reference]
    remainder = {}
    steps = 0
    while todo:
        steps += 1
        if steps > max_steps:
            raise RuntimeError("Independent normal-form step budget exceeded")
        m = leading(todo, order)
        c = todo[m]
        reducer = next(((lm, b) for lm, b in leaders if divides(lm, m)), None)
        if reducer is None:
            remainder[m] = c
            del todo[m]
            continue
        lm, b = reducer
        multiplier = c * pow(b[lm], -1, p) % p
        shift = tuple(e - f for e, f in zip(m, lm, strict=False))
        for t, coefficient in b.items():
            monomial = tuple(a + d for a, d in zip(shift, t, strict=False))
            v = (todo.get(monomial, 0) - multiplier * coefficient) % p
            if v:
                todo[monomial] = v
            else:
                todo.pop(monomial, None)
    return remainder


def s_polynomial(f, g, p, order):
    a, b = leading(f, order), leading(g, order)
    monomial = tuple(max(x, y) for x, y in zip(a, b, strict=False))
    fa, gb = pow(f[a], -1, p), pow(g[b], -1, p)
    result = {}
    for poly, lm, scale in ((f, a, fa), (g, b, -gb)):
        shift = tuple(x - y for x, y in zip(monomial, lm, strict=False))
        for term, c in poly.items():
            key = tuple(x + y for x, y in zip(term, shift, strict=False))
            value = (result.get(key, 0) + c * scale) % p
            if value:
                result[key] = value
            else:
                result.pop(key, None)
    return result


def structural_checks(case, p, order, basis, *, max_spairs=6000, max_steps=1_000_000):
    """Necessary tests + Buchberger criterion; never claims ideal equality."""
    b = [normalize(g, p) for g in basis]
    if not b or not all(b):
        return dict(
            monic_reduced=False,
            generators_reduce_to_zero=False,
            s_pairs_reduce_to_zero=False,
            checked_spairs=0,
        )
    leading_monomials = [leading(g, order) for g in b]
    monic = all(g[lm] == 1 for g, lm in zip(b, leading_monomials, strict=False))
    reduced = (
        monic
        and all(
            not divides(other, term)
            for i, g in enumerate(b)
            for term in g
            for j, other in enumerate(leading_monomials)
            if i != j
        )
        and all(
            not divides(lm, term)
            for g, lm in zip(b, leading_monomials, strict=False)
            for term in g
            if term != lm
        )
    )
    # Note: the second check also rules out nonleading terms divisible by the
    # polynomial's own leading monomial (impossible for a valid term order).
    inputs_reduce = all(
        not normal_form(g, b, p, order, max_steps=max_steps) for g in case.polynomials
    )
    # Independent Buchberger checker. Product criterion and certified chain
    # dependencies avoid redundant explicit polynomial divisions; each chain
    # skip has TWO already-proven antecedents, never a speculative assumption.
    from gb_witness_certificate import efficient_spair_checks

    pairs = efficient_spair_checks(b, p, order, max_pairs=max_spairs, max_steps=max_steps)
    return dict(
        monic_reduced=reduced,
        generators_reduce_to_zero=inputs_reduce,
        s_pairs_reduce_to_zero=pairs["ok"],
        checked_spairs=pairs["pairs_checked"],
        product_skipped_spairs=pairs["product_pairs"],
        chain_skipped_spairs=pairs["chain_pairs"],
        total_spairs=pairs["total_pairs"],
    )


def full_certificate(case, p, order, basis, *, max_spairs=6000, max_steps=1_000_000):
    """Certified two-sided containment using an independent SymPy input GB."""
    result = structural_checks(case, p, order, basis, max_spairs=max_spairs, max_steps=max_steps)
    import sympy as sp
    from sympy.polys.groebnertools import groebner
    from sympy.polys.rings import PolyRing

    ring = PolyRing(sp.symbols(f"x0:{case.nvars}"), sp.GF(p), order)
    inputs = [ring.from_dict(normalize(g, p)) for g in case.polynomials]
    reference = groebner([g for g in inputs if g], ring) if any(inputs) else []
    reference_dicts = [normalize(dict(g.terms()), p) for g in reference]
    result["output_in_original_ideal"] = all(
        not normal_form(g, reference_dicts, p, order, max_steps=max_steps) for g in basis
    )
    # Log logically distinct results, so no caller interprets the strict
    # forward-membership check as complete ideal equality.
    result["full_certificate_ok"] = (
        result["monic_reduced"]
        and result["generators_reduce_to_zero"]
        and result["s_pairs_reduce_to_zero"]
        and result["output_in_original_ideal"]
    )
    return result


def standard_monomial_hilbert_values(basis, nvars, order, max_degree=4):
    """Graded Hilbert function from leading-monomial ideal (homogeneous I)."""
    leaders = [leading(g, order) for g in basis if g]
    values = []
    for degree in range(max_degree + 1):
        count = 0
        for indices in combinations_with_replacement(range(nvars), degree):
            exponent = tuple(indices.count(i) for i in range(nvars))
            if not any(divides(m, exponent) for m in leaders):
                count += 1
        values.append(count)
    return values


def dimension_of_artinian_quotient(basis, nvars, order):
    """Standard-monomial count if pure powers bound every variable."""
    leaders = [leading(g, order) for g in basis if g]
    bounds = []
    for i in range(nvars):
        values = [m[i] for m in leaders if m[i] and sum(m) == m[i]]
        if not values:
            return None
        bounds.append(min(values))
    from itertools import product

    return sum(
        not any(divides(m, exponent) for m in leaders)
        for exponent in product(*(range(x) for x in bounds))
    )
