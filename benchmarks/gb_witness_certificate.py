"""Portable exact GF(p) ideal-membership witnesses and independent checker.

A witness for h is a vector of polynomials a_i with h = sum(a_i*f_i),
where f_i are the original input generators. Witness construction uses a small,
independent Buchberger routine WITH transformation vectors; verification uses
only sparse modular polynomial operations and does not call a CAS or native F4.
This is intentionally bounded: oversized proofs are *unverified*, not accepted.
"""

from __future__ import annotations

from itertools import combinations

from gb_exact_certificate import divides, leading, normalize, s_polynomial


def add_shifted(target, source, shift, scalar, prime):
    """In-place sparse polynomial addition, with zero coefficients removed."""
    scalar %= prime
    if not scalar:
        return
    for monomial, coefficient in source.items():
        exponent = tuple(a + b for a, b in zip(monomial, shift, strict=False))
        value = (target.get(exponent, 0) + scalar * coefficient) % prime
        if value:
            target[exponent] = value
        else:
            target.pop(exponent, None)


def divide_with_quotients(polynomial, basis, prime, order, *, max_steps=100_000):
    """Return (remainder, quotients) satisfying f=sum(q_i*g_i)+remainder."""
    current = normalize(polynomial, prime)
    reducers = [normalize(g, prime) for g in basis]
    leads = [leading(g, order) if g else None for g in reducers]
    quotient = [{} for _ in reducers]
    remainder = {}
    zero = (0,) * len(next(iter(current or next((b for b in reducers if b), {(0,): 1}))))
    for _step in range(max_steps):
        if not current:
            return remainder, quotient
        monomial = leading(current, order)
        coefficient = current[monomial]
        index = next(
            (i for i, lm in enumerate(leads) if lm is not None and divides(lm, monomial)), None
        )
        if index is None:
            remainder[monomial] = coefficient
            del current[monomial]
            continue
        lm = leads[index]
        shift = tuple(a - b for a, b in zip(monomial, lm, strict=False))
        factor = coefficient * pow(reducers[index][lm], -1, prime) % prime
        add_shifted(quotient[index], {zero: 1}, shift, factor, prime)
        add_shifted(current, reducers[index], shift, -factor, prime)
    raise RuntimeError("Bounded division exhausted its step budget")


def _lcm(a, b):
    return tuple(max(x, y) for x, y in zip(a, b, strict=False))


def efficient_spair_checks(basis, prime, order, *, max_pairs=6000, max_steps=1_000_000):
    """Exact Buchberger criterion with product and certified chain pruning.

    Chain pruning is performed only after BOTH prerequisite pairs have been
    validated, so cycles or unsupported anticipatory pair deletion are impossible.
    """
    polys = [normalize(g, prime) for g in basis]
    if any(not g for g in polys):
        return dict(ok=False, pairs_checked=0, product_pairs=0, chain_pairs=0, total_pairs=0)
    leads = [leading(g, order) for g in polys]
    pairs = sorted(
        combinations(range(len(polys)), 2),
        key=lambda ij: (sum(_lcm(leads[ij[0]], leads[ij[1]])), ij),
    )
    if len(pairs) > max_pairs:
        raise RuntimeError("Independent S-pair count exceeds verification limit")
    valid = set()
    product = chain = computed = 0
    for i, j in pairs:
        key = (i, j)
        lcm = _lcm(leads[i], leads[j])
        if all(min(a, b) == 0 for a, b in zip(leads[i], leads[j], strict=False)):
            product += 1
        elif any(
            k != i
            and k != j
            and divides(leads[k], lcm)
            and tuple(sorted((i, k))) in valid
            and tuple(sorted((j, k))) in valid
            for k in range(len(polys))
        ):
            chain += 1
        else:
            remainder, _ = divide_with_quotients(
                s_polynomial(polys[i], polys[j], prime, order),
                polys,
                prime,
                order,
                max_steps=max_steps,
            )
            computed += 1
            if remainder:
                return dict(
                    ok=False,
                    pairs_checked=computed,
                    product_pairs=product,
                    chain_pairs=chain,
                    total_pairs=len(pairs),
                    failed_pair=[i, j],
                )
        valid.add(key)
    return dict(
        ok=True,
        pairs_checked=computed,
        product_pairs=product,
        chain_pairs=chain,
        total_pairs=len(pairs),
    )


def _term_total(polys):
    return sum(len(poly) for poly in polys)


def membership_witnesses(
    inputs,
    outputs,
    prime,
    order,
    *,
    max_pairs=600,
    max_basis=100,
    max_steps=100_000,
    max_terms=60_000,
):
    """Construct output-in-input certificates by traced independent Buchberger.

    Exceeding any resource bound raises RuntimeError; it is NEVER a pass.
    """
    input_polys = [normalize(f, prime) for f in inputs]
    num_vars = len(
        next((m for p in input_polys for m in p), next((m for p in outputs for m in p), (0,)))
    )
    one = (0,) * num_vars
    basis = []
    reps = []
    for index, f in enumerate(input_polys):
        if not f:
            continue
        lead = leading(f, order)
        inv = pow(f[lead], -1, prime)
        normal = {m: c * inv % prime for m, c in f.items()}
        rep = [{} for _ in input_polys]
        rep[index] = {one: inv}
        basis.append(normal)
        reps.append(rep)
    pending = list(combinations(range(len(basis)), 2))
    count = 0
    while pending:
        if len(basis) > max_basis or len(pending) + count > max_pairs:
            raise RuntimeError("Traced Buchberger resource budget exceeded")
        i, j = pending.pop(0)
        count += 1
        a, b = leading(basis[i], order), leading(basis[j], order)
        if all(min(x, y) == 0 for x, y in zip(a, b, strict=False)):
            continue  # Buchberger product criterion
        lcm = _lcm(a, b)
        shift_i = tuple(x - y for x, y in zip(lcm, a, strict=False))
        shift_j = tuple(x - y for x, y in zip(lcm, b, strict=False))
        syzygy = [{} for _ in input_polys]
        for index in range(len(input_polys)):
            add_shifted(syzygy[index], reps[i][index], shift_i, 1, prime)
            add_shifted(syzygy[index], reps[j][index], shift_j, -1, prime)
        s_poly = s_polynomial(basis[i], basis[j], prime, order)
        remainder, quotient = divide_with_quotients(
            s_poly, basis, prime, order, max_steps=max_steps
        )
        if not remainder:
            continue
        for q, rep in zip(quotient, reps, strict=False):
            for monomial, coeff in q.items():
                for k in range(len(input_polys)):
                    add_shifted(syzygy[k], rep[k], monomial, -coeff, prime)
        lm = leading(remainder, order)
        inv = pow(remainder[lm], -1, prime)
        remainder = {m: c * inv % prime for m, c in remainder.items()}
        syzygy = [{m: c * inv % prime for m, c in r.items()} for r in syzygy]
        if sum(_term_total(row) for row in reps) + _term_total(syzygy) > max_terms:
            raise RuntimeError("Traced witness term budget exceeded")
        new_index = len(basis)
        pending.extend((i, new_index) for i in range(new_index))
        basis.append(remainder)
        reps.append(syzygy)
    witnesses = []
    for output in outputs:
        remainder, quotient = divide_with_quotients(
            output, basis, prime, order, max_steps=max_steps
        )
        if remainder:
            raise ValueError("Output polynomial is NOT in the original ideal")
        vector = [{} for _ in input_polys]
        for q, rep in zip(quotient, reps, strict=False):
            for monomial, coeff in q.items():
                for k in range(len(input_polys)):
                    add_shifted(vector[k], rep[k], monomial, coeff, prime)
        if _term_total(vector) > max_terms:
            raise RuntimeError("Output witness term budget exceeded")
        witnesses.append(vector)
    return {
        "characteristic": prime,
        "order": order,
        "original_generator_count": len(input_polys),
        "witnesses": [
            [[[list(m), c] for m, c in sorted(poly.items())] for poly in row] for row in witnesses
        ],
        "traced_basis_size": len(basis),
        "traced_spairs": count,
    }


def verify_membership_witnesses(inputs, outputs, certificate, prime, order):
    """Check explicit polynomial identities without any Groebner computation.

    False means invalid; malformed certificate raises ValueError/TypeError.
    """
    if certificate["characteristic"] != prime or certificate["order"] != order:
        return False
    if certificate["original_generator_count"] != len(inputs):
        return False
    if len(certificate["witnesses"]) != len(outputs):
        return False
    normalized_inputs = [normalize(f, prime) for f in inputs]
    nvars = len(next((m for f in inputs for m in f), next((m for f in outputs for m in f), ())))
    for output, row in zip(outputs, certificate["witnesses"], strict=False):
        if len(row) != len(normalized_inputs):
            return False
        result = {}
        for entries, f in zip(row, normalized_inputs, strict=False):
            for m, c in entries:
                if len(m) != nvars or any(not isinstance(e, int) or e < 0 for e in m):
                    return False
                add_shifted(result, f, tuple(m), int(c), prime)
        if result != normalize(output, prime):
            return False
    return True
