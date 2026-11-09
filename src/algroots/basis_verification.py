"""Independent exact checks for rational bases and membership witnesses."""

from time import perf_counter

from sympy.polys.groebnertools import groebner as reference_groebner


def _divides(left, right):
    return all(a <= b for a, b in zip(left, right, strict=True))


def verify_rational_basis(
    original,
    basis,
    ring,
    *,
    membership_witnesses=None,
    sparse_memberships=None,
    sparse_strategy="accumulate",
    profile=None,
    rational_normal_form_backend="sympy",
    rational_dag=None,
):
    """Independent verification; does not assume CRT or modular correctness.

    With explicit witnesses, no Gröbner basis is recomputed over QQ.
    The original reference fallback is available for callers without proofs.
    """
    verification_start = perf_counter()
    if rational_normal_form_backend not in ("sympy", "flint"):
        raise ValueError("unknown rational normal-form backend")
    qq_reducer = None
    if rational_normal_form_backend == "flint" and basis:
        from .qq_normal_form import FLINTQQNormalForm

        qq_reducer = FLINTQQNormalForm(basis, ring)
    if sum(x is not None for x in (membership_witnesses, sparse_memberships, rational_dag)) > 1:
        return False
    if any(not poly or ring.domain.one != poly.LC for poly in basis):
        return False
    if not basis:
        return (
            not any(original)
            and (membership_witnesses is None or not membership_witnesses)
            and (sparse_memberships is None or not sparse_memberships)
        )
    leading = tuple(tuple(poly.LM) for poly in basis)
    if len(set(leading)) != len(leading):
        return False
    for index, poly in enumerate(basis):
        if any(
            _divides(lm, powers)
            for powers, _ in poly.terms()
            for j, lm in enumerate(leading)
            if j != index
        ):
            return False
    reduction_start = perf_counter()
    if any(
        (qq_reducer.remainder(poly) if qq_reducer is not None else poly.rem(basis))
        for poly in original
    ):
        return False
    if profile is not None:
        profile["verify_input_normal_forms"] = (
            profile.get("verify_input_normal_forms", 0.0) + perf_counter() - reduction_start
        )
    membership_start = perf_counter()
    if rational_dag is not None:
        if qq_reducer is not None:
            dag_verified = qq_reducer.verify_rational_dag(rational_dag, original, basis)
        else:
            from .compact_rational_proof import verify_rational_dag

            dag_verified = verify_rational_dag(rational_dag, original, basis, ring)
        if not dag_verified:
            return False
    elif sparse_memberships is not None:
        from .rational_certificate import (
            verify_sparse_memberships,
            verify_sparse_memberships_accumulate,
        )

        if sparse_strategy not in ("accumulate", "heap"):
            return False
        verifier = (
            verify_sparse_memberships_accumulate
            if sparse_strategy == "accumulate"
            else verify_sparse_memberships
        )
        if qq_reducer is not None:
            if not qq_reducer.verify_sparse_memberships(original, basis, sparse_memberships):
                return False
        elif not verifier(original, basis, sparse_memberships, ring):
            return False
    elif membership_witnesses is None:
        reference = tuple(reference_groebner(original, ring))
        if any(poly.rem(reference) for poly in basis):
            return False
    else:
        if len(membership_witnesses) != len(basis):
            return False
        if qq_reducer is not None:
            if not qq_reducer.verify_expanded_memberships(original, basis, membership_witnesses):
                return False
        else:
            for polynomial, factors in zip(basis, membership_witnesses, strict=True):
                if len(factors) != len(original):
                    return False
                if (
                    sum((q * f for q, f in zip(factors, original, strict=True)), ring.zero)
                    != polynomial
                ):
                    return False
    if profile is not None:
        profile["verify_membership_identities"] = (
            profile.get("verify_membership_identities", 0.0) + perf_counter() - membership_start
        )
    spair_start = perf_counter()
    checked = skipped = 0
    for i, left in enumerate(basis):
        for j, right in enumerate(basis[:i]):
            first, second = tuple(left.LM), tuple(right.LM)
            common = tuple(max(a, b) for a, b in zip(first, second, strict=True))
            # Buchberger's product criterion: relatively prime leading
            # monomials have S-polynomial reducing to zero identically.
            if all(a + b == c for a, b, c in zip(first, second, common, strict=True)):
                skipped += 1
                continue
            checked += 1
            if qq_reducer is not None:
                if qq_reducer.s_pair_remainder(i, j):
                    return False
            else:
                factor_left = ring.from_dict(
                    {tuple(a - b for a, b in zip(common, first, strict=True)): 1}
                )
                factor_right = ring.from_dict(
                    {tuple(a - b for a, b in zip(common, second, strict=True)): 1}
                )
                if (factor_left * left - factor_right * right).rem(basis):
                    return False
    if profile is not None:
        profile["verify_spair_normal_forms"] = (
            profile.get("verify_spair_normal_forms", 0.0) + perf_counter() - spair_start
        )
        profile["verify_spair_checked"] = profile.get("verify_spair_checked", 0) + checked
        profile["verify_spair_product_skipped"] = (
            profile.get("verify_spair_product_skipped", 0) + skipped
        )
        profile["verify_total"] = (
            profile.get("verify_total", 0.0) + perf_counter() - verification_start
        )
    return True
