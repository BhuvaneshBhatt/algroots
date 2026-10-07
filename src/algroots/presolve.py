"""Exact affine elimination; no division by variable-dependent coefficients."""

from dataclasses import dataclass

import sympy as sp


@dataclass(frozen=True)
class AffinePresolve:
    equations: tuple
    variables: tuple
    substitutions: tuple
    inconsistent: bool = False
    cancellation_probes: int = 0
    cancellation_acceptances: int = 0

    def reconstruct(self, root, original_variables, digits):
        assignment = dict(zip(self.variables, root, strict=True))
        for variable, expression in reversed(self.substitutions):
            assignment[variable] = sp.N(expression.subs(assignment), digits)
        return tuple(assignment[v] for v in original_variables)


def affine_presolve(equations, variables):
    """Eliminate affine rows repeatedly, including newly exposed affine rows."""
    remaining = tuple(variables)
    rows = tuple(equations)
    substitutions = []
    while True:
        rows = tuple(sp.expand(row) for row in rows if row != 0)
        if any(not row.free_symbols and row != 0 for row in rows):
            return AffinePresolve(rows, remaining, tuple(substitutions), True)
        affine = next(
            (row for row in rows if remaining and sp.Poly(row, *remaining).total_degree() == 1),
            None,
        )
        if affine is None:
            return AffinePresolve(rows, remaining, tuple(substitutions))
        # Eliminate the earliest supplied variable, leaving later parameters.
        variable = next(v for v in remaining if affine.coeff(v) != 0)
        coefficient = affine.coeff(variable)
        expression = sp.cancel(-(affine - coefficient * variable) / coefficient)
        substitutions.append((variable, expression))
        rows = tuple(sp.expand(row.subs(variable, expression)) for row in rows)
        remaining = tuple(v for v in remaining if v != variable)


def _bounded_substitution(
    poly, position, expression, others, *, term_limit, degree_limit, work_limit=32768
):
    """Probe cancellation with exact sparse arithmetic and hard work/support caps.

    None means the probe declined; it never relaxes the final row limits.
    A generous but bounded intermediate support permits cancellation between
    original terms. It does not materialize unbounded symbolic expansions.
    """
    support_limit = min(4096, max(64, term_limit * 8))
    zero = (0,) * len(others)
    substitution = dict(sp.Poly(expression, *others).terms()) if others else {zero: expression}
    if len(substitution) > support_limit:
        return None
    spent = 0

    def multiply(left, right):
        nonlocal spent
        spent += len(left) * len(right)
        if spent > work_limit:
            return None
        result = {}
        for a, ca in left.items():
            for b, cb in right.items():
                # Avoid enormous new rational coefficients even for tiny support.
                if (
                    isinstance(ca, sp.Rational)
                    and isinstance(cb, sp.Rational)
                    and max(
                        int(ca.p).bit_length() + int(cb.p).bit_length(),
                        int(ca.q).bit_length() + int(cb.q).bit_length(),
                    )
                    > 8192
                ):
                    return None
                exponent = tuple(x + y for x, y in zip(a, b, strict=True))
                value = sp.expand(result.get(exponent, sp.S.Zero) + ca * cb)
                if value == 0:
                    result.pop(exponent, None)
                else:
                    result[exponent] = value
                if len(result) > support_limit:
                    return None
        return result

    powers = {0: {zero: sp.S.One}, 1: substitution}

    def power(n):
        if n.bit_length() > 64:
            return None
        if n in powers:
            return powers[n]
        half = power(n // 2)
        if half is None:
            return None
        value = multiply(half, half)
        if value is not None and n % 2:
            value = multiply(value, substitution)
        powers[n] = value
        return value

    result = {}
    # Lower powers first: existing explicit counterterms are accumulated before
    # substituted high powers. This often keeps cancellation intermediates small.
    for exponent, coefficient in sorted(poly.terms(), key=lambda item: item[0][position]):
        value = power(int(exponent[position]))
        if value is None:
            return None
        spent += len(value)
        if spent > work_limit:
            return None
        shift = exponent[:position] + exponent[position + 1 :]
        for monomial, c in value.items():
            if (
                isinstance(coefficient, sp.Rational)
                and isinstance(c, sp.Rational)
                and max(
                    int(coefficient.p).bit_length() + int(c.p).bit_length(),
                    int(coefficient.q).bit_length() + int(c.q).bit_length(),
                )
                > 8192
            ):
                return None
            key = tuple(a + b for a, b in zip(shift, monomial, strict=True))
            updated = sp.expand(result.get(key, sp.S.Zero) + coefficient * c)
            if updated == 0:
                result.pop(key, None)
            else:
                result[key] = updated
            if len(result) > support_limit:
                return None
    if len(result) > term_limit or any(sum(p) > degree_limit for p in result):
        return None
    return sp.Poly.from_dict(result, others).as_expr() if others else result.get(zero, sp.S.Zero)


def polynomial_presolve(equations, variables, *, max_terms=1000, max_degree=16, growth_factor=4):
    """Eliminate constant-unit pivots, with bounds checked before expansion.

    A variable-dependent pivot is never divided out. Rejected substitutions
    remain in the system. Bounds apply to each resulting equation; existing
    higher degrees/term counts are permitted but cannot grow past their input size.
    """
    for value, name in (
        (max_terms, "max_terms"),
        (max_degree, "max_degree"),
        (growth_factor, "growth_factor"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    rows, remaining, substitutions = tuple(equations), tuple(variables), []
    probes = acceptances = 0
    while remaining:
        rows = tuple(sp.expand(row) for row in rows if row != 0)
        if any(not row.free_symbols and row != 0 for row in rows):
            return AffinePresolve(rows, remaining, tuple(substitutions), True, probes, acceptances)
        polynomials = [sp.Poly(row, *remaining) for row in rows]
        accepted = None
        # Affine rows first, then the least expensive nonlinear pivot.
        candidates = []
        for index, row in enumerate(rows):
            for variable in remaining:
                p = sp.Poly(row, variable)
                if p.degree() != 1:
                    continue
                coefficient = p.nth(1)
                if coefficient.free_symbols or coefficient == 0:
                    continue
                expression = sp.cancel(-p.nth(0) / coefficient)
                others = tuple(v for v in remaining if v != variable)
                ep = sp.Poly(expression, *others) if others else None
                terms = len(ep.terms()) if ep is not None else 1
                degree = ep.total_degree() if ep is not None else 0
                candidates.append(
                    (degree, terms, index, remaining.index(variable), variable, expression)
                )
        for _, _, index, _, variable, expression in sorted(candidates):
            others = tuple(v for v in remaining if v != variable)
            ep = sp.Poly(expression, *others) if others else None
            term_count = len(ep.terms()) if ep is not None else 1
            degree = ep.total_degree() if ep is not None else 0
            position = remaining.index(variable)
            safe = True
            transformed = []
            used_probe = False
            for row_index, poly in enumerate(polynomials):
                if row_index == index:
                    transformed.append(sp.S.Zero)
                    continue
                limit = max(max_terms, len(poly.terms()))
                limit = min(limit, max(1, len(poly.terms())) * growth_factor)
                degree_limit = max(max_degree, poly.total_degree())
                estimate = 0
                row_safe = True
                for powers, _ in poly.terms():
                    if sum(powers) - powers[position] + powers[position] * degree > degree_limit:
                        row_safe = False
                        break
                    # Saturating exponentiation avoids constructing enormous integers.
                    contribution = 1
                    if term_count > 1:
                        for _ in range(powers[position]):
                            contribution *= term_count
                            if contribution > limit:
                                break
                    estimate += contribution
                    if estimate > limit:
                        row_safe = False
                        break
                if not row_safe:
                    probes += 1
                    candidate_row = _bounded_substitution(
                        poly,
                        position,
                        expression,
                        others,
                        term_limit=limit,
                        degree_limit=degree_limit,
                    )
                    if candidate_row is None:
                        safe = False
                        break
                    transformed.append(candidate_row)
                    used_probe = True
                else:
                    transformed.append(None)
            if safe:
                accepted = variable, expression, transformed
                acceptances += int(used_probe)
                break
        if accepted is None:
            return AffinePresolve(rows, remaining, tuple(substitutions), False, probes, acceptances)
        variable, expression, transformed = accepted
        substitutions.append((variable, expression))
        rows = tuple(
            value if value is not None else sp.expand(row.subs(variable, expression))
            for row, value in zip(rows, transformed, strict=True)
        )
        remaining = tuple(v for v in remaining if v != variable)
    inconsistent = any(sp.expand(row) != 0 for row in rows)
    return AffinePresolve(rows, remaining, tuple(substitutions), inconsistent, probes, acceptances)
