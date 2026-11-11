"""Portable exact domain and polynomial state without mutable SymPy ring caches."""

import sympy as sp


def domain_state(domain):
    """Encode exact coefficient-field data without carrying mutable ring caches."""
    if domain == sp.QQ:
        return ("QQ",)
    if domain.is_FiniteField:
        return ("prime", int(domain.characteristic()))
    if domain.is_AlgebraicField:
        return ("number", tuple(domain.orig_ext))
    if domain.is_FractionField:
        return ("fraction", domain_state(domain.domain), domain.symbols)
    if getattr(domain, "is_AlgebraicFunctionField", False):
        return (
            "function",
            polynomial_state(domain.modulus),
            domain.tower_symbols,
            domain.images,
            tuple(polynomial_state(p) for p in domain.relations),
            tuple(
                (
                    polynomial_state(p.old_modulus),
                    polynomial_state(p.new_modulus),
                    polynomial_state(p.primitive_modulus),
                    p.old_image,
                    p.new_image,
                    p.primitive_expression,
                )
                for p in domain.compressions
            ),
        )
    if getattr(domain, "is_FiniteExtension", False):
        return ("finite_extension", polynomial_state(domain.modulus))
    raise TypeError(f"unsupported certificate coefficient domain {domain}")


def restore_domain(state):
    """Reconstruct an exact coefficient field from its portable mathematical data."""
    kind, *data = state
    if kind == "QQ":
        return sp.QQ
    if kind == "prime":
        return sp.GF(data[0])
    if kind == "number":
        return sp.QQ.algebraic_field(*data[0])
    if kind == "fraction":
        return restore_domain(data[0]).frac_field(*data[1])
    if kind == "finite_extension":
        from .finite_fields import FiniteFieldExtension

        return FiniteFieldExtension(restore_polynomial(data[0]))
    if kind == "function":
        from .algebraic_function_fields import AlgebraicFunctionField, TowerCompression

        field = AlgebraicFunctionField(restore_polynomial(data[0]))
        field.tower_symbols, field.images = data[1:3]
        field.relations = tuple(restore_polynomial(p) for p in data[3])
        field.compressions = tuple(
            TowerCompression(*(restore_polynomial(p) for p in proof[:3]), *proof[3:])
            for proof in data[4]
        )
        return field
    raise ValueError("unknown coefficient-state kind")


def polynomial_state(poly):
    """Encode a polynomial together with its coordinate order and exact domain."""
    return poly.as_expr(), poly.gens, domain_state(poly.domain)


def restore_polynomial(state):
    """Restore a polynomial while preserving bound algebraic-root symbols."""
    from .domains import polynomial

    expression, variables, domain = state
    return polynomial(expression, variables, restore_domain(domain))
