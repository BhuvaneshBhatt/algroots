"""Domain-aware tiled F4 computation and a portable SymPy fallback."""

import sympy as sp
from sympy.polys.groebnertools import groebner as sympy_groebner
from sympy.polys.rings import PolyElement

from .native_f4 import NativeF4Limits, native_modular_basis, native_prime_basis


class UnsupportedGroebnerDomain(ValueError):
    """The requested engine cannot compute over this coefficient domain."""


def _coerce(polynomials, ring):
    converted = []
    for expression in polynomials:
        if isinstance(expression, PolyElement) and expression.ring is ring:
            # The exact ring identity fixes coefficient representation and order.
            # Copy the sparse container so kernels cannot modify caller inputs.
            poly = expression.copy()
        elif ring.domain.is_AlgebraicField or ring.domain.is_FractionField:
            from .domains import polynomial

            poly = ring.from_dict(polynomial(expression, ring.symbols, ring.domain).rep.to_dict())
        else:
            poly = ring.from_expr(
                expression.as_expr() if hasattr(expression, "as_expr") else expression
            )
        if poly:
            converted.append(poly)
    return tuple(converted)


def _native_field_basis(polynomials, ring, limits, certify):
    """Use the same materialized inputs for computation and verification."""

    inputs = tuple(polynomials)
    result = native_prime_basis(inputs, ring, int(ring.domain.characteristic()), limits=limits)[0]
    if certify:
        reference = tuple(sympy_groebner(_coerce(inputs, ring), ring))
        # A reduced monic basis is unique, independently of row ordering.
        if {frozenset(p.items()) for p in result} != {frozenset(p.items()) for p in reference}:
            raise ArithmeticError("independent finite-field reference disagrees")
    return result


def compute_groebner_basis(polynomials, ring, *, backend="native", limits=None, certify=True):
    """Return ring polynomials using tiled native F4 or portable SymPy.

    Native QQ reconstruction produces a candidate when ``certify=False``;
    ``certify=True`` requests an independent exact proof. The public
    ``groebner_basis`` interface always verifies its returned mathematical result.
    """
    if backend not in ("sympy", "native"):
        raise ValueError("backend must be sympy or native")
    if backend == "sympy":
        if limits is not None:
            raise ValueError("native limits do not apply to SymPy")
        return tuple(sympy_groebner(_coerce(polynomials, ring), ring))
    if limits is not None and not isinstance(limits, NativeF4Limits):
        raise TypeError("native limits must be NativeF4Limits")
    from .finite_fields import FiniteFieldExtension

    if isinstance(ring.domain, FiniteFieldExtension):
        from .native_fields import native_finite_basis

        return native_finite_basis(_coerce(polynomials, ring), ring, limits=limits, certify=certify)
    if getattr(ring.domain, "is_FiniteField", False):
        return _native_field_basis(polynomials, ring, limits, certify)
    if (
        ring.domain.is_FractionField
        or ring.domain.is_AlgebraicField
        or getattr(ring.domain, "is_AlgebraicFunctionField", False)
    ):
        from .native_fields import native_field_basis

        return native_field_basis(
            _coerce(polynomials, ring), ring, limits=limits, certify=certify
        ).basis
    if ring.domain != sp.QQ:
        raise UnsupportedGroebnerDomain("native F4 requires an exact supported field")
    return native_modular_basis(polynomials, ring, limits=limits, certify=certify).basis
