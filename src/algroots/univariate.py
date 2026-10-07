"""Shared distinct univariate root extraction with bounded Arb retries."""

from typing import Any

import sympy as sp

from ._validation import validate_integer
from .errors import NumericalRootError
from .numerical import acb_mid_to_sympy, flint_available, flint_dps, sympy_to_acb


def numerical_univariate_roots(
    expression: Any,
    variable: Any,
    digits: int,
    maxsteps: int,
    max_precision_digits: int | None = None,
) -> list[Any]:
    """Find all distinct roots of a univariate polynomial.

    python-flint/Arb is the normal backend. SymPy is retained only as a
    development fallback for environments where python-flint is unavailable.
    """
    validate_integer(digits, "digits")
    validate_integer(maxsteps, "maxsteps")
    if max_precision_digits is not None:
        validate_integer(max_precision_digits, "max_precision_digits", minimum=digits)
    try:
        polynomial = sp.Poly(expression, variable, extension=True).sqf_part()
    except (sp.PolynomialError, TypeError, ValueError) as exc:
        raise NumericalRootError(
            f"could not construct a univariate polynomial in {variable}"
        ) from exc
    if polynomial.degree() <= 0:
        raise NumericalRootError(f"polynomial in {variable} has no roots to compute")

    # Center exactly before converting coefficients to balls. This avoids
    # cancellation in clusters near a large nonzero coordinate without changing
    # root separation, multiplicity, or the precision budget.
    center = -polynomial.nth(polynomial.degree() - 1) / (polynomial.degree() * polynomial.LC())
    polynomial = polynomial.to_field().shift(center)

    if not flint_available():  # pragma: no cover - development fallback
        try:
            roots = sp.nroots(polynomial, n=digits, maxsteps=maxsteps)
        except (ValueError, ArithmeticError, TypeError) as exc:
            raise NumericalRootError(
                f"numerical root computation failed for variable {variable}"
            ) from exc
        return [sp.N(root + center, digits) for root in roots]

    from flint import acb_poly, ctx

    max_digits = max_precision_digits or max(digits * 4, digits + 100)
    current = min(max(30, digits + 8), max_digits)
    coeffs = list(reversed(polynomial.all_coeffs()))
    while current <= max_digits:
        with flint_dps(current):
            arb_poly = acb_poly([sympy_to_acb(coeff, current) for coeff in coeffs])
            try:
                roots = arb_poly.roots(tol=f"1e-{digits + 2}", maxprec=ctx.prec)
            except ValueError:
                roots = None
            if roots is not None and len(roots) == polynomial.degree():
                return [sp.N(acb_mid_to_sympy(root, current) + center, digits) for root in roots]
        if current == max_digits:
            break
        current = min(current * 2, max_digits)
    raise NumericalRootError(
        f"Arb could not isolate all roots for variable {variable} up to {max_digits} decimal digits"
    )
