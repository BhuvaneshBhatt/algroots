"""Compiled arbitrary-precision numerical helpers."""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import mpmath as mp
import sympy as sp
from sympy.polys.numberfields import isolate

from ._validation import validate_integer

try:  # pragma: no cover - exercised when python-flint is installed
    from flint import acb, acb_mat, acb_poly, arb, ctx, fmpq
except ImportError:  # pragma: no cover - local development fallback
    acb = acb_mat = acb_poly = arb = ctx = fmpq = None


def flint_available() -> bool:
    """Return whether python-flint's Arb/ACB types are available."""
    return acb is not None


@contextmanager
def flint_dps(digits: int):
    """Temporarily set FLINT decimal precision and restore it afterward."""
    if ctx is None:
        yield
        return
    validate_integer(digits, "digits")
    old_precision = ctx.prec
    ctx.dps = digits
    try:
        yield
    finally:
        ctx.prec = old_precision


def _decimal_parts(value: Any, digits: int) -> tuple[str, str]:
    numeric = sp.N(value, digits)
    real = sp.N(sp.re(numeric), digits)
    imag = sp.N(sp.im(numeric), digits)
    return str(real), str(imag)


class AlgebraicIsolationError(ValueError):
    """An exact algebraic input could not be enclosed rigorously."""


def _algebraic_part_ball(part: sp.Expr, digits: int):
    """Enclose a real algebraic component using exact rational isolation."""
    part = sp.sympify(part)
    if part.is_Rational:
        rational = fmpq(int(part.p), int(part.q))
        return arb(rational)
    if part.is_real is not True or part.is_algebraic is not True:
        raise AlgebraicIsolationError("ACB conversion requires an exact real algebraic component")
    # isolate() uses exact real-root isolation; the requested width is a
    # rational number, not a tolerance inferred from numerical agreement.
    try:
        lower, upper = isolate(part, eps=sp.Rational(1, 10 ** max(digits, 1)))
    except (ValueError, NotImplementedError, TypeError, NameError) as exc:
        raise AlgebraicIsolationError("could not isolate the exact algebraic component") from exc
    if not (lower.is_Rational and upper.is_Rational and lower <= upper):
        raise AlgebraicIsolationError("algebraic isolation did not return rational endpoints")
    midpoint = (lower + upper) / 2
    radius = (upper - lower) / 2
    return arb(fmpq(int(midpoint.p), int(midpoint.q)), fmpq(int(radius.p), int(radius.q)))


def sympy_to_acb(value: Any, digits: int):
    """Return a rigorous Arb/ACB enclosure of an exact algebraic number.

    Real and imaginary parts are isolated using exact rational bounds. The Arb
    midpoint/radius constructor includes its own rounding errors, so the final
    balls enclose the exact components. No numerical evalf comparison is used.
    """
    if acb is None:
        raise RuntimeError("python-flint is required for the ACB backend")
    validate_integer(digits, "digits")
    value = sp.sympify(value)
    if value.has(sp.Float) or value.is_number is not True or value.is_algebraic is not True:
        raise AlgebraicIsolationError("ACB conversion requires an exact algebraic number")
    if value.is_Rational:
        return acb(fmpq(int(value.p), int(value.q)))
    return acb(
        _algebraic_part_ball(sp.re(value), digits),
        _algebraic_part_ball(sp.im(value), digits),
    )


def acb_mid_to_sympy(value: Any, digits: int):
    """Convert the midpoint of an ACB ball to a SymPy numeric value."""
    midpoint = value.mid()
    real_text = midpoint.real.str(digits + 5, radius=False)
    imag_text = midpoint.imag.str(digits + 5, radius=False)
    if any(token in text.lower() for text in (real_text, imag_text) for token in ("nan", "inf")):
        raise ValueError("ACB midpoint is nonfinite")
    real = sp.Float(real_text, digits)
    imag = sp.Float(imag_text, digits)
    if real.is_finite is not True or imag.is_finite is not True:
        raise ValueError("ACB midpoint is nonfinite")
    return real + sp.I * imag


def mpc_to_sympy(value: Any, digits: int):
    """Round a finite numerical scalar without discarding small components."""
    real = sp.Float(mp.nstr(mp.re(value), n=digits + 5), digits)
    imag = sp.Float(mp.nstr(mp.im(value), n=digits + 5), digits)
    if real.is_finite is not True or imag.is_finite is not True:
        raise ValueError("numerical scalar is nonfinite")
    return real + sp.I * imag


def acb_mid_to_mpc(value: Any, digits: int) -> mp.mpc:
    """Extract the midpoint of an Arb complex ball at the requested precision.

    The midpoint is numerical data; the original ball radius and any proof
    obligation must be retained by callers when certification is required.
    """
    midpoint = value.mid()
    real = mp.mpf(midpoint.real.str(digits + 5, radius=False))
    imag = mp.mpf(midpoint.imag.str(digits + 5, radius=False))
    return mp.mpc(real, imag)


@dataclass(frozen=True)
class CompiledPolynomial:
    """Sparse polynomial representation for repeated numerical evaluation."""

    terms: tuple[tuple[Any, tuple[int, ...]], ...]
    variable_count: int

    @classmethod
    def from_expr(cls, expression: Any, variables: tuple[Any, ...]) -> CompiledPolynomial:
        """Compile an exact polynomial into sparse coefficient and exponent data."""
        poly = sp.Poly(expression, *variables, extension=True)
        return cls(
            terms=tuple((coeff, tuple(int(p) for p in powers)) for powers, coeff in poly.terms()),
            variable_count=len(variables),
        )

    def evaluate_mp(self, values: tuple[Any, ...], digits: int) -> tuple[mp.mpc, mp.mpf]:
        """Evaluate the polynomial and a cancellation-aware magnitude bound.

        Powers are cached per coordinate so repeated monomials share work. The
        returned scale is used in relative residual checks, not exact proofs.
        """
        with mp.workdps(digits + 8):
            total = mp.mpc(0)
            scale = mp.mpf(0)
            powers_cache: list[dict[int, mp.mpc]] = [{0: mp.mpc(1)} for _ in values]
            for coefficient, powers in self.terms:
                term = sympy_to_mpc(coefficient, digits + 8)
                for index, power in enumerate(powers):
                    if not power:
                        continue
                    cached = powers_cache[index]
                    if power not in cached:
                        cached[power] = mp.mpc(values[index]) ** power
                    term *= cached[power]
                total += term
                scale += abs(term)
            return total, max(mp.mpf(1), scale)


def sympy_to_mpc(value: Any, digits: int) -> mp.mpc:
    """Convert a SymPy-compatible scalar to arbitrary-precision mpmath complex."""
    real, imag = _decimal_parts(value, digits)
    with mp.workdps(digits + 5):
        return mp.mpc(mp.mpf(real), mp.mpf(imag))


def number_to_mpc(value: Any, digits: int) -> mp.mpc:
    """Convert mpmath, Python, or SymPy numeric scalars without binary64 fallback."""
    if isinstance(value, mp.mpc):
        return mp.mpc(value)
    if isinstance(value, mp.mpf):
        return mp.mpc(value)
    if isinstance(value, complex):
        with mp.workdps(digits + 5):
            return mp.mpc(str(value.real), str(value.imag))
    return sympy_to_mpc(value, digits)


def compile_polynomials(
    expressions: Iterable[Any], variables: tuple[Any, ...]
) -> tuple[CompiledPolynomial, ...]:
    """Compile a sequence of polynomials for repeated high-precision evaluation."""
    return tuple(CompiledPolynomial.from_expr(expr, variables) for expr in expressions)


def compiled_residual(
    polynomials: tuple[CompiledPolynomial, ...],
    values: tuple[Any, ...],
    digits: int,
) -> mp.mpf:
    """Return the maximum coefficient-scaled residual across the system.

    Each residual is divided by a scale at least one computed from absolute
    term magnitudes; this is a numerical diagnostic, not a certificate.
    """
    residual = mp.mpf(0)
    for polynomial in polynomials:
        value, scale = polynomial.evaluate_mp(values, digits)
        residual = max(residual, abs(value) / scale)
    return residual


def compiled_jacobian(
    expressions: tuple[Any, ...], variables: tuple[Any, ...]
) -> tuple[tuple[CompiledPolynomial, ...], ...]:
    """Compile the symbolic Jacobian entries for numerical evaluation."""
    return tuple(
        tuple(CompiledPolynomial.from_expr(sp.diff(expr, var), variables) for var in variables)
        for expr in expressions
    )


def newton_refine(
    polynomials: tuple[CompiledPolynomial, ...],
    jacobian: tuple[tuple[CompiledPolynomial, ...], ...],
    start: tuple[Any, ...],
    digits: int,
    maxsteps: int,
) -> tuple[tuple[mp.mpc, ...], bool]:
    """Refine a square/overdetermined polynomial system numerically."""
    if len(polynomials) < len(start):
        return tuple(mp.mpc(v) for v in start), False
    with mp.workdps(digits + 10):
        x = mp.matrix([mp.mpc(v) for v in start])
        target = mp.power(10, -(digits - 6))
        for _ in range(maxsteps):
            vals = tuple(x[i] for i in range(len(start)))
            f = mp.matrix([poly.evaluate_mp(vals, digits + 5)[0] for poly in polynomials])
            if max((abs(v) for v in f), default=mp.mpf(0)) <= target:
                return tuple(x[i] for i in range(len(start))), True
            jmat = mp.matrix(
                [[entry.evaluate_mp(vals, digits + 5)[0] for entry in row] for row in jacobian]
            )
            try:
                if jmat.rows == jmat.cols:
                    delta = mp.lu_solve(jmat, -f)
                else:
                    delta = mp.qr_solve(jmat, -f)[0]
            except (ZeroDivisionError, ValueError):
                return tuple(x[i] for i in range(len(start))), False
            trial = x + delta
            trial_vals = tuple(trial[i] for i in range(len(start)))
            old_res = compiled_residual(polynomials, vals, digits + 5)
            new_res = compiled_residual(polynomials, trial_vals, digits + 5)
            if new_res > old_res:
                # Simple damping avoids accepting a Newton step that degrades the root.
                accepted = False
                factor = mp.mpf("0.5")
                for _ in range(8):
                    trial = x + factor * delta
                    trial_vals = tuple(trial[i] for i in range(len(start)))
                    new_res = compiled_residual(polynomials, trial_vals, digits + 5)
                    if new_res <= old_res:
                        accepted = True
                        break
                    factor /= 2
                if not accepted:
                    return tuple(x[i] for i in range(len(start))), False
            x = trial
            if max((abs(delta[i]) for i in range(len(start))), default=mp.mpf(0)) <= target:
                return tuple(x[i] for i in range(len(start))), True
        return tuple(x[i] for i in range(len(start))), False


class NumericExpression:
    """Compiled evaluator for the algebraic expression subset accepted by algroots."""

    def __init__(self, expression: Any, variables: tuple[Any, ...]):
        self.expression = sp.sympify(expression)
        self.variables = variables
        self.index = {var: i for i, var in enumerate(variables)}
        self._eval = self._compile(self.expression)

    def _compile(self, expression: Any):
        if expression.is_number:
            return lambda values, digits, value=expression: sympy_to_mpc(value, digits)
        if isinstance(expression, sp.Symbol):
            index = self.index[expression]
            return lambda values, digits, index=index: mp.mpc(values[index])
        if expression.is_Add:
            funcs = tuple(self._compile(arg) for arg in expression.args)
            return lambda values, digits: sum((f(values, digits) for f in funcs), mp.mpc(0))
        if expression.is_Mul:
            funcs = tuple(self._compile(arg) for arg in expression.args)

            def mul(values, digits):
                result = mp.mpc(1)
                for func in funcs:
                    result *= func(values, digits)
                return result

            return mul
        if expression.is_Pow:
            base, exponent = expression.as_base_exp()
            base_func = self._compile(base)
            if exponent.is_Integer:
                power = int(exponent)
                return lambda values, digits: base_func(values, digits) ** power
            if exponent.is_Rational:
                numerator = int(exponent.p)
                denominator = int(exponent.q)
                return lambda values, digits: mp.power(
                    base_func(values, digits),
                    mp.mpf(numerator) / denominator,
                )
        raise ValueError(
            f"unsupported algebraic expression for numerical compilation: {expression!r}"
        )

    def evaluate(self, values: tuple[Any, ...], digits: int) -> mp.mpc:
        with mp.workdps(digits + 8):
            return self._eval(values, digits + 5)
