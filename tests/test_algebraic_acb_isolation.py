"""Exact-bound tests independent of python-flint's optional importability."""

from dataclasses import dataclass

import pytest
import sympy as sp

from algroots import numerical


@dataclass(frozen=True)
class RationalBall:
    midpoint: sp.Rational
    radius: sp.Rational

    @property
    def bounds(self):
        return self.midpoint - self.radius, self.midpoint + self.radius


def test_exact_isolation_bounds(monkeypatch):
    # Capture the *rational* endpoints passed to Arb before it widens for
    # binary rounding, so the proof premise can be inspected independently.
    monkeypatch.setattr(
        numerical, "fmpq", lambda numerator, denominator: sp.Rational(numerator, denominator)
    )
    monkeypatch.setattr(numerical, "arb", lambda midpoint, radius=0: RationalBall(midpoint, radius))
    monkeypatch.setattr(numerical, "acb", lambda real, imaginary=0: (real, imaginary))

    cases = (
        sp.sqrt(2),
        sp.sqrt(2) + sp.I * sp.sqrt(3),
        (1 + sp.sqrt(5)) / 2,
        (sp.sqrt(2) - 1) ** 7,
        sp.I / 3,
    )
    for number in cases:
        real, imaginary = numerical.sympy_to_acb(number, 24)
        for component, ball in ((sp.re(number), real), (sp.im(number), imaginary)):
            lower, upper = ball.bounds if isinstance(ball, RationalBall) else (ball, ball)
            assert sp.simplify(component - lower).is_nonnegative is True
            assert sp.simplify(upper - component).is_nonnegative is True
            assert upper - lower <= sp.Rational(1, 10**24)


def test_algebraic_acb_rejects_inexact_values(monkeypatch):
    monkeypatch.setattr(numerical, "acb", lambda *args: args)
    for value in (sp.Float("1.25"), sp.pi, sp.Symbol("a"), sp.oo):
        with pytest.raises(ValueError, match="exact algebraic"):
            numerical.sympy_to_acb(value, 30)
    with pytest.raises(ValueError, match="positive integer"):
        numerical.sympy_to_acb(sp.sqrt(2), 0)


@pytest.mark.flint
@pytest.mark.skipif(not numerical.flint_available(), reason="python-flint is unavailable")
def test_native_acb_contains_exact_algebraic_values():
    from flint import acb

    cases = (
        (sp.sqrt(2), acb(2).sqrt()),
        (sp.I * sp.sqrt(3), acb(0, 1) * acb(3).sqrt()),
        (
            sp.sqrt(2) + sp.I * sp.sqrt(3),
            acb(2).sqrt() + acb(0, 1) * acb(3).sqrt(),
        ),
    )
    with numerical.flint_dps(50):
        for value, reference in cases:
            enclosure = numerical.sympy_to_acb(value, 40)
            assert enclosure.overlaps(reference)
            assert (enclosure - reference).contains(0)


def test_exact_isolation_failure_has_no_heuristic_fallback(monkeypatch):
    monkeypatch.setattr(numerical, "acb", lambda *args: args)

    def unsupported_isolation(*_args, **_kwargs):
        raise NotImplementedError("isolation unavailable")

    monkeypatch.setattr(numerical, "isolate", unsupported_isolation)
    with pytest.raises(numerical.AlgebraicIsolationError, match="could not isolate"):
        numerical.sympy_to_acb(sp.sqrt(2), 30)
