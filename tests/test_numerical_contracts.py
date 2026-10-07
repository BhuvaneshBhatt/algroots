"""Numerical topology, precision budgets, and public option contracts."""

import pytest
import sympy as sp
from flint import acb, acb_poly, ctx

from algroots import PolynomialSystemInputError, polysolve
from algroots.errors import NumericalRootError
from algroots.numerical import acb_mid_to_sympy, flint_dps, mpc_to_sympy
from algroots.quotient import QuotientAlgebra
from algroots.univariate import numerical_univariate_roots

x = sp.Symbol("x")


@pytest.mark.parametrize("method", ["auto", "shape", "rur", "action", "triangular"])
@pytest.mark.parametrize("sign", [-1, 1])
def test_small_real_and_complex_roots_preserve_topology(method, sign):
    result = polysolve(
        (x * x + sign * sp.Rational(1, 10**26),),
        (x,),
        digits=15,
        method=method,
        recognize=False,
    )
    assert len(result.roots) == 2
    components = [sp.im(root[0]) if sign == 1 else sp.re(root[0]) for root in result.roots]
    assert min(components) < 0 < max(components)
    assert all(abs(abs(c) - sp.Rational(1, 10**13)) < sp.Rational(1, 10**27) for c in components)


def test_acb_conversion_preserves_small_imaginary_component():
    value = acb_mid_to_sympy(acb("0", "1e-100"), 15)
    assert sp.im(value) > 0
    assert sp.re(value) == 0


@pytest.mark.parametrize("ceiling", [35, 50, 75])
def test_arb_isolation_never_receives_an_internal_ceiling_above_current_budget(
    monkeypatch, ceiling
):
    import flint

    calls = []

    class RecordingPolynomial:
        def __init__(self, coefficients):
            self.polynomial = acb_poly(coefficients)

        def roots(self, **kwargs):
            calls.append((ctx.dps, ctx.prec, kwargs["maxprec"]))
            return self.polynomial.roots(**kwargs)

    monkeypatch.setattr(flint, "acb_poly", RecordingPolynomial)
    before = ctx.prec
    if ceiling == 35:
        with pytest.raises(NumericalRootError, match="35 decimal digits"):
            numerical_univariate_roots(x * x - 2, x, 30, 100, ceiling)
    else:
        roots = numerical_univariate_roots(x * x - 2, x, 30, 100, ceiling)
        assert len(roots) == 2
    assert calls and all(dps <= ceiling and bits == limit for dps, bits, limit in calls)
    assert ctx.prec == before


@pytest.mark.parametrize("budget", [True, False, 1.0, float("nan"), float("inf"), 0, -1])
@pytest.mark.parametrize("factory", ["polynomials", "groebner"])
def test_quotient_dimension_budget_is_strict(budget, factory):
    with pytest.raises(ValueError, match="max_dimension"):
        if factory == "polynomials":
            QuotientAlgebra.from_polynomials((x * x - 1,), (x,), max_dimension=budget)
        else:
            QuotientAlgebra.from_groebner_basis(sp.groebner((x * x - 1,), x), max_dimension=budget)


@pytest.mark.parametrize(
    "option",
    [
        "digits",
        "verification_digits",
        "guard_digits",
        "maxsteps",
        "max_solutions",
        "max_action_dimension",
        "max_precision_digits",
        "recognition_max_degree",
    ],
)
@pytest.mark.parametrize("value", [True, 1.0, float("nan"), float("inf")])
def test_solver_integer_options_reject_flags_and_nonintegers(option, value):
    with pytest.raises(ValueError, match=option):
        polysolve((x * x - 1,), (x,), **{option: value})


@pytest.mark.parametrize("value", [0, 1, 0.0, 1.0, None, [], "yes"])
def test_certification_mode_has_explicit_types(value):
    with pytest.raises(PolynomialSystemInputError, match="certify"):
        polysolve((x * x - 1,), (x,), certify=value)


def test_unhashable_variable_has_public_input_error():
    with pytest.raises(PolynomialSystemInputError, match="Symbol"):
        polysolve((x * x - 1,), ([x],))


def test_flint_precision_restored_exactly_on_nested_exception():
    before = ctx.prec
    try:
        ctx.prec = 117
        with pytest.raises(RuntimeError), flint_dps(50):
            outer = ctx.prec
            with flint_dps(20):
                assert ctx.dps <= 20
            assert ctx.prec == outer
            raise RuntimeError("controlled failure")
        assert ctx.prec == 117
    finally:
        ctx.prec = before


def test_mpmath_conversion_preserves_small_components():
    import mpmath as mp

    with mp.workdps(50):
        value = mpc_to_sympy(mp.mpc("1e-100", "-1e-200"), 30)
    assert sp.re(value) > 0 and sp.im(value) < 0


@pytest.mark.parametrize("scale", [sp.Rational(1, 10**50), sp.Integer(10) ** 50])
def test_root_comparison_is_invariant_under_coordinate_rescaling(scale):
    y = sp.Symbol("y")
    result = polysolve((x * x - scale**2, y - x / scale), (x, y), digits=20, recognize=False)
    assert len(result.roots) == 2
    values = sorted(sp.re(root[1]) for root in result.roots)
    assert abs(values[0] + 1) < sp.Rational(1, 10**18)
    assert abs(values[1] - 1) < sp.Rational(1, 10**18)


def test_arb_exhaustion_stops_at_cap_and_restores_context(monkeypatch):
    import flint

    calls = []

    class UnresolvedPolynomial:
        def __init__(self, coefficients):
            pass

        def roots(self, **kwargs):
            calls.append((ctx.dps, ctx.prec, kwargs["maxprec"]))
            raise ValueError("controlled isolation failure")

    monkeypatch.setattr(flint, "acb_poly", UnresolvedPolynomial)
    before = ctx.prec
    with pytest.raises(NumericalRootError, match="90 decimal digits"):
        numerical_univariate_roots(x * x - 2, x, 30, 100, 90)
    assert [dps for dps, _, _ in calls] == [38, 76, 90]
    assert all(bits == limit for _, bits, limit in calls)
    assert ctx.prec == before


@pytest.mark.parametrize("method", ["auto", "shape", "rur", "action", "triangular"])
def test_close_roots_at_nonzero_center_remain_distinct(method):
    delta = sp.Rational(1, 10**13)
    result = polysolve(((x - 1) ** 2 - delta**2,), (x,), digits=15, method=method, recognize=False)
    assert len(result.roots) == 2
    values = sorted(sp.re(root[0]) for root in result.roots)
    assert values[0] < 1 < values[1]


@pytest.mark.parametrize("sign", [-1, 1])
def test_small_root_topology_remains_exactly_certifiable(sign):
    result = polysolve(
        (x * x + sign * sp.Rational(1, 10**26),),
        (x,),
        digits=15,
        recognize=False,
        certify="required",
    )
    assert len(result.roots) == 2
    assert all(attempt.status == "certified" for attempt in result.root_certifications)
    assert all(attempt.certificate.verify() for attempt in result.root_certifications)


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize(
    "option,value", [("digits", True), ("maxsteps", 1.0), ("max_precision_digits", float("nan"))]
)
def test_direct_rur_extraction_validates_before_empty_return(empty, option, value):
    from algroots.rational_univariate import compute_rational_univariate_representation
    from algroots.rational_univariate.solve import numerical_rur_roots

    representation = compute_rational_univariate_representation(
        (sp.S.One if empty else x * x - 1,), (x,)
    )
    options = dict(digits=30)
    options[option] = value
    with pytest.raises(ValueError, match=option):
        numerical_rur_roots(representation, **options)
