"""Tests of the cross-CAS benchmarking harness (no external CAS required)."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from bench_cross_cas import (  # noqa: E402
    ORDER_TRANSLATIONS,
    basis_fingerprint,
    external_scripts,
    input_polynomials,
    verified_fingerprint,
)
from gb_case_catalog import build_case, names_for_suite  # noqa: E402


def test_all_59_cases_exportable():
    cases = names_for_suite("all")
    assert len(cases) == len(set(cases)) == 59
    for name in cases:
        case = build_case(name)
        assert case.polynomials
        assert all(all(len(m) == case.nvars for m in poly) for poly in case.polynomials)
        for prime in (0, 101, 1009):
            for order in ORDER_TRANSLATIONS:
                script = external_scripts(case, prime, order)
                assert "GB_BASIS_BEGIN" in script["singular"]
                assert "GB_BASIS_END" in script["macaulay2"]
                assert f"ring R = {prime}," in script["singular"]
                assert ("R = QQ[" if prime == 0 else f"R = (ZZ/{prime})[") in script["macaulay2"]
                assert f"MonomialOrder => {ORDER_TRANSLATIONS[order][1]}" in script["macaulay2"]
                assert f"),{ORDER_TRANSLATIONS[order][0]};" in script["singular"]


def test_source_case_characteristic_changes_are_explicit():
    # Characteristic-dependent inputs such as Frobenius systems are allowed;
    # source exact rational lifts with noninvertible denominators are not.
    assert input_polynomials(build_case("cyclic4"), 101)
    assert input_polynomials(build_case("cyclic4"), 0)
    with pytest.raises(ValueError):
        input_polynomials(build_case("m2_puma_decimal"), 2)
    with pytest.raises(ValueError):
        input_polynomials(build_case("cyclic4"), 4)


@pytest.mark.parametrize("field", [0, 101])
@pytest.mark.parametrize("order", ["lex", "grlex", "grevlex"])
def test_candidate_superideal_is_detected(field, order):
    case = build_case("cyclic4")
    sparse = input_polynomials(case, field)
    one = {(0, 0, 0, 0): ("1" if field == 0 else 1)}
    ok, _ = verified_fingerprint(case, sparse, field, order, [one])
    assert not ok


def test_qq_hash_distinguishes_denominators():
    assert basis_fingerprint([{(1,): "1/2"}], 0) != basis_fingerprint([{(1,): "1/3"}], 0)


def test_gf_hash_distinguishes_coefficients():
    assert basis_fingerprint([{(1,): 1}], 101) != basis_fingerprint([{(1,): 2}], 101)


from bench_cross_cas import parse_external_polynomial  # noqa: E402


@pytest.mark.parametrize("prime", [0, 7, 101, 1009])
@pytest.mark.parametrize("order", ["lex", "grlex", "grevlex"])
def test_exact_external_output_parser_matches_ordered_sympy(prime, order):
    import sympy as sp
    from sympy.polys.rings import PolyRing

    symbols = sp.symbols("x0:3")
    poly = "x0^2*x1 - 3*x0*x2^2 + 5*x1*x2^2 - 7*x2 + 11"
    actual = parse_external_polynomial(poly, 3, prime, order)
    ring = PolyRing(symbols, sp.QQ if not prime else sp.GF(prime), order)
    expected = ring.from_expr(
        sp.sympify(poly.replace("^", "**"), locals={str(s): s for s in symbols})
    ).monic()
    assert actual == {
        tuple(m): (str(sp.Rational(c)) if not prime else int(c) % prime)
        for m, c in expected.terms()
        if c
    }


def test_external_monic_depends_on_requested_order():
    lex = parse_external_polynomial("2*x0+x1^2", 2, 101, "lex")
    grevlex = parse_external_polynomial("2*x0+x1^2", 2, 101, "grevlex")
    assert lex == {(1, 0): 1, (0, 2): 51}
    assert grevlex == {(1, 0): 2, (0, 2): 1}


def test_external_rational_and_zero_polynomials():
    assert parse_external_polynomial("-2*x0 + 1/2*x1", 2, 0, "lex") == {(1, 0): "1", (0, 1): "-1/4"}
    assert parse_external_polynomial("0", 2, 7, "grevlex") == {}
    assert parse_external_polynomial("x0-x0", 2, 101, "grevlex") == {}


@pytest.mark.parametrize("text", ["x3", "x0**2", "x0 + strange", ""])
def test_external_output_parser_fails_closed(text):
    with pytest.raises((ValueError, ZeroDivisionError)):
        parse_external_polynomial(text, 2, 101, "grevlex")


def test_singular_scripts_use_millisecond_timing_resolution():
    script = external_scripts(build_case("cyclic4"), 101, "grevlex")["singular"]
    assert script.startswith('system("--ticks-per-sec",1000);\n')
