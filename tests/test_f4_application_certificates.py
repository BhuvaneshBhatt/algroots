"""Application-derived cases and independent finite-field GB certificates."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from gb_case_catalog import build_case, names_for_suite
from gb_exact_certificate import (
    dimension_of_artinian_quotient,
    full_certificate,
    standard_monomial_hilbert_values,
    structural_checks,
)
from gb_published_additions import SYSTEMS, exact_lift_primes, exact_polynomial, published_case
from groebner_benchmark import _validate_exact_lift, load_native, verify_sympy


def native_case(name, prime=101, order="grevlex"):
    case = build_case(name)
    inp = [{m: c % prime for m, c in f.items() if c % prime} for f in case.polynomials]
    try:
        native = load_native()
    except RuntimeError:
        pytest.skip("optional native engine not built")
    basis, stats = native.compute(
        inp,
        prime,
        case.nvars,
        order,
        max_pairs=30000,
        max_basis=600,
        max_terms=200000,
        max_reduction_steps=3000000,
        batch_size=128,
    )
    return basis, stats


def test_catalog_has_59_distinct_cases_and_literal_provenance():
    names = names_for_suite("all")
    assert len(names) == 59 and len(set(names)) == 59
    for name in SYSTEMS:
        case = build_case(name)
        n, polys, source, scales = published_case(name)
        assert case.nvars == n and case.polynomials == polys
        assert "/ExampleSystems/" in source and source.endswith(".m2")
        assert len(scales) == len(case.polynomials)


def test_exact_decimal_translation_and_denominator_guards():
    puma = build_case("m2_puma_decimal")
    # Index 4: 0.004731*x1*x3 - 0.3578*x2*x3 + ...
    f = puma.polynomials[4]
    assert f[(1, 0, 1, 0, 0, 0, 0, 0)] == 4731
    assert f[(0, 1, 1, 0, 0, 0, 0, 0)] == -357800
    assert f[(0, 0, 0, 0, 0, 0, 1, 0)] == 1000000
    chemequ = build_case("m2_chemical_decimal")
    assert chemequ.polynomials[1][(0, 2, 0, 0, 0)] == 7692  # 1.9230e-6 * 4e9
    assert exact_lift_primes("m2_rose_rational") == (2, 3, 5, 7)
    assert exact_lift_primes("m2_lorentz") == ()
    for name in ("m2_boon_rational", "m2_puma_decimal", "m2_chemical_decimal"):
        for bad in (2, 5):
            with pytest.raises(ValueError, match="denominators"):
                _validate_exact_lift(name, bad)
    for bad in (2, 3, 5, 7):
        with pytest.raises(ValueError, match="denominators"):
            _validate_exact_lift("m2_rose_rational", bad)
    assert exact_polynomial("1.9230e-6*x+3/5", ("x",)) == {
        (1,): __import__("fractions").Fraction(1923, 10**9),
        (0,): __import__("fractions").Fraction(3, 5),
    }
    with pytest.raises(ValueError):
        exact_polynomial("x.__class__", ("x",))


@pytest.mark.parametrize(
    "case,prime,order",
    [
        ("m2_lorentz", 101, "grevlex"),
        ("m2_lorentz", 7, "lex"),
        ("m2_lorentz", 101, "grlex"),
        ("m2_boon_rational", 101, "grevlex"),
        ("nilpotent_primary", 7, "grevlex"),
        ("triangular_finite", 101, "lex"),
        ("minors2x3_r2", 101, "grevlex"),
        ("coordinate_axes", 101, "grlex"),
    ],
)
def test_full_two_sided_certificate(case, prime, order):
    definition = build_case(case)
    basis, _ = native_case(case, prime, order)
    result = full_certificate(definition, prime, order, basis)
    assert result["full_certificate_ok"], (case, result)
    assert verify_sympy(definition, prime, order, basis)


def test_structural_checks_alone_cannot_prove_two_sided_ideal_equality():
    # (x,y^3) is a *strictly larger* ideal than (x^2,xy,y^3).
    # Its reduced GB passes S-pairs and reduces all original generators.
    case = build_case("nilpotent_primary")
    fake = [{(1, 0): 1}, {(0, 3): 1}]
    structural = structural_checks(case, 101, "grevlex", fake)
    assert structural["monic_reduced"]
    assert structural["generators_reduce_to_zero"]
    assert structural["s_pairs_reduce_to_zero"]
    full = full_certificate(case, 101, "grevlex", fake)
    assert not full["output_in_original_ideal"]
    assert not full["full_certificate_ok"]


def test_detect_tampered_nonmonic_and_missing_basis_polynomials():
    case = build_case("nilpotent_primary")
    basis, _ = native_case("nilpotent_primary")
    bad = [g.copy() for g in basis]
    bad[0] = {m: (3 * c) % 101 for m, c in bad[0].items()}
    assert not structural_checks(case, 101, "grevlex", bad)["monic_reduced"]
    assert not full_certificate(case, 101, "grevlex", basis[1:])["full_certificate_ok"]


def test_known_quotient_dimensions_and_hilbert_values():
    for name, dimension in [("nilpotent_primary", 4), ("triangular_finite", 8)]:
        case = build_case(name)
        for order in ("lex", "grevlex"):
            basis, _ = native_case(name, 101, order)
            assert dimension_of_artinian_quotient(basis, case.nvars, order) == dimension
    for name, expected in [
        ("coordinate_axes", [1, 3, 3, 3, 3]),
        ("minors2x3_r2", [1, 6, 18, 40, 75]),
        ("minors3x3_r3", [1, 9, 45, 164, 486]),
    ]:
        case = build_case(name)
        for order in ("grlex", "grevlex"):
            basis, _ = native_case(name, 101, order)
            assert standard_monomial_hilbert_values(basis, case.nvars, order, 4) == expected


def test_application_certificate():
    for name in ("m2_lorentz", "m2_boon_rational", "minors2x3_r2"):
        basis, _ = native_case(name, 101, "grevlex")
        assert full_certificate(build_case(name), 101, "grevlex", basis)["full_certificate_ok"]
