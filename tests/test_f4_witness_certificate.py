"""Independent, proof-carrying GF(p) certificates (no SymPy membership oracle)."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from gb_case_catalog import build_case
from gb_exact_certificate import normalize, structural_checks
from gb_witness_certificate import (
    add_shifted,
    divide_with_quotients,
    efficient_spair_checks,
    membership_witnesses,
    verify_membership_witnesses,
)
from groebner_benchmark import load_native


@pytest.mark.parametrize(
    "name,prime,order",
    [
        ("nilpotent_primary", 101, "grevlex"),
        ("triangular_finite", 101, "lex"),
        ("coordinate_axes", 7, "grlex"),
        ("minors2x3_r2", 101, "grevlex"),
        ("m2_lorentz", 101, "grevlex"),
    ],
)
def test_native_output_has_explicit_input_membership_witness(name, prime, order):
    pytest.importorskip("algroots._native_f4")
    case = build_case(name)
    native = load_native()
    f = [normalize(g, prime) for g in case.polynomials]
    basis, _ = native.compute(
        f,
        prime,
        case.nvars,
        order,
        max_pairs=30000,
        max_basis=600,
        max_terms=200000,
        max_reduction_steps=3000000,
    )
    cert = membership_witnesses(f, basis, prime, order, max_pairs=600, max_steps=100000)
    assert verify_membership_witnesses(f, basis, cert, prime, order)
    checks = structural_checks(case, prime, order, basis)
    assert checks["monic_reduced"] and checks["s_pairs_reduce_to_zero"]
    assert checks["generators_reduce_to_zero"]
    assert (
        checks["checked_spairs"] + checks["product_skipped_spairs"] + checks["chain_skipped_spairs"]
        == checks["total_spairs"]
    )
    import json

    assert verify_membership_witnesses(f, basis, json.loads(json.dumps(cert)), prime, order)
    if any(poly for vec in cert["witnesses"] for poly in vec):
        corrupt = copy.deepcopy(cert)
        for vec in corrupt["witnesses"]:
            for poly in vec:
                if poly:
                    poly[0][1] = (poly[0][1] + 1) % prime
                    assert not verify_membership_witnesses(f, basis, corrupt, prime, order)
                    return


def test_reject_strictly_larger_ideal_and_check_quotient_identity():
    case = build_case("nilpotent_primary")
    fake = [{(1, 0): 1}, {(0, 3): 1}]
    with pytest.raises(ValueError, match="NOT in"):
        membership_witnesses(case.polynomials, fake, 101, "grevlex")
    f = {(2, 0): 3, (1, 1): 5, (0, 0): 1}
    b = [{(1, 0): 1, (0, 0): 1}, {(0, 2): 1, (0, 0): 1}]
    r, q = divide_with_quotients(f, b, 7, "grevlex")
    recombined = dict(r)
    for multiplier, g in zip(q, b, strict=False):
        for mon, coeff in multiplier.items():
            add_shifted(recombined, g, mon, coeff, 7)
    assert recombined == normalize(f, 7)


def test_product_and_chain_criterion_proven_dependencies():
    # Reduced monomial ideal with LM(x^2*y), LM(x*z^2), LM(x*y*z).
    # The (0,1) pair is justified by the two previously proven degree-4 pairs.
    b = [{(2, 1, 0): 1}, {(1, 0, 2): 1}, {(1, 1, 1): 1}]
    res = efficient_spair_checks(b, 101, "grevlex")
    assert res["ok"] and res["chain_pairs"] >= 1
    assert res["pairs_checked"] + res["product_pairs"] + res["chain_pairs"] == res["total_pairs"]


def test_witness_budget_exhaustion_is_not_success():
    generators = [{(2, 0): 1, (0, 0): 1}, {(0, 2): 1, (0, 0): 1}]
    with pytest.raises(RuntimeError):
        membership_witnesses(generators, generators, 101, "grevlex", max_pairs=0)
