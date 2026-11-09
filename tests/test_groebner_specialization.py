"""Replay specialization identities and preserve all exceptional fibers."""

from dataclasses import replace

import pytest
import sympy as sp

from algroots import GroebnerVerificationError, groebner_basis, parametric_groebner_basis
from algroots.native_f4 import compiled_available

x, y, t, u = sp.symbols("x y t u")


@pytest.mark.parametrize("backend", ["sympy", "native"])
@pytest.mark.parametrize(
    "equations", [[t * x], [t * x - 1], [t * x, y - x], [t * x - 1, (t - 1) * y], [t, x]]
)
def test_exceptional_fibers(backend, equations):
    if backend == "native" and not compiled_available():
        pytest.skip("native F4 is not installed")
    proof = parametric_groebner_basis(equations, (x, y), (t,), backend=backend)
    assert proof.verify()
    for value in [0, 1, 2]:
        result = proof.specialize({t: value})
        expected = groebner_basis([p.subs(t, value) for p in equations], (x, y), backend="sympy")
        assert result == expected


def test_hidden_exception():
    proof = parametric_groebner_basis([t * x], (x,), (t,))
    assert list(proof.basis) == [x]
    assert proof.guard == t
    assert list(proof.specialize({t: 0})) == []
    assert list(proof.specialize({t: 2})) == [x]
    assert proof.as_formula().subs({t: 0, x: 7}) == sp.true


def test_input_poles():
    proof = parametric_groebner_basis([x / t], (x,), (t,))
    assert proof.input_guard == t
    with pytest.raises(ValueError, match="input coefficient domain"):
        proof.specialize({t: 0})
    assert proof.as_formula().subs({t: 0, x: 0}) == sp.false


def test_tampered_proofs():
    proof = parametric_groebner_basis([t * x + y, x * y - 1], (x, y), (t,))
    assert proof.verify()
    assert not replace(proof, guard=sp.S.One).verify()
    assert not replace(
        proof, memberships=tuple(tuple(sp.S.Zero for _ in row) for row in proof.memberships)
    ).verify()
    assert not replace(proof, pair_reductions=()).verify()
    with pytest.raises(GroebnerVerificationError):
        replace(proof, guard=sp.S.One).specialize({t: 1})


def test_multiple_parameters():
    proof = parametric_groebner_basis([(t + u) * x, y - x], (x, y), (t, u))
    assert proof.verify()
    assert proof.guard == t + u
    assert list(proof.specialize({t: 1, u: -1})) == [x - y]
    assert list(proof.specialize({t: 1, u: 2})) == [x, y]


def test_algebraic_exception():
    alpha = sp.sqrt(2)
    proof = parametric_groebner_basis(
        [(t - alpha) * x], (x,), (t,), domain=sp.QQ.algebraic_field(alpha)
    )
    assert proof.verify()
    assert list(proof.specialize({t: alpha})) == []
    assert list(proof.specialize({t: -alpha})) == [x]


def test_nonstandard_pair_witness():
    from sympy.polys.polyoptions import build_options

    source = (x * y - 1, y * y - 1)
    proof = parametric_groebner_basis(source, (x, y), (t,))
    options = build_options((x, y), {"domain": proof.domain, "order": "grevlex", "polys": False})
    bad_basis = sp.GroebnerBasis._new(
        [sp.Poly(p, x, y, domain=proof.domain) for p in source], options
    )
    forged = replace(
        proof,
        basis=bad_basis,
        memberships=((1, 0), (0, 1)),
        generator_reductions=((1, 0), (0, 1)),
        pair_reductions=((0, 1, (y, -x)),),
        guard=sp.S.One,
        input_guard=sp.S.One,
    )
    # The trivial S-pair identity is true for every pair, including a set that
    # is not a Gröbner basis. Product leading monomials must be strictly lower
    # than the pair LCM for a specialization certificate to prove completeness.
    assert not forged.verify()


def test_supervised_guard():
    from algroots import ExecutionLimits

    proof = parametric_groebner_basis([t * x], (x,), (t,), execution=ExecutionLimits(20))
    assert proof.verify()
    assert list(proof.specialize({t: 0})) == []


def test_wrong_characteristic():
    from sympy.polys.polyoptions import build_options

    proof = parametric_groebner_basis([0], (x,), (t,))
    domain = sp.GF(2).frac_field(t)
    basis = sp.GroebnerBasis._new(
        [], build_options((x,), {"domain": domain, "order": "grevlex", "polys": False})
    )
    forged = replace(proof, source=(2 * x,), domain=domain, basis=basis, generator_reductions=((),))
    assert not forged.verify()
