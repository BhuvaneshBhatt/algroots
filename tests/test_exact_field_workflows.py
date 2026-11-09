"""Sparse invariant factors, verified snapshots and local tower specializations."""

import pickle
import random
from dataclasses import replace

import pytest
import sympy as sp
from sympy.polys.matrices import DomainMatrix

from algroots import (
    AlgebraicFunctionField,
    GroebnerLimitError,
    NativeF4Limits,
    QuotientAlgebra,
    certify_groebner_basis,
    certify_rational_univariate_representation,
    groebner_basis,
)
from algroots.domains import domain_matrix
from algroots.rational_univariate.linear_algebra import sparse_characteristic_coefficients

x, y, t, a, b, c, z = sp.symbols("x y t a b c z")


def field():
    return AlgebraicFunctionField(sp.Poly(a * a - t, a, domain=sp.QQ.frac_field(t)))


@pytest.mark.parametrize(
    "domain", [sp.QQ, sp.GF(7), sp.QQ.algebraic_field(sp.sqrt(2)), sp.QQ.frac_field(t), field()]
)
@pytest.mark.parametrize("seed", range(6))
def test_sparse_characteristic(domain, seed):
    rng = random.Random(seed)
    matrix = sp.Matrix(7, 7, lambda i, j: rng.randrange(-2, 3) if rng.randrange(4) == 0 else 0)
    native = domain_matrix(matrix, domain)
    assert sparse_characteristic_coefficients(native) == native.charpoly()


def test_noncyclic_sparse_characteristic(monkeypatch):
    n = 200
    matrix = DomainMatrix({i: {i: sp.QQ(i % 3)} for i in range(n) if i % 3}, (n, n), sp.QQ)

    def forbidden(*args, **kwargs):
        pytest.fail("dense matrix expansion")

    monkeypatch.setattr(DomainMatrix, "to_dense", forbidden)
    monkeypatch.setattr(DomainMatrix, "to_list", forbidden)
    coefficients = sparse_characteristic_coefficients(matrix)
    expected = sp.Poly(z**67 * (z - 1) ** 67 * (z - 2) ** 66, z, domain=sp.QQ)
    assert sp.Poly.from_list(coefficients, z, domain=sp.QQ) == expected


def test_sparse_trace_pairing():
    q = QuotientAlgebra.from_polynomials([x**20, y**20], (x, y))
    assert isinstance(q.trace_pairing, sp.SparseMatrix)
    assert q.trace_pairing.todok() == {(0, 0): sp.Integer(400)}
    assert q.geometric_solution_count == 1
    assert q.operation_diagnostics["basis_stream_peak_entries"] <= 8


def test_prepared_basis_is_detached(monkeypatch):
    proof = certify_groebner_basis([t * x, y - x], (x, y), domain=sp.QQ.frac_field(t))
    family = proof.prepare()
    object.__setattr__(proof, "source", (x, y - 1))
    monkeypatch.setattr(type(proof), "verify", lambda self: pytest.fail("generic replay"))
    assert set(family.specialize({t: 1})) == {x, y}
    assert set(family.specialize({t: 0})) == {x - y}
    assert QuotientAlgebra.from_certificate(family).dimension == 1
    assert (
        groebner_basis(
            [t * x, y - x], (x, y), domain=sp.QQ.frac_field(t), certificate=family, backend="sympy"
        )
        == family.verified_basis()
    )
    with pytest.raises(AttributeError):
        family.guard = 1


def test_prepared_rur_is_detached(monkeypatch):
    proof = certify_rational_univariate_representation([x * x - t], (x,), parameters=(t,))
    family = proof.prepare()
    restored = pickle.loads(pickle.dumps(family))
    assert restored.specialize({t: 2}).solution_count == 2
    object.__setattr__(proof.basis_certificate, "source", (x - 9,))
    monkeypatch.setattr(type(proof), "verify", lambda self: pytest.fail("generic RUR replay"))
    assert family.specialize({t: 1}).solution_count == 2
    assert family.specialize({t: 0}).solution_count == 1
    assert family.specialize({t: 0}).dimension == 2


def test_coordinate_denominator_guard():
    proof = certify_rational_univariate_representation([x * x - 1], (x,), parameters=(t,))
    r = proof.representation
    u = r.parameter
    D = r.defining_polynomial.domain
    candidate = replace(
        r,
        coordinate_denominator=sp.Poly(t * u - 1, u, domain=D),
        coordinate_numerators=(sp.Poly(t - u, u, domain=D),),
    )
    checked = certify_rational_univariate_representation(
        [x * x - 1],
        (x,),
        parameters=(t,),
        candidate=candidate,
        basis_certificate=proof.basis_certificate,
    )
    assert checked.verify()
    assert checked.guard.subs(t, 1) == 0
    for result in (checked.specialize({t: 1}), checked.prepare().specialize({t: 1})):
        assert result.solution_count == 2
        assert result.normalized_coordinate_polynomials() == (
            sp.Poly(result.parameter, result.parameter, domain=sp.QQ),
        )


def test_compressed_pole_recovery():
    K = field()
    L = K.extend(sp.Poly(b * b - a - 1, b, domain=K))
    roots = {a: sp.sqrt(3) / 2, b: (1 + sp.sqrt(3)) / 2}
    embedding = L.specialize({t: sp.Rational(3, 4)}, roots=roots)
    assert L.original_expression(b) == b
    assert embedding(b) == roots[b]
    assert embedding(a) == roots[a]
    proof = certify_rational_univariate_representation([x * x - b], (x,), domain=L)
    result = proof.prepare().specialize_field(embedding)
    assert result.solution_count == 2
    assert (
        result.defining_polynomial.domain.convert(result.defining_polynomial.TC() + roots[b])
        == result.defining_polynomial.domain.zero
    )


def test_local_regular_branch():
    K = field()
    positive = K.specialize({t: 1}, roots={a: 1})
    negative = K.specialize({t: 1}, roots={a: -1})
    native = K.convert(1 / (a + 1))
    assert positive(native) == sp.Rational(1, 2)
    with pytest.raises(ValueError, match="pole"):
        negative(native)
    with pytest.raises(GroebnerLimitError):
        positive = K.specialize({t: 1}, roots={a: 1}, limits=NativeF4Limits(max_basis=1))
        positive(native)


def test_branch_enumeration():
    K = field()
    L = K.extend(sp.Poly(b * b - a - 1, b, domain=K))
    branches = L.specializations({t: 4})
    assert len(branches) == 4
    assert len(L.specializations({t: 4}, real=True)) == 2
    assert len(L.specializations({t: 0})) == 2
    for embedding in branches:
        assert embedding(a * a) == 4
        assert embedding(b * b - a - 1) == 0
    with pytest.raises(GroebnerLimitError, match="branch"):
        L.specializations({t: 4}, max_branches=3)


def test_parameters_after_tower():
    K = field()
    L = K.extend(sp.Poly(b * b - a - 1, b, domain=K))
    u = sp.Symbol("u")
    D = L.with_parameters(u)
    assert D.parameters == (t, u)
    assert D.convert(b * b - a - 1) == D.zero
    assert D.convert(L.convert(b + a)) == D.convert(b + a)
    assert all(proof.verify() for proof in D.compressions)
    for coefficient_field in (D, L.frac_field(u)):
        q = QuotientAlgebra.from_polynomials([x * x - b - u], (x,), domain=coefficient_field)
        assert q.domain == D
        assert q.normal_form(x * x - b - u) == 0
    assert D.specialize({t: 4, u: 3}, roots={a: 2, b: sp.sqrt(3)})(b + u) == sp.sqrt(3) + 3


def test_primitive_coordinate_collision():
    K = field()
    L = K.extend(sp.Poly(b * b - a - 1, b, domain=K))
    with pytest.raises(ValueError, match="distinct"):
        QuotientAlgebra.from_polynomials([L.symbol], (L.symbol,), domain=L)


def test_constant_prepared_field():
    proof = certify_groebner_basis([x - 5], (x,), domain=sp.GF(7)).prepare()
    result = proof.specialize({})
    assert result.domain == sp.GF(7)
    assert list(result) == [x + 2]


def test_inferred_rur_parameters():
    proof = certify_rational_univariate_representation(
        [x * x - t], (x,), domain=sp.QQ.frac_field(t)
    )
    assert proof.parameters == (t,)
    assert proof.guard == t
    assert proof.prepare().specialize({t: 0}).solution_count == 1


def test_regular_prefix_coefficients():
    K = field()
    L = K.extend(sp.Poly(b * b - 1 / (a + 1), b, domain=K))
    embedding = L.specialize({t: 1}, roots={a: 1, b: sp.sqrt(2) / 2})
    assert embedding(b) == sp.sqrt(2) / 2
    assert embedding(1 / (a + 1)) == sp.Rational(1, 2)
    branches = L.specializations({t: 1})
    assert len(branches) == 2
    assert all(e(a) == 1 for e in branches)
    assert all(e(b * b) == sp.Rational(1, 2) for e in branches)


def test_tower_certificate_persistence():
    K = field()
    L = K.extend(sp.Poly(b * b - a - 1, b, domain=K))
    proof = pickle.loads(pickle.dumps(L.compressions[0]))
    assert proof.verify()
    embedding = L.specialize({t: 4}, roots={a: 2, b: sp.sqrt(3)})
    restored = pickle.loads(pickle.dumps(embedding))
    assert restored.verify()
    assert restored(b) == sp.sqrt(3)


def test_prepared_result_domain_is_detached():
    K = field()
    family = certify_rational_univariate_representation([x * x - a], (x,), domain=K).prepare()
    result = family.verified_representation()
    result.defining_polynomial.domain.images = ((a, sp.S.Zero),)
    embedding = K.specialize({t: 4}, roots={a: 2})
    assert family.specialize_field(embedding).solution_count == 2
    assert family.verified_representation().defining_polynomial.domain.convert(a) != K.zero


def test_sparse_norm_avoids_dense_determinant(monkeypatch):
    q = QuotientAlgebra.from_polynomials([x**16 - 2], (x,))
    monkeypatch.setattr(DomainMatrix, "det", lambda self: pytest.fail("dense determinant"))
    assert q.norm(x) == -2
    assert q.norm(x + 1) == -1
