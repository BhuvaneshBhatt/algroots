"""Irreducibility, primitive maps, modular images, and exact tower fibers."""

import pickle
from dataclasses import replace

import pytest
import sympy as sp

from algroots import (
    AlgebraicFunctionField,
    GroebnerLimitError,
    GroebnerVerificationError,
    QuotientAlgebra,
    groebner_basis,
)
from algroots.finite_decomposition import primary_decomposition
from algroots.native_f4 import compiled_available

x, y, t, u, a, b, c = sp.symbols("x y t u a b c")


def quadratic():
    return AlgebraicFunctionField(sp.Poly(a * a - t, a, domain=sp.QQ.frac_field(t)))


def test_tower_relations():
    K = quadratic()
    L = K.extend(sp.Poly(b * b - a - 1, b, domain=K))
    assert L.rank == 4
    assert all(proof.verify() for proof in L.compressions)
    assert L.convert(b * b - a - 1) == L.zero
    assert L.convert(a * a - t) == L.zero
    assert L.convert(K.convert(a + 1)) == L.convert(a + 1)
    assert L.convert(1 / (b - a)) * L.convert(b - a) == L.one
    assert not replace(L.compressions[0], old_image=sp.S.Zero).verify()
    restored = pickle.loads(pickle.dumps(L))
    assert restored.convert(a * a - t) == restored.zero


def test_three_levels():
    K = quadratic()
    L = K.extend(sp.Poly(b * b - a - 1, b, domain=K))
    M = L.extend(sp.Poly(c * c - b - 1, c, domain=L))
    assert M.rank == 8
    assert M.convert(c * c - b - 1) == M.zero
    embedding = M.specialize({t: 4}, roots={a: 2, b: sp.sqrt(3), c: sp.sqrt(1 + sp.sqrt(3))})
    assert embedding(a) == 2
    assert embedding(b) == sp.sqrt(3)
    assert sp.simplify(embedding(c) ** 2 - embedding(b) - 1) == 0


def test_reducible_and_budgets():
    with pytest.raises(ValueError, match="reducible"):
        AlgebraicFunctionField(sp.Poly(a * a - t * t, a, domain=sp.QQ.frac_field(t)))
    K = quadratic()
    with pytest.raises(ValueError, match="reducible"):
        K.extend(sp.Poly(b * b - t, b, domain=K))
    with pytest.raises(GroebnerLimitError, match="degree"):
        K.extend(sp.Poly(b * b - a, b, domain=K), max_degree=3)
    with pytest.raises(sp.polys.polyerrors.CoercionFailed, match="embedding"):
        K.convert(
            AlgebraicFunctionField(sp.Poly(a * a - t - 1, a, domain=sp.QQ.frac_field(t))).generator
        )


def test_specialization_validation():
    K = quadratic()
    for values, roots in [({t: 1}, {a: 2}), ({}, {a: 1}), ({t: 1}, {}), ({t: 1.0}, {a: 1})]:
        with pytest.raises(ValueError):
            K.specialize(values, roots=roots)
    embedding = K.specialize({t: 4}, roots={a: -2})
    assert embedding(a) == -2
    assert embedding(1 / (a + 1)) == -1
    with pytest.raises(ValueError, match="pole"):
        embedding(1 / (t - 4))
    bad = replace(embedding, primitive_image=sp.S.One)
    assert not bad.verify()
    with pytest.raises(GroebnerVerificationError):
        bad(a)


def test_factorization_and_components():
    K = quadratic()
    p = sp.Poly((x - a) ** 2 * (x + a) ** 3, x, domain=K)
    unit, factors = K.factor_univariate(p)
    assert unit == 1
    assert {(f.as_expr(), m) for f, m in factors} == {(x - a, 2), (x + a, 3)}
    q = QuotientAlgebra.from_polynomials([p.as_expr()], (x,), domain=K)
    assert not q.is_field
    pieces = primary_decomposition(q)
    assert sorted((component.dimension, residue.dimension) for component, residue in pieces) == [
        (2, 1),
        (3, 1),
    ]
    assert QuotientAlgebra.from_polynomials([x * x - a], (x,), domain=K).is_field


@pytest.mark.parametrize("order", ["lex", "grlex", "grevlex"])
@pytest.mark.parametrize("kind", ["single", "tower", "mixed", "two_parameters"])
def test_native_tower(kind, order):
    if not compiled_available():
        pytest.skip("native F4 is not installed")
    if kind == "mixed":
        K = AlgebraicFunctionField(
            sp.Poly(
                a * a - t - sp.sqrt(2), a, domain=sp.QQ.algebraic_field(sp.sqrt(2)).frac_field(t)
            )
        )
    elif kind == "two_parameters":
        K = AlgebraicFunctionField(sp.Poly(a * a - t - u, a, domain=sp.QQ.frac_field(t, u)))
    else:
        K = quadratic()
        if kind == "tower":
            K = K.extend(sp.Poly(b * b - a - 1, b, domain=K))
    root = K.tower_symbols[-1]
    source = [x * y - root, y * y - x - 1]
    native = groebner_basis(source, (x, y), domain=K, order=order, backend="native")
    reference = groebner_basis(source, (x, y), domain=K, order=order, backend="sympy")
    assert native == reference
    q = QuotientAlgebra.from_polynomials(
        source, (x, y), domain=K, order=order, groebner_backend="native"
    )
    assert q.dimension == 3
    assert q.normal_form(x * y - root) == 0


def test_coordinate_collision():
    K = quadratic()
    with pytest.raises(ValueError, match="distinct"):
        QuotientAlgebra.from_polynomials([a * a - 1], (a,), domain=K)


def test_bound_root_coefficient():
    root = sp.CRootOf(x**3 - x - 1, 0)
    field = AlgebraicFunctionField(
        sp.Poly(a * a - t - root, a, domain=sp.QQ.algebraic_field(root).frac_field(t))
    )
    basis = groebner_basis([x * x - a + root], (x,), domain=field, backend="sympy")
    assert basis.polys[0].as_expr() == x * x - a + root


def test_native_tower_budget():
    from sympy.polys.rings import PolyRing

    from algroots import NativeF4Limits
    from algroots.domains import polynomial
    from algroots.native_fields import native_field_basis

    K = quadratic()
    ring = PolyRing((x,), K, "grevlex")
    source = ring.from_dict(polynomial(x * x - a, (x,), K).rep.to_dict())
    with pytest.raises(GroebnerLimitError, match="tower degree"):
        native_field_basis([source], ring, limits=NativeF4Limits(max_field_degree=1))


def test_unvalidated_extension():
    from sympy.polys.agca.extensions import MonogenicFiniteExtension

    fake = MonogenicFiniteExtension(sp.Poly(a * a - t * t, a, domain=sp.QQ.frac_field(t)))
    with pytest.raises(ValueError, match="validated"):
        groebner_basis([x - a], (x,), domain=fake, backend="sympy")
