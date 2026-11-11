"""Exact field snapshots must preserve bound algebraic-root symbols."""

import pickle

import pytest
import sympy as sp

from algroots import AlgebraicFunctionField, QuotientAlgebra, certify_groebner_basis
from algroots._coefficient_state import polynomial_state, restore_polynomial
from algroots.domains import polynomial

x, y, t = sp.symbols("x y t")
alpha = sp.CRootOf(x**3 - x - 1, 0)
base = sp.QQ.algebraic_field(alpha).frac_field(t)
modulus = polynomial(x * x - t - alpha, (x,), base)


def test_polynomial_snapshot():
    restored = restore_polynomial(polynomial_state(modulus))
    assert restored == modulus
    assert restored.gens == (x,)
    assert restored.domain == base


@pytest.mark.parametrize("snapshot", [False, True])
def test_field_snapshot(snapshot):
    field = AlgebraicFunctionField(modulus)
    restored = pickle.loads(pickle.dumps(field)) if snapshot else field
    assert restored == field
    assert restored.convert(x) ** 2 == restored.convert(t + alpha)
    assert restored.rank == 2


def test_prepared_snapshot():
    field = AlgebraicFunctionField(modulus)
    certificate = certify_groebner_basis([y * y - x], (y,), domain=field, backend="sympy")
    family = pickle.loads(pickle.dumps(certificate.prepare()))
    algebra = QuotientAlgebra.from_certificate(family)
    assert algebra.dimension == 2
    assert algebra.normal_form(y * y - x) == 0
    assert algebra.normal_form(y**4) == t + alpha


@pytest.mark.parametrize("snapshot", [False, True])
def test_tower_snapshot(snapshot):
    field = AlgebraicFunctionField(modulus)
    tower = field.extend(polynomial(y**2 - x - 1, (y,), field))
    if snapshot:
        tower = pickle.loads(pickle.dumps(tower))
    assert tower.rank == 4
    assert tower.convert(y) ** 2 == tower.convert(x + 1)
    assert tower.convert(x) ** 2 == tower.convert(t + alpha)
    assert all(proof.verify() for proof in tower.compressions)
    u = sp.Symbol("u")
    enlarged = tower.with_parameters(u)
    assert enlarged.convert(y) ** 4 == enlarged.convert(t + alpha + 2 * x + 1)


@pytest.mark.parametrize("bound", [x, y, t])
def test_bound_root_factorization(bound):
    constant = sp.CRootOf(bound**3 - bound - 1, 0)
    field = AlgebraicFunctionField(
        polynomial(x * x - t - constant, (x,), sp.QQ.algebraic_field(constant).frac_field(t))
    )
    source = polynomial((y - x) * (y + x), (y,), field)
    unit, factors = field.factor_univariate(source)
    assert unit == 1
    assert {(factor.as_expr(), multiplicity) for factor, multiplicity in factors} == {
        (y - x, 1),
        (y + x, 1),
    }
