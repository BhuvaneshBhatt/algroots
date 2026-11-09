"""Exact sparse actions, cyclic dependencies and pivoted one-column solves."""

import random

import pytest
import sympy as sp
from sympy.polys.matrices import DomainMatrix
from sympy.polys.matrices.exceptions import DMNonInvertibleMatrixError

from algroots import AlgebraicFunctionField, QuotientAlgebra
from algroots.domains import domain_matrix
from algroots.rational_univariate.linear_algebra import (
    packed_krylov_coefficients,
    sparse_solve,
)

x, y, t, a, z = sp.symbols("x y t a z")
FIELDS = [
    sp.QQ,
    sp.GF(7),
    sp.QQ.algebraic_field(sp.sqrt(2)),
    sp.QQ.frac_field(t),
    AlgebraicFunctionField(sp.Poly(a * a - t, a, domain=sp.QQ.frac_field(t))),
]


@pytest.mark.parametrize("domain", FIELDS)
@pytest.mark.parametrize("seed", range(8))
def test_pivoted_solve(domain, seed):
    rng = random.Random(seed)
    n = 7
    matrix = sp.zeros(n)
    for i in range(n):
        matrix[i, (i + 1) % n] = 1
        if i + 2 < n:
            matrix[i, i + 2] = rng.randrange(-2, 3)
    native = domain_matrix(matrix, domain)
    expected = domain_matrix(sp.Matrix([rng.randrange(-5, 6) for _ in range(n)]), domain)
    rhs = native * expected
    original = native.to_dod()
    assert sparse_solve(native, rhs) == expected
    assert native.to_dod() == original


@pytest.mark.parametrize("domain", FIELDS)
def test_singular_consistent_rhs(domain):
    matrix = domain_matrix(sp.Matrix([[0, 1], [0, 2]]), domain)
    rhs = domain_matrix(sp.Matrix([3, 6]), domain)
    with pytest.raises(DMNonInvertibleMatrixError):
        sparse_solve(matrix, rhs)


@pytest.mark.parametrize("domain", FIELDS)
@pytest.mark.parametrize("degree", [1, 2, 9, 18])
def test_cyclic_polynomial(domain, degree):
    q = QuotientAlgebra.from_polynomials([x**degree - 2 * x + 1], (x,), domain=domain)
    matrix = q.multiplication_matrix(x, representation="domain")
    coefficients = packed_krylov_coefficients(matrix, domain)
    expected = sp.Poly(x**degree - 2 * x + 1, x, domain=domain).monic()
    assert sp.Poly.from_list(list(reversed(coefficients)), x, domain=domain) == expected
    assert q.minimal_polynomial(x, z) == sp.Poly(expected.as_expr().subs(x, z), z, domain=domain)


def test_sparse_large_short_closure():
    # A dense expansion here would require billions of entries.
    n = 100000
    matrix = DomainMatrix.eye((n, n), sp.QQ)
    assert packed_krylov_coefficients(matrix, sp.QQ, allow_short_closure=True) == (
        -sp.QQ.one,
        sp.QQ.one,
    )
    with pytest.raises(ValueError, match="before full dimension"):
        packed_krylov_coefficients(matrix, sp.QQ)


def test_sparse_large_solve():
    n = 10000
    matrix = DomainMatrix({i: {i: sp.QQ.one} for i in range(n)}, (n, n), sp.QQ)
    rhs = DomainMatrix({n - 1: {0: sp.QQ(7)}}, (n, 1), sp.QQ)
    assert sparse_solve(matrix, rhs) == rhs


@pytest.mark.parametrize("domain", FIELDS)
def test_quotient_actions(domain):
    q = QuotientAlgebra.from_polynomials(
        [x**5 - t if domain.is_FractionField else x**5 - 2, y * y - x], (x, y), domain=domain
    )
    mx, my = q.variable_multiplication_matrices
    assert domain_matrix(my, domain).pow(2) == domain_matrix(mx, domain)
    assert q.normal_form(q.inverse(x + 1) * (x + 1) - 1) == 0
    assert q.multiplication_matrix(x + y, representation="sparse") == mx + my
    assert q.norm(x + 1) == domain.to_sympy(
        q.multiplication_matrix(x + 1, representation="domain").det()
    )


def test_noncyclic_nilpotents():
    q = QuotientAlgebra.from_polynomials([x * x, x * y, y * y], (x, y))
    assert q.minimal_polynomial(x + y, z) == sp.Poly(z * z, z, domain=sp.QQ)
    assert q.characteristic_polynomial(x + y, z) == sp.Poly(z**3, z, domain=sp.QQ)
    assert q.geometric_solution_count == 1
