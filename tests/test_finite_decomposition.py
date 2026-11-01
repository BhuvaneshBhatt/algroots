import sympy as sp

from algroots.finite_decomposition import nilradical_basis, primary_decomposition, reduced_algebra
from algroots.quotient import QuotientAlgebra

x, y, t = sp.symbols("x y t")


def test_nonmonogenic_local_algebra():
    algebra = QuotientAlgebra.from_polynomials((x**2, x * y, y**2), (x, y))
    basis = nilradical_basis(algebra)
    assert len(basis) == 2
    assert all(algebra.normal_form(b**2) == 0 for b in basis)
    assert sp.Matrix([[sp.expand(b).coeff(x), sp.expand(b).coeff(y)] for b in basis]).rank() == 2
    reduced = reduced_algebra(algebra)
    assert reduced.dimension == 1
    assert reduced.normal_form(x) == reduced.normal_form(y) == 0
    ((component, residue),) = primary_decomposition(algebra)
    assert component.dimension == 3
    assert residue.dimension == 1
    assert component.normal_form(x * y) == 0


def test_primary_lengths():
    algebra = QuotientAlgebra.from_polynomials(((x**2 + 1) ** 2, y**2), (x, y))
    ((component, residue),) = primary_decomposition(algebra)
    assert component.dimension == 8
    assert residue.dimension == 2
    assert residue.is_field
    field = sp.QQ.algebraic_field(sp.I).frac_field(t)
    algebra = QuotientAlgebra.from_polynomials(((x - t) ** 2 * (x + t), y**2), (x, y), domain=field)
    factors = primary_decomposition(algebra)
    assert sorted(a.dimension for a, _ in factors) == [2, 4]
    assert all(k.is_field and k.dimension == 1 for _, k in factors)
    assert sum(a.dimension for a, _ in factors) == algebra.dimension


def test_empty_algebra():
    algebra = QuotientAlgebra.from_polynomials((1,), (x,))
    assert nilradical_basis(algebra) == ()
    assert reduced_algebra(algebra).dimension == 0
    assert primary_decomposition(algebra) == ()
