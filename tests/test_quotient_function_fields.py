"""Exact finite algebras over rational-function coefficient fields."""

import pytest
import sympy as sp
from sympy.polys.matrices import DomainMatrix

from algroots.errors import QuotientAlgebraError
from algroots.quotient import QuotientAlgebra
from algroots.rational_univariate.linear_algebra import packed_krylov_coefficients

x, y, z, w = sp.symbols("x y z w")
K = sp.QQ.frac_field(z)


def test_fraction_field_factories_preserve_domains_and_basis():
    q = QuotientAlgebra.from_polynomials((x**3 + z * x + 1,), (x,), domain=K)
    r = QuotientAlgebra.from_groebner_basis(sp.groebner((x**3 + z * x + 1,), x, domain=K))
    assert q.domain == r.domain == K
    assert q.dimension == r.dimension == 3
    assert q.normal_form(x**3) == -z * x - 1
    assert q.coordinate_vector((x + 1) / z) == sp.Matrix([1 / z, 1 / z, 0])
    assert q.multiplication_matrix(x) == r.multiplication_matrix(x)
    assert q.geometric_solution_count == 3
    assert q.operation_diagnostics["rank_backend"] == "coefficient_domain"


def test_explicit_fraction_field():
    q = QuotientAlgebra.from_polynomials((x * x - 2,), (x,), domain=K)
    assert q.domain == K
    assert q.minimal_polynomial(x, y).domain == K
    assert (
        QuotientAlgebra.from_polynomials((x * x - z,), (x,), domain=sp.ZZ.frac_field(z)).domain == K
    )


def test_algebraic_constant_fraction_field_arithmetic():
    field = sp.QQ.algebraic_field(sp.sqrt(2)).frac_field(z)
    q = QuotientAlgebra.from_polynomials((x * x - z - sp.sqrt(2),), (x,), domain=field)
    assert q.domain == field
    assert q.normal_form(x * x - z - sp.sqrt(2)) == 0
    assert q.normal_form(x * q.inverse(x) - 1) == 0
    assert q.minimal_polynomial(x, y) == sp.Poly(y * y - z - sp.sqrt(2), y, domain=field)
    assert q.trace(x) == 0
    assert q.norm(x) == -z - sp.sqrt(2)
    assert q.is_field


@pytest.mark.parametrize("domain", [sp.RR, sp.CC, sp.EX, sp.GF(9), sp.RR.frac_field(z), "QQ(z)"])
def test_unsupported_coefficient_domains_remain_rejected(domain):
    with pytest.raises(QuotientAlgebraError):
        QuotientAlgebra.from_polynomials((x * x - 2,), (x,), domain=domain)


def test_parameter_collision_invalid_variables_and_float_inputs():
    with pytest.raises(QuotientAlgebraError, match="distinct"):
        QuotientAlgebra.from_polynomials((x * x - 2,), (x,), domain=sp.QQ.frac_field(x))
    with pytest.raises(QuotientAlgebraError, match="distinct"):
        QuotientAlgebra.from_polynomials((x * x - 2,), (x, x), domain=K)
    with pytest.raises(QuotientAlgebraError, match="Symbols"):
        QuotientAlgebra.from_polynomials((x * x - 2,), (x + 1,), domain=K)
    with pytest.raises(QuotientAlgebraError, match="floating"):
        QuotientAlgebra.from_polynomials((x * x - 0.5,), (x,), domain=K)
    q = QuotientAlgebra.from_polynomials((x * x - 2,), (x,), domain=K)
    with pytest.raises(QuotientAlgebraError, match="exact"):
        q.inverse(x + sp.Float("0.1"))
    with pytest.raises(QuotientAlgebraError, match="reduction"):
        q.normal_form(sp.pi * x)
    with pytest.raises(QuotientAlgebraError, match="at least one polynomial"):
        QuotientAlgebra.from_polynomials((), (x,), domain=K)


def test_element_invariants_and_inverse_preserve_exact_base():
    q = QuotientAlgebra.from_polynomials((x**3 + z * x + 1,), (x,), domain=K)
    t = sp.Symbol("t", real=True)
    assert q.minimal_polynomial(x, t) == sp.Poly(t**3 + z * t + 1, t, domain=K)
    assert q.characteristic_polynomial(x, t) == q.minimal_polynomial(x, t)
    assert q.trace(x) == 0 and q.norm(x) == -1
    assert q.trace(x + 1) == 3 and q.norm(x + 1) == z
    assert q.inverse(x) == -x * x - z
    assert q.normal_form((x + 1) * q.inverse(x + 1) - 1) == 0
    assert q.specialization_guard(q.inverse(x + 1)).subs(z, 0) is sp.false
    assert q.separating_element(t).defining_polynomial == q.minimal_polynomial(x, t)
    with pytest.raises(QuotientAlgebraError):
        q.minimal_polynomial(x, z)
    with pytest.raises(QuotientAlgebraError):
        q.characteristic_polynomial(x, x)


def test_domain_matrix_actions_are_exact_homomorphisms():
    q = QuotientAlgebra.from_polynomials((x * x - z, y * y - (z + 1)), (x, y), domain=K)
    a = (x + y) / (z + 1)
    b = x * y - z
    A = q.multiplication_matrix(a, representation="domain")
    B = q.multiplication_matrix(b, representation="domain")
    assert isinstance(A, DomainMatrix) and A.domain == K
    assert q.multiplication_matrix(a * b, representation="domain") == A * B
    assert q.multiplication_matrix(a + b, representation="domain") == A + B
    assert sp.cancel(q.norm(a * b) - q.norm(a) * q.norm(b)) == 0
    assert sp.cancel(q.trace(a + b) - q.trace(a) - q.trace(b)) == 0
    with pytest.raises(ValueError, match="representation"):
        q.multiplication_matrix(x, representation="approximate")


def test_proper_subfield_short_closure_is_not_characteristic_polynomial():
    q = QuotientAlgebra.from_polynomials((x**4 - z,), (x,), domain=K)
    assert q.minimal_polynomial(x * x, y) == sp.Poly(y * y - z, y, domain=K)
    assert q.characteristic_polynomial(x * x, y) == sp.Poly((y * y - z) ** 2, y, domain=K)
    assert q.norm(x * x) == z * z
    assert q.minimal_polynomial(2, y) == sp.Poly(y - 2, y, domain=K)
    assert q.norm(2) == 16
    assert q.minimal_polynomial(0, y) == sp.Poly(y, y, domain=K)
    with pytest.raises(ValueError, match="closed before"):
        packed_krylov_coefficients(
            q.multiplication_matrix(x * x, representation="domain").to_list(), K
        )


@pytest.mark.parametrize("domain", [sp.QQ, K])
def test_nonreduced_and_reducible_algebras_do_not_claim_to_be_fields(domain):
    q = QuotientAlgebra.from_polynomials((x**3,), (x,), domain=domain)
    assert not q.is_field
    assert q.minimal_polynomial(x, y) == sp.Poly(y**3, y, domain=domain)
    assert q.separating_element(y).defining_polynomial == sp.Poly(y, y, domain=domain)
    assert q.trace(x) == q.norm(x) == 0
    assert q.inverse(1 + x) == 1 - x + x * x
    with pytest.raises(QuotientAlgebraError, match="not a unit"):
        q.inverse(x)
    r = QuotientAlgebra.from_polynomials((x * x - 1,), (x,), domain=domain)
    assert not r.is_field
    assert r.inverse(x) == x
    with pytest.raises(QuotientAlgebraError, match="not a unit"):
        r.inverse(x - 1)


def test_specialization_rebuilds_exceptional_fibers_from_source():
    q = QuotientAlgebra.from_polynomials((z * x - 1,), (x,), domain=K)
    assert q.dimension == 1
    assert q.specialize({z: 2}).normal_form(x) == sp.Rational(1, 2)
    assert q.specialize({z: 0}).dimension == 0
    basis_only = QuotientAlgebra.from_groebner_basis(q.groebner_basis)
    with pytest.raises(QuotientAlgebraError, match="exceptional fiber"):
        basis_only.specialize({z: 0})
    with pytest.raises(QuotientAlgebraError, match="original defining"):
        basis_only.specialize({z: 2})
    r = QuotientAlgebra.from_polynomials((z * x,), (x,), domain=K)
    with pytest.raises(QuotientAlgebraError, match="zero-dimensional"):
        r.specialize({z: 0})
    pole = QuotientAlgebra.from_polynomials((x - 1 / z,), (x,), domain=K)
    with pytest.raises(QuotientAlgebraError, match="pole"):
        pole.specialize({z: 0})


def test_partial_specialization_ramification():
    q = QuotientAlgebra.from_polynomials((x * x - z - w,), (x,), domain=sp.QQ.frac_field(z, w))
    partial = q.specialize({w: 0})
    assert partial.domain == K
    singular = partial.specialize({z: 0})
    assert singular.dimension == 2 and singular.geometric_solution_count == 1
    with pytest.raises(QuotientAlgebraError, match="keys"):
        q.specialize({x: 0})
    with pytest.raises(QuotientAlgebraError, match="exact"):
        q.specialize({z: 0.5})


def test_parameter_symbol_identity_and_empty_algebra_conventions():
    q = QuotientAlgebra.from_polynomials((x * x - 2,), (x,))
    for t in (sp.Symbol("t", real=True), sp.Dummy("t", positive=True)):
        assert q.separating_element(t).defining_polynomial == sp.Poly(t * t - 2, t, domain=sp.QQ)
    empty = QuotientAlgebra.from_polynomials((1,), (x,), domain=K)
    assert empty.trace(x) == 0 and empty.norm(x) == 1
    assert (
        empty.minimal_polynomial(x, y)
        == empty.characteristic_polynomial(x, y)
        == sp.Poly(1, y, domain=K)
    )
    assert not empty.is_field
    with pytest.raises(QuotientAlgebraError, match="zero algebra"):
        empty.inverse(x)
