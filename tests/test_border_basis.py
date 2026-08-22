from __future__ import annotations

import sympy as sp

from algroots import BorderBasisError, compute_border_basis, compute_border_basis_linear


def test_border_basis_for_two_reduced_points():
    x, y = sp.symbols("x y")
    result = compute_border_basis([x**2 - 1, y - x], [x, y])

    assert result.dimension == 2
    assert result.order_monomials == (sp.Integer(1), x)
    assert all(poly.as_expr() != 0 for poly in result.border_polynomials)
    assert result.has_commuting_multiplication_matrices()
    assert result.multiplication_matrix(x).shape == (2, 2)
    assert result.multiplication_matrix(y) == result.multiplication_matrix(x)


def test_border_basis_records_multiplicity_for_non_radical_ideal():
    x, y = sp.symbols("x y")
    result = compute_border_basis([x**2, y - 1], [x, y])

    assert result.dimension == 2
    assert result.order_monomials == (sp.Integer(1), x)
    assert sp.Poly(y - 1, x, y, domain=sp.QQ) in result.border_polynomials
    assert result.has_commuting_multiplication_matrices()


def test_border_basis_rejects_positive_dimensional_ideal():
    x, y = sp.symbols("x y")
    try:
        compute_border_basis([x**2 + y**2 - 1], [x, y])
    except BorderBasisError as exc:
        assert "zero-dimensional" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("positive-dimensional ideal unexpectedly accepted")


def test_border_basis_normal_form_and_coordinates():
    x, y = sp.symbols("x y")
    result = compute_border_basis([x**2 - 1, y - x], [x, y])

    assert result.normal_form(y) == x
    assert result.normal_form(y**2 + y) == x + 1
    assert result.coordinates(y) == sp.Matrix([0, 1])
    assert result.multiplication_matrix(x + y) == 2 * result.multiplication_matrix(x)
    assert result.border_reduction_matrix.shape[0] == result.dimension
    assert result.normal_form_matrix.rank() == result.dimension
    assert result.diagnostics.success
    assert result.diagnostics.quotient_basis_rank == result.dimension
    assert result.diagnostics.commutators_zero is True


def test_border_basis_strict_false_reports_invalid_order_ideal():
    x, y = sp.symbols("x y")
    result = compute_border_basis([x**2 - 1, y - x], [x, y], order_ideal=[1, x * y], strict=False)

    assert not result.diagnostics.success
    assert result.diagnostics.messages
    assert "order ideal" in result.diagnostics.messages[0]


def test_border_basis_strict_false_reports_positive_dimensional_ideal():
    x, y = sp.symbols("x y")
    result = compute_border_basis([x**2 + y**2 - 1], [x, y], strict=False)

    assert not result.diagnostics.success
    assert "zero-dimensional" in result.diagnostics.messages[0]


def test_linear_border_basis_matches_groebner_derived_basis():
    x, y = sp.symbols("x y")
    groebner_result = compute_border_basis([x**2 - 1, y - x], [x, y])
    linear_result = compute_border_basis([x**2 - 1, y - x], [x, y], algorithm="linear")

    assert linear_result.source == "macaulay-linear-algebra"
    assert linear_result.dimension == groebner_result.dimension
    assert linear_result.order_monomials == groebner_result.order_monomials
    assert set(linear_result.as_exprs()) == set(groebner_result.as_exprs())
    assert linear_result.has_commuting_multiplication_matrices()
    assert linear_result.multiplication_matrix(x) == groebner_result.multiplication_matrix(x)


def test_linear_border_basis_public_helper():
    x, y = sp.symbols("x y")
    result = compute_border_basis_linear([x**2, y - 1], [x, y])

    assert result.source == "macaulay-linear-algebra"
    assert result.dimension == 2
    assert result.order_monomials == (sp.Integer(1), x)
    assert result.has_commuting_multiplication_matrices()


def test_linear_border_basis_strict_false_reports_non_stabilization():
    x, y = sp.symbols("x y")
    result = compute_border_basis(
        [x**2 - 1, y - x], [x, y], algorithm="linear", max_degree=1, strict=False
    )

    assert not result.diagnostics.success
    assert "did not stabilize" in result.diagnostics.messages[0]


def test_border_basis_coordinate_conversion_does_not_swallow_programming_errors(monkeypatch):
    x, y = sp.symbols("x y")
    result = compute_border_basis([x**2 - 1, y - x], [x, y])
    original = sp.Poly

    def crash(*args, **kwargs):
        raise RuntimeError("synthetic programming error")

    monkeypatch.setattr(sp, "Poly", crash)
    try:
        with __import__("pytest").raises(RuntimeError, match="synthetic"):
            result._coordinates_from_multiplication_matrices(x)
    finally:
        monkeypatch.setattr(sp, "Poly", original)
