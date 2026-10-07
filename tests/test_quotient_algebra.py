from __future__ import annotations

import pytest
import sympy as sp

from algroots.border_basis import compute_border_basis
from algroots.errors import QuotientAlgebraError
from algroots.quotient import QuotientAlgebra, standard_exponent_count, standard_exponents
from algroots.rational_univariate import compute_rational_univariate_representation


def test_quotient_algebra_exposes_exact_structural_operations() -> None:
    x, y = sp.symbols("x y")
    quotient = QuotientAlgebra.from_polynomials((x**2 - 1, y - x), (x, y))

    assert quotient.dimension == 2
    assert quotient.normal_form(x) == y
    assert quotient.coordinate_vector(y**2 + y) == sp.Matrix([1, 1])
    assert quotient.multiplication_matrix(x) == quotient.multiplication_matrix(y)
    assert quotient.trace_pairing.rank() == 2
    assert quotient.geometric_solution_count == 2


def test_quotient_algebra_normalizes_exact_algebraic_coefficient_fields() -> None:
    x, y = sp.symbols("x y")
    quotient = QuotientAlgebra.from_polynomials((x - sp.sqrt(2), y - sp.sqrt(3)), (x, y))

    assert getattr(quotient.domain, "is_AlgebraicField", False)
    assert quotient.dimension == 1
    assert quotient.normal_form(x + y) == sp.sqrt(2) + sp.sqrt(3)


def test_quotient_algebra_normalizes_gaussian_coefficients_to_algebraic_field() -> None:
    x = sp.Symbol("x")
    quotient = QuotientAlgebra.from_polynomials((x**2 - (1 + sp.I),), (x,))

    assert getattr(quotient.domain, "is_AlgebraicField", False)
    assert quotient.dimension == 2


def test_standard_exponent_count_matches_materialized_staircase() -> None:
    leading = ((2, 0, 0), (0, 3, 0), (0, 0, 2), (1, 1, 0))

    assert standard_exponent_count(leading, 3) == len(standard_exponents(leading, 3)) == 8


def test_quotient_dimension_guard_is_enforced_before_public_use() -> None:
    x, y, z = sp.symbols("x y z")
    with pytest.raises(QuotientAlgebraError, match="max_dimension=7"):
        QuotientAlgebra.from_polynomials(
            (x**2 - 2, y**2 - 3, z**2 - 5),
            (x, y, z),
            max_dimension=7,
        )


def test_quotient_algebra_rejects_positive_dimensional_ideal() -> None:
    x, y = sp.symbols("x y")
    with pytest.raises(QuotientAlgebraError, match="zero-dimensional"):
        QuotientAlgebra.from_polynomials((x**2 + y**2 - 1,), (x, y))


def test_separator_retries_when_first_linear_form_does_not_separate() -> None:
    x, y, t = sp.symbols("x y t")
    quotient = QuotientAlgebra.from_polynomials((x * (x - 1), x + y), (x, y))

    with pytest.raises(QuotientAlgebraError, match="separating linear form"):
        quotient.separating_element(t, max_attempts=1)

    separator = quotient.separating_element(t, max_attempts=2)
    assert separator.coefficients == (1, 2)
    assert separator.geometric_solution_count == 2
    assert separator.defining_polynomial.degree() == 2


def test_rur_border_basis_and_quotient_share_one_dimension_contract() -> None:
    x, y = sp.symbols("x y")
    equations = ((x**2 - 2) ** 2, y - x)

    quotient = QuotientAlgebra.from_polynomials(equations, (x, y))
    representation = compute_rational_univariate_representation(equations, (x, y))
    border = compute_border_basis(equations, (x, y))

    assert quotient.dimension == representation.quotient_dimension == border.dimension == 4
    assert quotient.geometric_solution_count == representation.geometric_solution_count == 2


def test_quotient_from_lex_groebner_basis_respects_actual_basis_order() -> None:
    x, y, z = sp.symbols("x y z")
    basis = sp.groebner((x**2 - 2, y**2 - 3, z - x * y), x, y, z, order="lex")

    quotient = QuotientAlgebra.from_groebner_basis(basis)

    assert quotient.dimension == 4
    assert quotient.normal_form(z - x * y) == 0


def test_separator_matrix_direct_path_uses_one_reduction_per_basis_monomial(monkeypatch) -> None:
    x, y, z = sp.symbols("x y z")
    quotient = QuotientAlgebra.from_polynomials((x**2 - 2, y**2 - 3, z**2 - 5), (x, y, z))
    original = QuotientAlgebra.coordinate_vector
    calls = {"count": 0}

    def counted(self, expression):
        calls["count"] += 1
        return original(self, expression)

    monkeypatch.setattr(QuotientAlgebra, "coordinate_vector", counted)
    columns = quotient.separator_matrix_columns((1, 2, 3))

    assert len(columns) == quotient.dimension
    assert calls["count"] == quotient.dimension
