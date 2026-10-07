import sympy as sp

import algroots.solver as solver
from algroots.quotient import QuotientAlgebra

x, y, z = sp.symbols("x y z")


def _quotient(equations, variables, max_dimension):
    normalized = solver._normalize_equations(equations, variables)
    basis = solver._groebner_basis(normalized, variables)
    return QuotientAlgebra.from_groebner_basis(basis, variables, max_dimension=max_dimension)


def test_separator_construction_uses_one_reduction_per_staircase_monomial(monkeypatch):
    variables = (x, y, z)
    quotient = _quotient((x**2 - 2, y**2 - 3, z**2 - 5), variables, 64)

    original = QuotientAlgebra.coordinate_vector
    calls = {"count": 0}

    def counted(self, expression):
        calls["count"] += 1
        return original(self, expression)

    monkeypatch.setattr(QuotientAlgebra, "coordinate_vector", counted)
    quotient.separator_matrix_columns((1, 2, 3))
    assert calls["count"] == quotient.dimension


def test_coordinate_normal_forms_need_at_most_one_reduction_per_variable(monkeypatch):
    variables = (x, y, z)
    quotient = _quotient(
        ((x + y) ** 2 - 2, (y + z) ** 2 - 3, z**2 - 5),
        variables,
        128,
    )

    original = QuotientAlgebra.coordinate_vector
    calls = {"count": 0}

    def counted(self, expression):
        calls["count"] += 1
        return original(self, expression)

    monkeypatch.setattr(QuotientAlgebra, "coordinate_vector", counted)
    _ = quotient.coordinate_normal_forms
    assert calls["count"] <= len(variables)


def test_separator_candidate_count_is_bounded_by_variable_count():
    for count in range(1, 10):
        candidates = solver._separator_candidates(count)
        assert len(candidates) <= count + 4


def test_separator_reduction_count_scales_with_quotient_dimension(monkeypatch):
    for variables in ((x, y), (x, y, z)):
        equations = tuple(variable**2 - (index + 2) for index, variable in enumerate(variables))
        quotient = _quotient(equations, variables, 128)

        original = QuotientAlgebra.coordinate_vector
        calls = {"count": 0}

        def counted(self, expression, _calls=calls, _original=original):
            _calls["count"] += 1
            return _original(self, expression)

        monkeypatch.setattr(QuotientAlgebra, "coordinate_vector", counted)
        quotient.separator_matrix_columns(tuple(range(1, len(variables) + 1)))
        assert calls["count"] == quotient.dimension
        monkeypatch.setattr(QuotientAlgebra, "coordinate_vector", original)
