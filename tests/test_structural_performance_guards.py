import sympy as sp

import algroots.solver as solver

x, y, z = sp.symbols("x y z")


def test_separator_construction_uses_one_reduction_per_staircase_monomial(monkeypatch):
    variables = (x, y, z)
    normalized = solver._normalize_equations((x**2 - 2, y**2 - 3, z**2 - 5), variables)
    basis = solver._groebner_basis(normalized, variables)
    monomials = solver._standard_monomials(basis, variables, 64)

    original = solver._normal_form_coeffs
    calls = {"count": 0}

    def counted(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(solver, "_normal_form_coeffs", counted)
    solver._separator_matrix_coeffs(basis, variables, monomials, (1, 2, 3))
    assert calls["count"] == len(monomials)


def test_coordinate_normal_forms_need_at_most_one_reduction_per_variable(monkeypatch):
    variables = (x, y, z)
    normalized = solver._normalize_equations(
        ((x + y) ** 2 - 2, (y + z) ** 2 - 3, z**2 - 5), variables
    )
    basis = solver._groebner_basis(normalized, variables)
    monomials = solver._standard_monomials(basis, variables, 128)

    original = solver._normal_form_coeffs
    calls = {"count": 0}

    def counted(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(solver, "_normal_form_coeffs", counted)
    solver._coordinate_normal_forms(basis, variables, monomials)
    assert calls["count"] <= len(variables)


def test_separator_candidate_count_is_bounded_by_variable_count():
    for count in range(1, 10):
        candidates = solver._separator_candidates(count)
        assert len(candidates) <= count + 4


def test_separator_reduction_count_scales_with_quotient_dimension(monkeypatch):
    # Independent quadratics have D=2**n. The optimized action constructor
    # should perform exactly D reductions for M_L, not n*D.
    for variables in ((x, y), (x, y, z)):
        equations = tuple(variable**2 - (index + 2) for index, variable in enumerate(variables))
        normalized = solver._normalize_equations(equations, variables)
        basis = solver._groebner_basis(normalized, variables)
        monomials = solver._standard_monomials(basis, variables, 128)

        original = solver._normal_form_coeffs
        calls = {"count": 0}

        def counted(*args, _calls=calls, _original=original, **kwargs):
            _calls["count"] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(solver, "_normal_form_coeffs", counted)
        solver._separator_matrix_coeffs(
            basis,
            variables,
            monomials,
            tuple(range(1, len(variables) + 1)),
        )
        assert calls["count"] == len(monomials)

        monkeypatch.setattr(solver, "_normal_form_coeffs", original)
