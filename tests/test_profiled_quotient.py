import pytest
import sympy as sp
from sympy.polys.matrices import DomainMatrix

from algroots import polysolve
from algroots.quotient import QuotientAlgebra

x, y = sp.symbols("x y")


@pytest.mark.parametrize(
    "equations,variables",
    [
        ((x**5 - 2, y**5 - 3), (x, y)),
        ((x**6, y**4), (x, y)),
        ((x**4 - sp.sqrt(2), y**3 - 3), (x, y)),
        ((x**12 - x - 1,), (x,)),
    ],
)
def test_profiled_trace_and_rank_match_dense_exact_oracle(equations, variables):
    q = QuotientAlgebra.from_polynomials(equations, variables)
    actions = [q._monomial_matrix(e) for e in q.standard_exponents]
    expected = sp.Matrix([[sp.simplify((a * b).trace()) for b in actions] for a in actions])
    assert q.trace_pairing.applyfunc(sp.simplify) == expected
    assert q.geometric_solution_count == expected.rank()
    info = q.operation_diagnostics
    assert info["geometric_rank_seconds"] >= 0
    assert info["rank_backend"] == ("rational_domain" if q.domain == sp.QQ else "expression")
    assert info["basis_stream_peak_entries"] <= 8
    assert "basis_multiplication_matrices" not in q.__dict__
    if len(variables) > 1:
        assert info["basis_stream_cache_hits"] > 0


def test_rational_rank_avoids_expression_elimination(monkeypatch):
    q = QuotientAlgebra.from_polynomials(((x - y) ** 2, y**2 - 1), (x, y))
    _ = q.trace_pairing
    monkeypatch.setattr(sp.MatrixBase, "rank", lambda *a, **k: pytest.fail("expression rank"))
    assert q.geometric_solution_count == 2


def test_algebraic_rank_keeps_profiled_expression_path(monkeypatch):
    q = QuotientAlgebra.from_polynomials((x**2 - sp.sqrt(2),), (x,))
    _ = q.trace_pairing
    monkeypatch.setattr(
        DomainMatrix, "rank", lambda *a, **k: pytest.fail("costly algebraic domain rank")
    )
    assert q.geometric_solution_count == 2


def test_standard_coordinate_vectors_skip_reduction(monkeypatch):
    q = QuotientAlgebra.from_polynomials((x**4 - 2, y**3 - 3), (x, y))
    monkeypatch.setattr(
        QuotientAlgebra, "normal_form", lambda *args: pytest.fail("basis reduction")
    )
    for index, monomial in enumerate(q.standard_monomials):
        assert q.coordinate_vector(monomial) == sp.eye(q.dimension).col(index)
    assert q.operation_diagnostics["coordinate_basis_hits"] == q.dimension


def test_stream_cache_is_bounded_beyond_eight_basis_actions():
    q = QuotientAlgebra.from_polynomials((x**8, y**8), (x, y))
    assert q.geometric_solution_count == 1
    assert q.operation_diagnostics["basis_stream_peak_entries"] == 8
    snapshot = q.operation_diagnostics
    snapshot["basis_stream_peak_entries"] = 1000
    assert q.operation_diagnostics["basis_stream_peak_entries"] == 8


def test_solver_exposes_quotient_operation_observations():
    result = polysolve((x**3, y**2), (x, y), presolve=False, recognize=False)
    stats = dict(result.cost_diagnostics.quotient_operation_statistics)
    assert stats["rank_backend"] == "rational_domain"
    assert stats["geometric_rank_seconds"] >= 0
    assert result.geometric_solution_count == 1 and not result.is_radical
