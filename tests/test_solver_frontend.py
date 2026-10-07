"""Exactness, reuse and structural performance guards for the new portfolio."""

import pytest
import sympy as sp

from algroots import polysolve
from algroots.presolve import affine_presolve
from algroots.quotient import QuotientAlgebra
from algroots.rational_univariate.construction import rur_from_quotient

x, y, z = sp.symbols("x y z")


@pytest.mark.parametrize("method", ["auto", "action", "rur", "shape", "triangular"])
def test_affine_reconstruction_across_backends(method):
    equations = (x - y - z, y - 2 * z, z**3 - 2)
    result = polysolve(equations, (x, y, z), method=method, digits=30, recognize=False)
    assert result.solver_variables == (z,)
    assert len(result.roots) == 3
    for root in result.roots:
        assignment = dict(zip((x, y, z), root, strict=True))
        assert max(abs(complex(e.subs(assignment))) for e in equations) < 1e-20
    assert result.total_multiplicity == 3
    assert result.is_radical
    assert result.completeness.status == "conditional"
    if method == "rur":
        representation = result.rational_univariate_representation
        assert representation.variables == (x, y, z)
        coords = dict(
            zip((x, y, z), representation.normalized_coordinate_polynomials(), strict=True)
        )
        for equation in equations:
            assert (
                sp.Poly(
                    equation.subs({v: p.as_expr() for v, p in coords.items()}),
                    representation.parameter,
                )
                .rem(representation.defining_polynomial)
                .is_zero
            )


@pytest.mark.parametrize(
    "equations,dimension,count",
    [
        ((x**2, y**2), 4, 1),
        ((x**3, y - 2 * x), 3, 1),
        (((x - 1) ** 2 * (x + 2), y - x), 3, 2),
        ((x - 1, y - 2), 1, 1),
        ((x - 1, x - 2, y), 0, 0),
    ],
)
def test_multiplicity_and_completeness(equations, dimension, count):
    result = polysolve(equations, (x, y), digits=25, recognize=False)
    assert result.total_multiplicity == dimension
    assert result.geometric_solution_count == count
    assert result.is_radical == (dimension == count)
    assert result.has_multiple_roots == (dimension > count)
    assert result.completeness.expected_count == count


def test_presolve_algebraic_coefficients_and_new_linear_rows():
    reduced = affine_presolve((x - y, x * y - y**2 + z, y - sp.sqrt(2)), (x, y, z))
    assert not reduced.variables
    assert reduced.reconstruct((), (x, y, z), 30)[-1] == 0


def test_presolve_never_divides_by_variable_coefficients():
    reduced = affine_presolve((x * y, y**2 - y), (x, y))
    assert not reduced.substitutions
    result = polysolve((x * y, y**2 - y, x**2 - x), (x, y), recognize=False)
    assert len(result.roots) == 3


@pytest.mark.parametrize(
    "method,conversions", [("auto", 0), ("action", 0), ("rur", 0), ("shape", 1), ("triangular", 1)]
)
def test_one_groebner_and_lazy_fglm(monkeypatch, method, conversions):
    calls, fglm_calls = [], []
    original = sp.groebner
    basis_type = type(original((x**2 - 2,), x))
    original_fglm = basis_type.fglm

    def groebner(*args, **kwargs):
        calls.append(kwargs.get("order"))
        return original(*args, **kwargs)

    def fglm(self, *args, **kwargs):
        fglm_calls.append(kwargs)
        return original_fglm(self, *args, **kwargs)

    monkeypatch.setattr(sp, "groebner", groebner)
    monkeypatch.setattr(basis_type, "fglm", fglm)
    polysolve((x**2 - 2,), (x,), method=method, recognize=False)
    assert calls == ["grevlex"]
    assert len(fglm_calls) == conversions


def test_rur_consumes_already_constructed_quotient(monkeypatch):
    quotient = QuotientAlgebra.from_polynomials((x**2, y**2), (x, y))
    monkeypatch.setattr(sp, "groebner", lambda *a, **k: pytest.fail("recomputed Groebner basis"))
    result = rur_from_quotient(quotient)
    assert result.dimension == 4
    assert result.solution_count == 1


def test_auto_rur_on_action_budget_without_fglm(monkeypatch):
    basis_type = type(sp.groebner((x**2 - 1,), x))
    monkeypatch.setattr(basis_type, "fglm", lambda *a, **k: pytest.fail("unnecessary FGLM"))
    result = polysolve((x**2 - 1, y**2 - 4), (x, y), max_action_dimension=3, recognize=False)
    assert result.method == "rational_univariate"
    assert len(result.roots) == 4


@pytest.mark.parametrize("ordering", ["input", "auto"])
def test_ordering_preserves_public_coordinates(ordering):
    equations = (x - y**2, y**3 - 2, z - 3 * y)
    result = polysolve(equations, (z, y, x), variable_order=ordering, recognize=False, digits=25)
    assert result.variables == (z, y, x)
    for root in result.roots:
        assert abs(complex(root[0] - 3 * root[1])) < 1e-18
        assert abs(complex(root[2] - root[1] ** 2)) < 1e-18


def test_presolve_opt_out_retains_staircase_variable_order():
    result = polysolve((x - y, y**2 - 2), (x, y), presolve=False, recognize=False)
    assert result.solver_variables == (x, y)
    assert not result.affine_substitutions


def test_sparse_krylov_matches_characteristic_and_noncyclic_fallback(monkeypatch):
    quotient = QuotientAlgebra.from_polynomials((x**12 - 1,), (x,))
    matrix = quotient.variable_multiplication_matrices[0]
    assert isinstance(matrix, sp.SparseMatrix)
    t = sp.Symbol("t")
    separator = quotient.separating_element(t)
    assert separator.defining_polynomial == sp.Poly(t**12 - 1, t, domain="QQ")
    nilpotent = QuotientAlgebra.from_polynomials((x**4, y**3), (x, y))
    separator = nilpotent.separating_element(t)
    assert separator.geometric_solution_count == 1
    assert separator.defining_polynomial == sp.Poly(t, t, domain="QQ")
    assert separator.coordinate_denominator.degree() == 0


@pytest.mark.recognition
def test_recognition_is_attempted_only_after_reconstruction():
    result = polysolve((x - 2 * y, y**2 - 2), (x, y), method="rur", digits=30)
    assert result.recognition_attempted
    assert result.recognized_roots is not None
    assert len(result.recognized_roots) == 2


def test_many_affine_variables_reduce_before_groebner(monkeypatch):
    variables = sp.symbols("u:25")
    equations = tuple(variables[i] - (i + 1) * variables[-1] for i in range(24)) + (
        variables[-1] ** 2 - 2,
    )
    original = sp.groebner
    sizes = []

    def record(equations, *gens, **kwargs):
        sizes.append(len(gens))
        return original(equations, *gens, **kwargs)

    monkeypatch.setattr(sp, "groebner", record)
    result = polysolve(equations, variables, recognize=False, digits=25)
    assert sizes == [1]
    assert len(result.roots) == 2
    assert result.total_multiplicity == 2


def test_auto_ordering_changes_internal_order_without_changing_roots():
    equations = (x**2 - 2, y**3 - x)
    result = polysolve(
        equations, (x, y), variable_order="auto", recognize=False, digits=25, presolve=False
    )
    assert result.solver_variables == (y, x)
    assert result.variables == (x, y)
    assert len(result.roots) == 6
    for root in result.roots:
        assert abs(complex(root[1] ** 3 - root[0])) < 1e-18


def test_nonreduced_dimension_does_not_consume_distinct_solution_budget():
    result = polysolve((x**5, y**5), (x, y), max_solutions=2, recognize=False, digits=25)
    assert result.method == "rational_univariate"
    assert result.total_multiplicity == 25
    assert len(result.roots) == 1


def test_sparse_cyclic_krylov_avoids_characteristic_polynomial(monkeypatch):
    quotient = QuotientAlgebra.from_polynomials((x**12 - 1,), (x,))
    matrix = quotient.linear_combination_matrix((1,))
    monkeypatch.setattr(
        type(matrix), "charpoly", lambda *a, **k: pytest.fail("dense characteristic path")
    )
    separator = quotient.separating_element(sp.Symbol("t"))
    assert separator.geometric_solution_count == 12


def test_noncyclic_krylov_retains_multiplicities(monkeypatch):
    from algroots.rational_univariate import linear_algebra

    original = linear_algebra.packed_krylov_minimal_polynomial
    closures = []

    def record(matrix, parameter):
        try:
            return original(matrix, parameter)
        except ValueError:
            closures.append(matrix.rows)
            raise

    monkeypatch.setattr(linear_algebra, "packed_krylov_minimal_polynomial", record)
    quotient = QuotientAlgebra.from_polynomials((x**4, y**4), (x, y))
    separator = quotient.separating_element(sp.Symbol("t"))
    assert closures == [16]
    assert separator.geometric_solution_count == 1
    assert separator.coordinate_denominator.as_expr() == 16


@pytest.mark.parametrize("kwargs", [{"recognize": 1}, {"presolve": 1}, {"variable_order": "lex"}])
def test_frontend_does_not_bypass_option_validation(kwargs):
    from algroots import PolynomialSystemInputError

    with pytest.raises((PolynomialSystemInputError, ValueError, TypeError)):
        polysolve((x - 1,), (x,), **kwargs)


def test_algebraic_result_evidence_respects_cover_projection():
    from algroots import algsolve

    polynomial = algsolve((x**3,), (x,), recognize=False)
    assert polynomial.total_multiplicity == 3
    assert polynomial.geometric_solution_count == 1
    assert not polynomial.is_radical
    branched = algsolve((sp.sqrt(x) - 1,), (x,), recognize=False)
    assert branched.completeness.status == "numerical"
    assert branched.completeness.expected_count is None
    assert branched.total_multiplicity is None
    assert branched.is_radical is None
