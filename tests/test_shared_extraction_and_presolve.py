import pytest
import sympy as sp

from algroots import polysolve
from algroots.presolve import polynomial_presolve
from algroots.quotient import QuotientAlgebra
from algroots.rational_univariate import compute_rational_univariate_representation
from algroots.rational_univariate.solve import numerical_rur_roots

x, y, z = sp.symbols("x y z")


@pytest.mark.parametrize(
    "equations,variables",
    [
        ((x - y**2, y**2 - 2), (x, y)),
        ((x - y**2, y - z**2, z**2 - 2), (x, y, z)),
        ((2 * x - y**3, y**2 + 1), (x, y)),
        ((x - y**2, (y - 1) ** 3), (x, y)),
    ],
)
def test_polynomial_substitution_matches_unreduced(equations, variables):
    reduced = polysolve(equations, variables, recognize=False, method="rur")
    original = polysolve(equations, variables, recognize=False, presolve=False, method="rur")
    assert len(reduced.roots) == len(original.roots)
    assert reduced.total_multiplicity == original.total_multiplicity
    assert reduced.affine_substitutions
    for root in reduced.roots:
        assert (
            min(
                max(abs(complex(a - b)) for a, b in zip(root, other, strict=True))
                for other in original.roots
            )
            < 1e-30
        )
    rur = reduced.rational_univariate_representation
    assert rur.variables == variables
    for equation in equations:
        mapping = dict(
            zip(
                variables,
                (p.as_expr() for p in rur.normalized_coordinate_polynomials()),
                strict=True,
            )
        )
        assert (
            sp.Poly(sp.expand(equation.subs(mapping)), rur.parameter)
            .rem(rur.defining_polynomial)
            .is_zero
        )


def test_never_divide_variable_pivot():
    result = polynomial_presolve((x * y - 1, y**2 - 1), (x, y))
    assert result.variables == (x, y)
    assert not result.substitutions


def test_guard_prevents_expansion():
    result = polynomial_presolve(
        (x - (y + z) ** 8, x**20 + y, z**2 - 1), (x, y, z), max_terms=10, max_degree=16
    )
    assert not result.substitutions
    assert result.equations[1] == x**20 + y


@pytest.mark.parametrize(
    "option,value", [("max_terms", 0), ("max_degree", False), ("growth_factor", -1)]
)
def test_invalid_presolve_controls(option, value):
    with pytest.raises(ValueError):
        polynomial_presolve((x - y**2, y**2 - 1), (x, y), **{option: value})


def test_rur_uses_arb_not_sympy_nroots(monkeypatch):
    rur = compute_rational_univariate_representation((x**5 - x + 1,), (x,))
    monkeypatch.setattr(
        sp, "nroots", lambda *a, **k: pytest.fail("unexpected SymPy root extraction")
    )
    roots = numerical_rur_roots(rur, digits=60)
    assert len(roots) == 5
    assert max(abs(complex(sp.N(r[0] ** 5 - r[0] + 1, 30))) for r in roots) < 1e-50


def test_clustered_rur_roots():
    delta = sp.Rational(1, 10**25)
    rur = compute_rational_univariate_representation((x * (x - delta) * (x - 1),), (x,))
    roots = numerical_rur_roots(rur, digits=70, max_precision_digits=180)
    assert len(roots) == 3
    for target in (0, delta, 1):
        assert min(abs(complex(r[0] - target)) for r in roots) < 1e-60


@pytest.mark.parametrize(
    "equations,variables",
    [
        ((x**7 - 1,), (x,)),
        ((x**3, y**2), (x, y)),
        ((x**2 - 2, y**2 - 3), (x, y)),
        ((x**2 - sp.sqrt(2),), (x,)),
    ],
)
def test_streaming_trace_matches_dense_oracle(equations, variables):
    quotient = QuotientAlgebra.from_polynomials(equations, variables)
    matrices = [quotient._monomial_matrix(e) for e in quotient.standard_exponents]
    expected = sp.Matrix([[sp.simplify((a * b).trace()) for b in matrices] for a in matrices])
    assert quotient.trace_pairing.applyfunc(sp.simplify) == expected
    assert quotient.multiplication_matrix(sum(quotient.standard_monomials)) == sum(
        matrices, sp.zeros(quotient.dimension)
    )
    assert "basis_multiplication_matrices" not in quotient.__dict__


def test_large_trace_storage_guard(monkeypatch):
    quotient = QuotientAlgebra.from_polynomials((x**32 - 1,), (x,))
    monkeypatch.setattr(
        QuotientAlgebra,
        "basis_multiplication_matrices",
        property(lambda _: pytest.fail("cubic basis-action allocation")),
    )
    assert quotient.geometric_solution_count == 32
    assert quotient.trace_pairing.shape == (32, 32)


def test_shared_extractor_tries_final_precision_budget(monkeypatch):
    import flint
    from flint import acb, ctx

    from algroots.univariate import numerical_univariate_roots

    seen = []
    old_precision = ctx.dps

    class Backend:
        def __init__(self, coefficients):
            # x - 1 is translated exactly to x before isolation.
            assert coefficients[0] == 0 and coefficients[1] == 1

        def roots(self, **kwargs):
            seen.append(ctx.dps)
            assert kwargs["tol"] == "1e-22"
            if ctx.dps < 61:
                raise ValueError("needs final precision")
            return [acb(0)]

    monkeypatch.setattr(flint, "acb_poly", Backend)
    roots = numerical_univariate_roots(x - 1, x, 20, 100, 61)
    assert len(roots) == 1 and roots[0] - 1 == 0
    assert seen == [30, 60, 61]
    assert ctx.dps == old_precision


@pytest.mark.parametrize(
    "options", [dict(digits=True), dict(maxsteps=0), dict(max_precision_digits=19)]
)
def test_shared_extractor_invalid_budget(options):
    from algroots.univariate import numerical_univariate_roots

    controls = dict(digits=20, maxsteps=100)
    controls.update(options)
    with pytest.raises(ValueError):
        numerical_univariate_roots(x - 1, x, **controls)


def test_unit_ideal_streaming_traces():
    quotient = QuotientAlgebra.from_polynomials((1,), (x,))
    assert quotient.trace_vector.shape == (0, 0)
    assert quotient.trace_pairing.shape == (0, 0)
    assert quotient.multiplication_matrix(x).shape == (0, 0)


def test_high_degree_row_does_not_raise_other_row_limits():
    result = polynomial_presolve((x - y**10, x**2 + z, z**20 - 1), (x, y, z), max_degree=16)
    assert not result.substitutions
    assert result.variables == (x, y, z)
