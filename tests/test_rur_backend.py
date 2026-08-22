import pytest
import sympy as sp

from algroots import RationalUnivariateError, polysolve


def _sorted_numeric(roots, digits=25):
    return sorted(
        tuple((float(sp.re(sp.N(v, digits))), float(sp.im(sp.N(v, digits)))) for v in root)
        for root in roots
    )


def test_rur_backend_returns_all_complex_roots_and_representation():
    x, y = sp.symbols("x y")
    equations = (x**2 - 2, y - x)
    result = polysolve(equations, (x, y), method="rur", digits=35, recognize=False)
    assert result.method == "rational_univariate"
    assert len(result.roots) == 2
    assert result.rational_univariate_representation is not None
    assert result.rational_univariate_representation.solution_count == 2
    for root in result.roots:
        assert abs(complex(sp.N(root[0] ** 2 - 2, 25))) < 1e-20
        assert abs(complex(sp.N(root[1] - root[0], 25))) < 1e-20


def test_rur_backend_agrees_with_action_backend():
    x, y = sp.symbols("x y")
    equations = (x**2 - 1, y**2 - 4)
    rur = polysolve(equations, (x, y), method="rur", digits=30, recognize=False)
    action = polysolve(equations, (x, y), method="action", digits=30, recognize=False)
    assert _sorted_numeric(rur.roots) == _sorted_numeric(action.roots)


def test_rur_backend_unit_ideal_is_empty():
    x = sp.symbols("x")
    result = polysolve((sp.Integer(1),), (x,), method="rur", recognize=False)
    assert result.roots == ()
    assert result.rational_univariate_representation is not None
    assert result.rational_univariate_representation.solution_count == 0


def test_rur_backend_supports_exact_algebraic_coefficients():
    x = sp.symbols("x")
    result = polysolve((x**2 - sp.sqrt(2),), (x,), method="rur", digits=35, recognize=False)
    assert len(result.roots) == 2
    rep = result.rational_univariate_representation
    assert rep is not None
    assert rep.defining_polynomial.domain.is_AlgebraicField
    for root in result.roots:
        assert abs(complex(sp.N(root[0] ** 2 - sp.sqrt(2), 30))) < 1e-25


def test_rur_constructor_rejects_inexact_coefficients():
    from algroots import compute_rational_univariate_representation

    x = sp.symbols("x")
    with pytest.raises(RationalUnivariateError, match="exact"):
        compute_rational_univariate_representation((x**2 - sp.Float("1.2"),), (x,))


def test_rur_constructor_rejects_transcendental_coefficients():
    from algroots import compute_rational_univariate_representation

    x = sp.symbols("x")
    with pytest.raises(RationalUnivariateError, match="algebraic"):
        compute_rational_univariate_representation((x - sp.pi,), (x,))


def test_invalid_method_message_includes_rur():
    x = sp.symbols("x")
    with pytest.raises(ValueError, match="rur"):
        polysolve((x - 1,), (x,), method="bad", recognize=False)  # type: ignore[arg-type]


def test_numerical_rur_backend_does_not_require_exact_root_enumeration(monkeypatch):
    x = sp.symbols("x")

    def fail_exact_roots(self, *args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("numerical RUR backend called exact all_roots()")

    monkeypatch.setattr(sp.Poly, "all_roots", fail_exact_roots)
    result = polysolve(
        (x**5 - sp.sqrt(2),),
        (x,),
        method="rur",
        digits=35,
        recognize=False,
    )
    assert len(result.roots) == 5
    assert max(abs(complex(sp.N(root[0] ** 5 - sp.sqrt(2), 25))) for root in result.roots) < 1e-20


def test_rur_normalized_coordinate_polynomials_are_cached(monkeypatch):
    from algroots import compute_rational_univariate_representation

    x, y = sp.symbols("x y")
    representation = compute_rational_univariate_representation((x**2 - 2, y - x), (x, y))
    calls = 0
    original = sp.invert

    def counting_invert(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(sp, "invert", counting_invert)
    first = representation.normalized_coordinate_polynomials()
    second = representation.normalized_coordinate_polynomials()
    assert first == second
    assert calls == 1


@pytest.mark.parametrize(
    ("equations", "variables", "expected_count"),
    [
        (
            (sp.Symbol("x") ** 2 - sp.sqrt(2), sp.Symbol("y") - sp.Symbol("x")),
            (sp.Symbol("x"), sp.Symbol("y")),
            2,
        ),
        (
            (sp.Symbol("x") - sp.sqrt(2), sp.Symbol("y") - sp.sqrt(3)),
            (sp.Symbol("x"), sp.Symbol("y")),
            1,
        ),
        (
            (sp.Symbol("x") ** 2 - sp.sqrt(2) - sp.sqrt(3), sp.Symbol("y") - sp.Symbol("x")),
            (sp.Symbol("x"), sp.Symbol("y")),
            2,
        ),
    ],
)
def test_number_field_rur_agrees_with_action(equations, variables, expected_count):
    rur = polysolve(equations, variables, method="rur", digits=32, recognize=False)
    action = polysolve(equations, variables, method="action", digits=32, recognize=False)
    assert len(rur.roots) == len(action.roots) == expected_count
    assert _sorted_numeric(rur.roots) == _sorted_numeric(action.roots)


def test_crootof_coefficient_rur_agrees_with_action():
    x, z = sp.symbols("x z")
    coefficient = sp.CRootOf(z**3 - z - 1, 0)
    equations = (x - coefficient,)
    rur = polysolve(equations, (x,), method="rur", digits=32, recognize=False)
    action = polysolve(equations, (x,), method="action", digits=32, recognize=False)
    assert _sorted_numeric(rur.roots) == _sorted_numeric(action.roots)
