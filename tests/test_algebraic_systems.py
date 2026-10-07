import pytest
import sympy as sp

from algroots import AlgebraicSystemRoots, PolynomialSystemInputError, algsolve
from algroots.algebraization import algebraize_system


def _near(value, expected, digits=30):
    return abs(sp.N(value - expected, digits)) < sp.Float(10, digits) ** (-20)


def test_algebraize_square_root_introduces_auxiliary():
    x = sp.symbols("x")
    system = algebraize_system((sp.sqrt(x) - x + 2,), (x,))
    assert len(system.auxiliary_variables) == 1
    aux = system.auxiliary_variables[0]
    assert any(sp.expand(eq - (aux**2 - x)) == 0 for eq in system.polynomial_equations)


def test_square_root_branch_filters_spurious_squared_root():
    x = sp.symbols("x")
    result = algsolve((sp.sqrt(x) - (x - 2),), (x,), digits=50)
    assert isinstance(result, AlgebraicSystemRoots)
    assert len(result.roots) == 1
    assert _near(result.roots[0][0], 4)
    assert sp.N(sp.sqrt(result.roots[0][0]) - (result.roots[0][0] - 2), 30) == 0


def test_rational_equation():
    x = sp.symbols("x")
    result = algsolve(((x + 1) / (x - 2) - 3,), (x,), digits=45)
    assert len(result.roots) == 1
    assert _near(result.roots[0][0], sp.Rational(7, 2))
    assert result.nonzero_constraints


def test_polynomialized_pole_is_rejected():
    x = sp.symbols("x")
    quotient = sp.Mul(
        x**2 - 1,
        sp.Pow(x - 1, -1, evaluate=False),
        evaluate=False,
    )
    equation = sp.Add(quotient, -2, evaluate=False)
    result = algsolve((equation,), (x,), digits=45)
    assert result.roots == ()


def test_rational_power_uses_principal_branch():
    x = sp.symbols("x")
    expression = sp.Pow(x, sp.Rational(2, 3), evaluate=False) - 4
    result = algsolve((expression,), (x,), digits=50)
    assert len(result.roots) == 1
    assert _near(result.roots[0][0], 8)


def test_nested_radical():
    x = sp.symbols("x")
    expression = sp.sqrt(x + sp.sqrt(x)) - 2
    result = algsolve((expression,), (x,), digits=55)
    expected_t = (-1 + sp.sqrt(17)) / 2
    expected_x = sp.expand(expected_t**2)
    assert len(result.roots) == 1
    assert _near(result.roots[0][0], expected_x)


def test_multivariate_radical_system():
    x, y = sp.symbols("x y")
    result = algsolve(
        (sp.sqrt(x) - y, y**2 - 2),
        (x, y),
        digits=50,
    )
    assert len(result.roots) == 1
    root = result.roots[0]
    # Principal sqrt(2) is the positive real square root, so the negative branch is rejected.
    assert _near(root[0], 2)
    assert _near(root[1], sp.sqrt(2))


def test_exact_algebraic_coefficients_remain_coefficients():
    x = sp.symbols("x")
    system = algebraize_system((sp.sqrt(2) * x - 1,), (x,))
    assert system.auxiliary_variables == ()
    result = algsolve((sp.sqrt(2) * x - 1,), (x,), digits=45)
    assert len(result.roots) == 1
    assert _near(result.roots[0][0], 1 / sp.sqrt(2))


def test_negative_rational_power_forces_nonzero_base():
    x = sp.symbols("x")
    expression = sp.Pow(x, sp.Rational(-1, 2), evaluate=False) - 1
    result = algsolve((expression,), (x,), digits=45)
    assert len(result.roots) == 1
    assert _near(result.roots[0][0], 1)


def test_transcendental_function_rejected():
    x = sp.symbols("x")
    with pytest.raises(PolynomialSystemInputError):
        algsolve((sp.sin(x),), (x,))


def test_undeclared_symbol_rejected():
    x, y = sp.symbols("x y")
    with pytest.raises(PolynomialSystemInputError):
        algebraize_system((x + y,), (x,))


def test_auxiliary_variable_limit():
    x = sp.symbols("x")
    expression = sp.sqrt(x) + sp.sqrt(x + 1)
    with pytest.raises(PolynomialSystemInputError):
        algebraize_system((expression,), (x,), max_auxiliary_variables=1)


def test_denominator_saturation_removes_pole_component():
    x, y, z = sp.symbols("x y z")

    def raw_div(numerator):
        return sp.Mul(
            numerator,
            sp.Pow(x, -1, evaluate=False),
            evaluate=False,
        )

    equations = (
        sp.Add(raw_div(x * y), -1, evaluate=False),
        sp.Add(raw_div(x * z), -2, evaluate=False),
        raw_div(x * (x - 2)),
    )
    result = algsolve(equations, (x, y, z), digits=45)

    assert len(result.roots) == 1
    root = result.roots[0]
    assert _near(root[0], 2)
    assert _near(root[1], 1)
    assert _near(root[2], 2)
    assert result.nonzero_constraints


def test_projected_deduplication_keeps_diagnostics_and_residual_aligned():
    from algroots.algebraic import _filter_algebraic_roots
    from algroots.algebraization import AlgebraizedSystem
    from algroots.solver import PolynomialSystemRoots, RootDiagnostics

    x = sp.Symbol("x")
    auxiliary = sp.Symbol("_aux")
    digits = 40
    first = sp.Float(1, 80) + sp.Float("1e-50", 80)
    second = sp.Float(1, 80) + sp.Float("2e-50", 80)
    diag_first = RootDiagnostics(sp.Integer(1), sp.Integer(11), False, False, False)
    diag_second = RootDiagnostics(sp.Integer(2), sp.Integer(22), False, False, False)
    polynomial_result = PolynomialSystemRoots(
        roots=((first, sp.Integer(0)), (second, sp.Integer(1))),
        variables=(x, auxiliary),
        equations=(x - 1,),
        groebner_basis=(),
        precision_digits=digits,
        working_digits=80,
        verification_digits=20,
        max_relative_residual=sp.Integer(0),
        method="test",
        diagnostics=(diag_first, diag_second),
    )
    algebraized = AlgebraizedSystem(
        original_equations=(x - 1,),
        polynomial_equations=(x - 1,),
        original_variables=(x,),
        augmented_variables=(x, auxiliary),
        auxiliary_variables=(auxiliary,),
        nonzero_constraints=(),
    )

    roots, diagnostics, max_residual = _filter_algebraic_roots(polynomial_result, algebraized)

    assert len(roots) == 1
    assert diagnostics == (diag_first,)
    assert max_residual < sp.Float("1.5e-50", 80)
