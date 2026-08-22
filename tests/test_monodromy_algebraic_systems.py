"""High-level monodromy support for algebraic equation systems."""

import pytest
import sympy as sp

from algroots import (
    PathTrackerOptions,
    PolynomialSystemInputError,
    discover_monodromy_orbit,
)
from algroots.monodromy import normalized_root_distance

x, y = sp.symbols("x y")
OPTIONS = PathTrackerOptions(
    initial_step=0.03,
    max_step=0.07,
    initial_digits=45,
    residual_digits=25,
)


def _close(value, expected, digits=30):
    del digits
    return normalized_root_distance((value,), (expected,)) < 1e-15


def test_rational_system_discovers_projected_orbit():
    equation = (x**2 - 1) / (x - 3)
    result = discover_monodromy_orbit(
        (equation,),
        (x,),
        ((1,),),
        max_loops=4,
        radius=2.0,
        random_seed=0,
        options=OPTIONS,
        recognize=False,
    )

    assert len(result.roots) == 2
    assert any(_close(root[0], 1) for root in result.roots)
    assert any(_close(root[0], -1) for root in result.roots)
    assert result.variables == (x,)
    assert result.equations == (sp.together(equation),)
    assert len(result.auxiliary_variables) == 1
    assert len(result.tracking_variables) == 2
    assert len(result.tracking_equations) == 2
    assert result.nonzero_constraints
    assert len(result.root_info) == len(result.roots)
    assert all(info.verified for info in result.root_info)


def test_principal_square_root_branch_filters_polynomial_cover():
    equation = sp.sqrt(x) - (x - 2)
    result = discover_monodromy_orbit(
        (equation,),
        (x,),
        ((4,),),
        max_loops=4,
        radius=2.0,
        random_seed=0,
        options=OPTIONS,
        recognize=False,
    )

    assert len(result.roots) == 1
    assert _close(result.roots[0][0], 4)
    assert _close(sp.sqrt(result.roots[0][0]) - (result.roots[0][0] - 2), 0)
    assert result.auxiliary_variables
    # The polynomial cover also contains x=1 on the wrong sqrt sheet; it must
    # never appear in the public root set even if monodromy reaches that sheet.
    assert not any(_close(root[0], 1) for root in result.roots)


def test_rational_power_seed_is_lifted_on_principal_branch():
    equation = sp.Pow(x, sp.Rational(2, 3), evaluate=False) - 4
    result = discover_monodromy_orbit(
        (equation,),
        (x,),
        ((8,),),
        expected_root_count=1,
        max_loops=2,
        options=OPTIONS,
        recognize=False,
    )

    assert result.loops_completed == 0
    assert result.completeness_basis == "exact_root_count"
    assert len(result.roots) == 1
    assert _close(result.roots[0][0], 8)
    assert result.auxiliary_variables


def test_nested_radical_seed_lifting_matches_algebraic_semantics():
    equation = sp.sqrt(x + sp.sqrt(x)) - 2
    t = (-1 + sp.sqrt(17)) / 2
    expected = sp.expand(t**2)
    result = discover_monodromy_orbit(
        (equation,),
        (x,),
        ((sp.N(expected, 50),),),
        expected_root_count=1,
        max_loops=1,
        options=OPTIONS,
        recognize=False,
    )

    assert len(result.auxiliary_variables) == 2
    assert len(result.tracking_variables) == 3
    assert len(result.tracking_equations) == 3
    assert _close(result.roots[0][0], expected)


def test_multivariate_radical_system_projects_to_original_variables():
    equations = (sp.sqrt(x) - y, y**2 - 2)
    result = discover_monodromy_orbit(
        equations,
        (x, y),
        ((2, sp.sqrt(2)),),
        expected_root_count=1,
        max_loops=1,
        options=OPTIONS,
        recognize=False,
    )

    assert result.variables == (x, y)
    assert len(result.roots) == 1
    assert len(result.roots[0]) == 2
    assert _close(result.roots[0][0], 2)
    assert _close(result.roots[0][1], sp.sqrt(2))
    assert len(result.tracking_variables) == 3


def test_exact_algebraic_coefficients_need_no_auxiliary_variables():
    equations = (sp.sqrt(2) * (x**2 - 1),)
    result = discover_monodromy_orbit(
        equations,
        (x,),
        ((1,), (-1,)),
        expected_root_count=2,
        max_loops=1,
        options=OPTIONS,
        recognize=False,
    )

    assert result.auxiliary_variables == ()
    assert result.tracking_variables == (x,)
    assert len(result.roots) == 2


def test_algebraic_monodromy_respects_auxiliary_limit():
    equation = sp.sqrt(x) + sp.sqrt(x + 1)
    with pytest.raises(PolynomialSystemInputError, match="max_auxiliary_variables"):
        discover_monodromy_orbit(
            (equation,),
            (x,),
            ((1,),),
            max_auxiliary_variables=1,
            max_loops=1,
            recognize=False,
        )


def test_transcendental_system_is_still_rejected():
    with pytest.raises(PolynomialSystemInputError, match="unsupported non-algebraic"):
        discover_monodromy_orbit(
            (sp.sin(x),),
            (x,),
            ((0,),),
            max_loops=1,
            recognize=False,
        )


def test_complex_principal_square_root_seed_lifts_correctly():
    equation = sp.sqrt(x) - sp.I
    result = discover_monodromy_orbit(
        (equation,),
        (x,),
        ((-1,),),
        expected_root_count=1,
        max_loops=1,
        options=OPTIONS,
        recognize=False,
    )

    assert len(result.roots) == 1
    assert _close(result.roots[0][0], -1)
    assert result.root_info[0].verified
