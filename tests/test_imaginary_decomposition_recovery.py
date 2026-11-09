"""Keep a primitive algebraic constant whole when embedding into K(x)."""

import pytest
import sympy as sp

from algroots.functional_decomposition import rational_decompose


@pytest.mark.parametrize("multivariate", [False, True])
def test_imaginary_constant_field_recovers_all_two_factor_classes(multivariate):
    x, t = sp.symbols("x t")
    K = sp.QQ.algebraic_field(sp.sqrt(-3))
    result = rational_decompose(
        x * x - 3 / x**2, (x,) if multivariate else x, parameter=t, constants=K
    )
    assert result.complete and result.verify()
    assert len(result.decompositions) == 3
    assert all(decomposition.verify() for decomposition in result.decompositions)
