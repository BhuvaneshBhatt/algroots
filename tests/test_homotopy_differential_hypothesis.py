"""Generated differential tests for regular total-degree homotopies."""

import pytest
import sympy as sp

from algroots import polysolve

hypothesis = pytest.importorskip("hypothesis")
st = pytest.importorskip("hypothesis.strategies")
assume = hypothesis.assume
given = hypothesis.given
settings = hypothesis.settings


x, y = sp.symbols("x y")


def _numeric_root_set(result):
    return sorted(
        (tuple(complex(sp.N(value, 20)) for value in root) for root in result.roots),
        key=lambda root: tuple((round(v.real, 12), round(v.imag, 12)) for v in root),
    )


@settings(max_examples=8, deadline=None)
@given(
    a=st.integers(-2, 2),
    b=st.integers(-2, 2),
    x_shift=st.integers(-2, 2),
    y_shift=st.integers(-2, 2),
    u_radius=st.integers(1, 3),
    v_radius=st.integers(1, 3),
)
def test_generated_affine_quadratic_systems_agree_across_backends(
    a, b, x_shift, y_shift, u_radius, v_radius
):
    # u = x + a*y + c, v = b*x + y + d is invertible iff 1-a*b != 0.
    assume(1 - a * b != 0)
    u = x + a * y + x_shift
    v = b * x + y + y_shift
    equations = (
        sp.expand(u**2 - u_radius**2),
        sp.expand(v**2 - v_radius**2),
    )
    action = polysolve(equations, (x, y), method="action", digits=28, recognize=False)
    rur = polysolve(equations, (x, y), method="rur", digits=28, recognize=False)
    homotopy = polysolve(
        equations,
        (x, y),
        method="homotopy",
        digits=28,
        recognize=False,
        homotopy_gamma_attempts=2,
        max_homotopy_paths=8,
    )
    expected = _numeric_root_set(action)
    assert len(expected) == 4
    for result in (rur, homotopy):
        actual = _numeric_root_set(result)
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            assert (
                max(abs(a_value - b_value) for a_value, b_value in zip(left, right, strict=True))
                < 1e-10
            )
