import itertools

import pytest
import sympy as sp

from algroots import polysolve

hypothesis = pytest.importorskip("hypothesis")
st = pytest.importorskip("hypothesis.strategies")
given = hypothesis.given
settings = hypothesis.settings


x, y = sp.symbols("x y")


@st.composite
def finite_point_systems(draw):
    x1, x2 = draw(
        st.sampled_from(
            tuple((left, right) for left in range(-4, 5) for right in range(-4, 5) if left != right)
        )
    )
    y1, y2 = draw(
        st.sampled_from(
            tuple((left, right) for left in range(-4, 5) for right in range(-4, 5) if left != right)
        )
    )
    a = draw(st.sampled_from((-3, -2, -1, 1, 2, 3)))
    b = draw(st.integers(-3, 3))
    fx = (x - x1) * (x - x2)
    fy = (y - y1) * (y - y2)
    mixed = (a * fx + b * fy, fy)
    expected = tuple(itertools.product((x1, x2), (y1, y2)))
    return mixed, expected


@given(finite_point_systems())
@settings(max_examples=18, deadline=None)
def test_generated_system_recovers_known_complete_point_set(data):
    equations, expected = data
    result = polysolve(equations, (x, y), digits=30, recognize=False)
    assert len(result.roots) == 4
    for point in expected:
        assert any(
            max(abs(complex(sp.N(got - want, 16))) for got, want in zip(root, point, strict=True))
            < 1e-10
            for root in result.roots
        )
