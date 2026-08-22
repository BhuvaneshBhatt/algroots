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
def constructed_systems(draw):
    a, b = draw(
        st.sampled_from(
            tuple((left, right) for left in range(1, 6) for right in range(1, 6) if left != right)
        )
    )
    shear = draw(st.integers(min_value=-2, max_value=2))
    # u=x+shear*y, v=y is invertible over Q.  The system has exactly four simple roots.
    u = x + shear * y
    v = y
    return (u**2 - a, v**2 - b), (x, y), a, b, shear


@given(constructed_systems())
@settings(max_examples=12, deadline=None)
def test_constructed_zero_dimensional_systems_are_complete(data):
    equations, variables, a, b, shear = data
    result = polysolve(
        equations, variables, method="auto", digits=28, verification_digits=16, recognize=False
    )
    assert len(result.roots) == 4
    assert result.max_relative_residual < sp.Float("1e-16")

    expected = []
    for u_sign, v_sign in itertools.product((-1, 1), repeat=2):
        v_value = v_sign * sp.sqrt(b)
        u_value = u_sign * sp.sqrt(a)
        expected.append((u_value - shear * v_value, v_value))

    for root in result.roots:
        assert any(
            max(
                abs(complex(sp.N(got - want, 18)))
                for got, want in zip(root, candidate, strict=True)
            )
            < 1e-12
            for candidate in expected
        )


@given(
    st.sampled_from((-3, -2, -1, 1, 2, 3)),
    st.sampled_from((-3, -2, -1, 1, 2, 3)),
)
@settings(max_examples=10, deadline=None)
def test_invertible_equation_recombination_preserves_roots(a, b):
    original = (x**2 - 2, y**2 - 3)
    transformed = (a * original[0] + b * original[1], original[1])
    left = polysolve(original, (x, y), digits=25, recognize=False)
    right = polysolve(transformed, (x, y), digits=25, recognize=False)
    assert len(left.roots) == len(right.roots) == 4
    for root in left.roots:
        assert any(
            max(abs(complex(sp.N(got - want, 16))) for got, want in zip(root, other, strict=True))
            < 1e-10
            for other in right.roots
        )
