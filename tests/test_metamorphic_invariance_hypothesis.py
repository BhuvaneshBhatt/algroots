"""Property tests for exact transformations that preserve a finite zero set."""

import itertools

import pytest
import sympy as sp

from algroots import polysolve

hypothesis = pytest.importorskip("hypothesis")
st = pytest.importorskip("hypothesis.strategies")
given = hypothesis.given
settings = hypothesis.settings


x, y = sp.symbols("x y")


def _complex_roots(roots):
    return [tuple(complex(sp.N(value, 24)) for value in root) for root in roots]


def _assert_same_root_set(left, right, *, tolerance=1e-9):
    unmatched = _complex_roots(right)
    assert len(left) == len(unmatched)
    for root in _complex_roots(left):
        distances = [
            max(abs(a - b) for a, b in zip(root, candidate, strict=True)) for candidate in unmatched
        ]
        index = min(range(len(distances)), key=distances.__getitem__)
        assert distances[index] < tolerance
        unmatched.pop(index)
    assert not unmatched


@st.composite
def finite_separable_systems(draw):
    a, b = draw(
        st.sampled_from(
            tuple((left, right) for left in range(1, 6) for right in range(1, 6) if left != right)
        )
    )
    shift_x = draw(st.integers(min_value=-2, max_value=2))
    shift_y = draw(st.integers(min_value=-2, max_value=2))
    return ((x - shift_x) ** 2 - a, (y - shift_y) ** 2 - b)


@given(
    finite_separable_systems(),
    st.sampled_from((-4, -3, -2, -1, 1, 2, 3, 4)),
    st.sampled_from((-4, -3, -2, -1, 1, 2, 3, 4)),
)
@settings(max_examples=12, deadline=None)
def test_nonzero_equation_scaling_preserves_roots(equations, first_scale, second_scale):
    baseline = polysolve(equations, (x, y), digits=30, recognize=False)
    transformed = polysolve(
        (first_scale * equations[0], second_scale * equations[1]),
        (x, y),
        digits=30,
    )
    _assert_same_root_set(baseline.roots, transformed.roots)


@given(
    finite_separable_systems(),
    st.integers(min_value=-3, max_value=3),
    st.integers(min_value=-3, max_value=3),
)
@settings(max_examples=12, deadline=None)
def test_unimodular_equation_recombination_preserves_roots(equations, upper, lower):
    first, second = equations
    # Product of elementary row operations; determinant is exactly one.
    transformed_equations = (
        first + upper * second,
        lower * first + (1 + lower * upper) * second,
    )
    baseline = polysolve(equations, (x, y), digits=30, recognize=False)
    transformed = polysolve(transformed_equations, (x, y), digits=30, recognize=False)
    _assert_same_root_set(baseline.roots, transformed.roots)


@given(finite_separable_systems(), st.booleans())
@settings(max_examples=10, deadline=None)
def test_equation_order_and_eq_wrapping_preserve_roots(equations, reverse):
    baseline = polysolve(equations, (x, y), digits=30, recognize=False)
    ordered = tuple(reversed(equations)) if reverse else equations
    wrapped = tuple(sp.Eq(equation, 0) for equation in ordered)
    transformed = polysolve(wrapped, (x, y), digits=30, recognize=False)
    _assert_same_root_set(baseline.roots, transformed.roots)


@given(
    finite_separable_systems(),
    st.sampled_from((-2, -1, 1, 2)),
    st.sampled_from((-2, -1, 1, 2)),
    st.integers(min_value=-2, max_value=2),
    st.integers(min_value=-2, max_value=2),
)
@settings(max_examples=12, deadline=None)
def test_invertible_affine_coordinate_change_maps_back_to_same_roots(
    equations,
    scale_x,
    scale_y,
    translate_x,
    translate_y,
):
    baseline = polysolve(equations, (x, y), digits=32, recognize=False)
    substitution = {
        x: scale_x * x + translate_x,
        y: scale_y * y + translate_y,
    }
    changed_equations = tuple(sp.expand(eq.subs(substitution)) for eq in equations)
    changed = polysolve(changed_equations, (x, y), digits=32, recognize=False)
    mapped_back = tuple(
        (
            scale_x * root[0] + translate_x,
            scale_y * root[1] + translate_y,
        )
        for root in changed.roots
    )
    _assert_same_root_set(baseline.roots, mapped_back)


@given(
    st.integers(min_value=1, max_value=5),
    st.integers(min_value=1, max_value=5),
    st.integers(min_value=-2, max_value=2),
)
@settings(max_examples=10, deadline=None)
def test_shear_coordinate_change_preserves_constructed_four_root_set(a, b, shear):
    equations = (x**2 - a, y**2 - b)
    baseline = polysolve(equations, (x, y), digits=30, recognize=False)
    changed_equations = tuple(sp.expand(eq.subs(x, x + shear * y)) for eq in equations)
    changed = polysolve(changed_equations, (x, y), digits=30, recognize=False)
    mapped_back = tuple((root[0] + shear * root[1], root[1]) for root in changed.roots)
    _assert_same_root_set(baseline.roots, mapped_back)


def test_all_small_equation_permutations_preserve_three_variable_root_set():
    z = sp.symbols("z")
    equations = (x**2 - 2, y**2 - 3, z - x - y)
    baseline = polysolve(equations, (x, y, z), digits=30, recognize=False)
    for permutation in itertools.permutations(equations):
        transformed = polysolve(permutation, (x, y, z), digits=30, recognize=False)
        _assert_same_root_set(baseline.roots, transformed.roots)
