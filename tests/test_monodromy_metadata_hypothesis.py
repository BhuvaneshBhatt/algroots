"""Property-based invariants for monodromy per-root metadata."""

import mpmath as mp
import pytest

from algroots.monodromy import MonodromyOrbitResult, MonodromyRootInfo

hypothesis = pytest.importorskip("hypothesis")
st = pytest.importorskip("hypothesis.strategies")
given = hypothesis.given


@st.composite
def root_info_strategy(draw):
    digits = draw(st.integers(min_value=15, max_value=250))
    verification_digits = draw(st.integers(min_value=1, max_value=digits))
    exponent = draw(st.integers(min_value=-200, max_value=-1))
    source_loop = draw(st.one_of(st.none(), st.integers(min_value=0, max_value=20)))
    path_index = draw(st.one_of(st.none(), st.integers(min_value=0, max_value=20)))
    return MonodromyRootInfo(
        digits=digits,
        verification_digits=verification_digits,
        relative_residual=mp.mpf(10) ** exponent,
        verified=True,
        source_loop=source_loop,
        path_index=path_index,
    )


@given(st.lists(root_info_strategy(), min_size=0, max_size=8))
def test_aligned_root_info_round_trips_as_public_result_metadata(items):
    roots = tuple((index,) for index in range(len(items)))
    result = MonodromyOrbitResult(
        roots=roots,
        variables=(),
        equations=(),
        loops_completed=0,
        paths_tracked=0,
        new_roots_per_loop=(),
        perturbations=(),
        all_paths_successful=True,
        root_info=tuple(items),
    )
    assert len(result.root_info) == len(result.roots)
    for info in result.root_info:
        assert 1 <= info.verification_digits <= info.digits
        assert mp.isfinite(info.relative_residual)
        assert info.relative_residual >= 0


@given(root_info_strategy(), st.integers(min_value=1, max_value=5))
def test_misaligned_nonempty_root_info_is_rejected(info, extra_roots):
    roots = tuple((index,) for index in range(extra_roots + 1))
    with pytest.raises(ValueError, match="one-to-one"):
        MonodromyOrbitResult(
            roots=roots,
            variables=(),
            equations=(),
            loops_completed=0,
            paths_tracked=0,
            new_roots_per_loop=(),
            perturbations=(),
            all_paths_successful=True,
            root_info=(info,),
        )
