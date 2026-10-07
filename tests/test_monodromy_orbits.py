import random

import mpmath as mp
import pytest
import sympy as sp

from algroots.continuation import PathTrackerOptions
from algroots.monodromy import (
    deduplicate_roots,
    discover_monodromy_orbit,
    match_root,
    normalized_root_distance,
    random_perturbation,
)

x = sp.symbols("x")
OPTIONS = PathTrackerOptions(
    initial_step=0.03,
    max_step=0.07,
    initial_digits=40,
    residual_digits=24,
)


def test_scale_aware_duplicate_matching():
    roots = ((1000 + 0j,), (-2 + 0j,))
    assert match_root((1000 + 1e-6,), roots, tolerance=1e-8) == 0
    assert match_root((1001,), roots, tolerance=1e-8) is None
    assert normalized_root_distance((1000,), (1000 + 1e-6,)) < 1e-8


def test_deduplication_keeps_first_seen_representative():
    roots = deduplicate_roots([(1,), (1 + 1e-10,), (-1,), (-1 - 1e-11,)], tolerance=1e-8)
    assert len(roots) == 2
    assert roots[0] == (1 + 0j,)
    assert roots[1] == (-1 + 0j,)


def test_random_perturbation_has_requested_norm_and_is_reproducible():
    left = random_perturbation(3, radius=2.5, rng=random.Random(7))
    right = random_perturbation(3, radius=2.5, rng=random.Random(7))
    assert left == right
    assert sum(abs(value) ** 2 for value in left) ** 0.5 == pytest.approx(2.5)


def test_multiple_random_loops_discover_quadratic_orbit_from_one_seed():
    result = discover_monodromy_orbit(
        [x**2 - 1],
        (x,),
        [(1,)],
        max_loops=4,
        radius=2.0,
        random_seed=0,
        options=OPTIONS,
    )
    assert len(result.roots) == 2
    assert result.new_roots_per_loop[0] == 1
    assert result.paths_tracked >= 4
    assert result.all_paths_successful


def test_orbit_discovery_parallel_mode():
    result = discover_monodromy_orbit(
        [x**2 - 1],
        (x,),
        [(1,), (-1,)],
        max_loops=2,
        radius=2.0,
        random_seed=0,
        options=OPTIONS,
        parallel=True,
        max_workers=2,
    )
    assert len(result.roots) == 2
    assert result.paths_tracked == 4


def test_orbit_result_makes_no_completeness_claim():
    result = discover_monodromy_orbit(
        [x**3 - 1],
        (x,),
        [(1,)],
        max_loops=1,
        radius=0.1,
        random_seed=5,
        options=OPTIONS,
    )
    assert result.loops_completed == 1
    assert not hasattr(result, "complete")


def test_ambiguous_root_match_returns_none():
    candidates = ((1.0,), (1.0 + 1e-9,))
    assert match_root((1.0 + 5e-10,), candidates, tolerance=1e-8) is None


def test_failed_paths_do_not_create_roots(monkeypatch):
    from algroots import monodromy as monodromy_module
    from algroots.continuation import PathResult

    failed = PathResult(
        start=(1,),
        endpoint=(99,),
        success=False,
        final_t=0.25,
        final_digits=40,
        accepted_steps=1,
        rejected_steps=3,
        precision_increases=0,
        max_relative_residual=1.0,
        message="controlled failure",
    )

    monkeypatch.setattr(monodromy_module, "track_loop", lambda *args, **kwargs: (failed,))
    result = discover_monodromy_orbit(
        [x**2 - 1],
        (x,),
        [(1,)],
        max_loops=1,
        random_seed=0,
        options=OPTIONS,
    )
    assert result.roots == ((1,),)
    assert result.new_roots_per_loop == (0,)
    assert not result.all_paths_successful


def test_random_orbit_discovery_is_reproducible():
    kwargs = dict(
        equations=[x**2 - 1],
        variables=(x,),
        seeds=[(1,)],
        max_loops=3,
        radius=2.0,
        random_seed=17,
        options=OPTIONS,
    )
    left = discover_monodromy_orbit(**kwargs)
    right = discover_monodromy_orbit(**kwargs)
    assert left.perturbations == right.perturbations
    assert left.new_roots_per_loop == right.new_roots_per_loop
    assert len(left.roots) == len(right.roots)
    for first, second in zip(left.roots, right.roots, strict=True):
        assert normalized_root_distance(first, second) < 1e-12


def test_high_precision_root_distance_does_not_round_through_binary64():
    with mp.workdps(110):
        left = (mp.mpf("1.00000000000000000000000000000000000000000000000001"),)
        right = (mp.mpf("1.00000000000000000000000000000000000000000000000002"),)
    distance = normalized_root_distance(left, right)
    assert distance > 0
    assert distance < mp.mpf("1e-49")


def test_ambiguous_tolerance_chain_is_still_deduplicated():
    roots = deduplicate_roots([(0.0,), (1.5e-8,), (0.75e-8,)], tolerance=1e-8)
    assert len(roots) == 2


def test_expected_root_count_cannot_be_smaller_than_seed_set():
    with pytest.raises(ValueError, match="smaller than"):
        discover_monodromy_orbit(
            [x**2 - 1],
            (x,),
            [(1,), (-1,)],
            expected_root_count=1,
            max_loops=1,
        )


def test_expected_root_count_early_return_still_validates_system():
    with pytest.raises(ValueError):
        discover_monodromy_orbit(
            [],
            (),
            [(1,)],
            expected_root_count=1,
            max_loops=1,
        )


def test_seed_dimension_is_validated_before_expected_count_early_return():
    with pytest.raises(ValueError, match="one coordinate per variable"):
        discover_monodromy_orbit(
            [x**2 - 1],
            (x,),
            [(1, 2)],
            expected_root_count=1,
            max_loops=1,
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0])
def test_invalid_monodromy_radius_is_rejected_before_early_return(value):
    with pytest.raises(ValueError, match="radius"):
        discover_monodromy_orbit(
            [x**2 - 1],
            (x,),
            [(1,), (-1,)],
            radius=value,
            expected_root_count=2,
            max_loops=1,
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0.0, -1.0])
def test_invalid_match_tolerance_is_rejected_before_early_return(value):
    with pytest.raises(ValueError, match="match_tolerance"):
        discover_monodromy_orbit(
            [x**2 - 1],
            (x,),
            [(1,), (-1,)],
            match_tolerance=value,
            expected_root_count=2,
            max_loops=1,
        )


def test_expected_root_count_rejects_discovery_overshoot(monkeypatch):
    from algroots import monodromy as monodromy_module
    from algroots.continuation import PathResult

    paths = (
        PathResult((1,), (2,), True, 1.0, 40, 1, 0, 0, 0.0),
        PathResult((1,), (3,), True, 1.0, 40, 1, 0, 0, 0.0),
    )
    monkeypatch.setattr(monodromy_module, "track_loop", lambda *args, **kwargs: paths)
    with pytest.raises(ValueError, match="more distinct roots"):
        discover_monodromy_orbit(
            [(x - 1) * (x - 2) * (x - 3)],
            (x,),
            [(1,)],
            expected_root_count=2,
            max_loops=1,
        )


def test_expected_count_does_not_trust_an_uncorrectable_seed():
    with pytest.raises(ValueError, match="seed must correct"):
        discover_monodromy_orbit(
            [x**2 + 1],
            (x,),
            [(0,)],
            expected_root_count=1,
            max_loops=1,
        )


def test_orbit_roots_retain_precision_and_verification_metadata():
    result = discover_monodromy_orbit(
        [x**2 - 1],
        (x,),
        [(1,)],
        max_loops=4,
        radius=2.0,
        random_seed=0,
        options=OPTIONS,
    )
    assert len(result.root_info) == len(result.roots) == 2
    assert result.variables == (x,)
    assert result.equations == (x**2 - 1,)
    assert result.root_info[0].source_loop is None
    assert result.root_info[0].path_index == 0
    assert result.root_info[1].source_loop is not None
    for info in result.root_info:
        assert info.digits >= OPTIONS.initial_digits
        assert 1 <= info.verification_digits <= info.digits
        assert info.verification_digits <= OPTIONS.residual_digits
        assert isinstance(info.relative_residual, mp.mpf)
        assert mp.isfinite(info.relative_residual)
        assert info.verified


def test_exact_count_early_return_retains_seed_verification_metadata():
    result = discover_monodromy_orbit(
        [x**2 - 1],
        (x,),
        [(1,), (-1,)],
        expected_root_count=2,
        max_loops=1,
        options=OPTIONS,
    )
    assert result.loops_completed == 0
    assert len(result.root_info) == 2
    assert all(info.source_loop is None for info in result.root_info)
    assert all(info.verified for info in result.root_info)


def test_orbit_result_rejects_misaligned_root_metadata():
    from algroots.monodromy import MonodromyOrbitResult, MonodromyRootInfo

    info = MonodromyRootInfo(40, 24, mp.mpf("1e-30"), True)
    with pytest.raises(ValueError, match="one-to-one"):
        MonodromyOrbitResult(
            roots=((1,), (-1,)),
            variables=(x,),
            equations=(x**2 - 1,),
            loops_completed=0,
            paths_tracked=0,
            new_roots_per_loop=(),
            perturbations=(),
            all_paths_successful=True,
            root_info=(info,),
        )
