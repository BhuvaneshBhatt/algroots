import mpmath as mp
import pytest
import sympy as sp

from algroots.continuation import PathTrackerOptions, SympyHomotopy, track_path
from algroots.endgames import cauchy_endgame, projective_homotopy

x, t = sp.symbols("x t")


@pytest.mark.parametrize("multiplicity", [1, 2, 3])
def test_cauchy_endpoint_and_cycle_number(multiplicity):
    system = SympyHomotopy((x**multiplicity - (1 - t),), (x,), t)
    with mp.workdps(40):
        start = (mp.mpf("0.1") ** (mp.mpf(1) / multiplicity),)
        result = cauchy_endgame(
            system,
            start,
            samples=16,
            levels=3,
            options=PathTrackerOptions(initial_digits=35, max_digits=70, residual_digits=20),
        )
    assert result.converged
    assert result.cycle_number == multiplicity
    assert abs(result.endpoint[0]) < mp.mpf("1e-10")
    assert result.status == "numerical"
    assert result.limitations


def test_cauchy_nonclosing_cycle_is_not_certified():
    system = SympyHomotopy((x**2 - (1 - t),), (x,), t)
    result = cauchy_endgame(system, (mp.sqrt(mp.mpf("0.1")),), samples=8, max_cycle=1)
    assert not result.converged
    assert result.cycle_number is None


def test_projective_chart_follows_affine_infinity():
    projective = projective_homotopy(((1 - t) * x - 1,), (x,), t, patch=(1, 0))
    path = track_path(
        projective.system,
        projective.lift((sp.Integer(1),)),
        options=PathTrackerOptions(residual_digits=20),
    )
    assert path.success
    assert abs(path.endpoint[0] - 1) < mp.mpf("1e-15")
    assert abs(path.endpoint[1]) < mp.mpf("1e-15")
    assert projective.limitations
    with pytest.raises(ValueError, match="infinity"):
        projective.dehomogenize((1, 0))


def test_projective_finite_endpoint_and_patch_failure():
    projective = projective_homotopy((x - (1 + t),), (x,), t, patch=(0, 1))
    path = track_path(projective.system, projective.lift((sp.Integer(1),)))
    assert path.success
    assert abs(projective.dehomogenize(path.endpoint)[0] - 2) < mp.mpf("1e-15")
    projective = projective_homotopy((x - t,), (x,), t, patch=(1, 0))
    with pytest.raises(ValueError, match="outside"):
        projective.lift((0,))


@pytest.mark.parametrize("kwargs", [{"radius": 0}, {"samples": 2}, {"levels": 1}, {"max_cycle": 0}])
def test_endgame_limits_are_validated(kwargs):
    system = SympyHomotopy((x - t,), (x,), t)
    with pytest.raises(ValueError):
        cauchy_endgame(system, (0,), **kwargs)
