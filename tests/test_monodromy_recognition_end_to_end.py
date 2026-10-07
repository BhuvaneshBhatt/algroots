"""End-to-end differential tests across core solving, monodromy, and recognition."""

import pytest
import sympy as sp

from algroots import polysolve
from algroots.continuation import PathTrackerOptions
from algroots.monodromy import discover_monodromy_orbit
from algroots.recognition import recognize_system_roots

pytestmark = pytest.mark.recognition


x, y = sp.symbols("x y")
OPTIONS = PathTrackerOptions(
    initial_step=0.03,
    max_step=0.07,
    initial_digits=45,
    residual_digits=28,
)


def _exact_set(recognized):
    return {tuple(sp.simplify(value) for value in item.exact_coordinates) for item in recognized}


@pytest.mark.slow
def test_core_seed_monodromy_and_real_recognition_agree_for_quadratic() -> None:
    pytest.importorskip("flint")
    pytest.importorskip("algrecognize")

    equations = (x**2 - 1,)
    core = polysolve(equations, (x,), digits=60, recognize=False)
    orbit = discover_monodromy_orbit(
        equations,
        (x,),
        core.roots[:1],
        expected_root_count=2,
        max_loops=4,
        radius=2.0,
        random_seed=0,
        options=OPTIONS,
        recognize=False,
    )
    recognized = recognize_system_roots(
        orbit,
        max_degree=1,
        require_certified=True,
    )

    assert _exact_set(recognized) == {(sp.Integer(-1),), (sp.Integer(1),)}
    assert orbit.completeness_basis == "exact_root_count"
    assert all(item.certified for item in recognized)


@pytest.mark.slow
def test_multivariate_core_roots_round_trip_through_monodromy_result_and_recognition() -> None:
    pytest.importorskip("flint")
    pytest.importorskip("algrecognize")

    equations = (x**2 - 1, y - x)
    core = polysolve(equations, (x, y), method="action", digits=60, recognize=False)
    orbit = discover_monodromy_orbit(
        equations,
        (x, y),
        core.roots,
        expected_root_count=2,
        max_loops=1,
        options=OPTIONS,
        recognize=False,
    )
    recognized = recognize_system_roots(
        orbit,
        max_degree=1,
        require_certified=True,
    )

    assert _exact_set(recognized) == {
        (sp.Integer(-1), sp.Integer(-1)),
        (sp.Integer(1), sp.Integer(1)),
    }
    assert all(item.equation_values == (0, 0) for item in recognized)
