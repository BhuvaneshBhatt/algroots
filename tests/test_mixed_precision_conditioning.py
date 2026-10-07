import sympy as sp

from algroots.continuation import PathTrackerOptions, SympyHomotopy, track_path

x, y, t = sp.symbols("x y t")


def test_conditioning_triggers_precision_increase_without_mocking():
    eps = sp.Rational(1, 10**20)
    target_x = 1 + t
    target_y = 2 - t
    equations = [
        (x - target_x) + (y - target_y),
        (x - target_x) + (1 + eps) * (y - target_y),
    ]
    homotopy = SympyHomotopy(equations, (x, y), t)
    result = track_path(
        homotopy,
        (1, 2),
        options=PathTrackerOptions(
            initial_digits=30,
            max_digits=120,
            residual_digits=18,
            condition_guard_digits=10,
            initial_step=0.05,
            max_step=0.1,
        ),
    )
    assert result.success
    assert result.precision_increases >= 1
    assert result.final_digits > 30
    assert abs(result.endpoint[0] - 2) < 1e-12
    assert abs(result.endpoint[1] - 1) < 1e-12
