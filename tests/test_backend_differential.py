import pytest
import sympy as sp

from algroots import polysolve

x, y, z = sp.symbols("x y z")


def _as_complex(root):
    return tuple(complex(sp.N(value, 20)) for value in root)


def _match_roots(left, right, tol=1e-14):
    unmatched = list(map(_as_complex, right))
    for root in map(_as_complex, left):
        index = min(
            range(len(unmatched)),
            key=lambda i: max(abs(a - b) for a, b in zip(root, unmatched[i], strict=True)),
        )
        distance = max(abs(a - b) for a, b in zip(root, unmatched[index], strict=True))
        assert distance < tol
        unmatched.pop(index)
    assert not unmatched


@pytest.mark.parametrize(
    ("equations", "variables"),
    [
        ([x - y**2, y**3 - 2], (x, y)),
        ([x + y - 1, x - y], (x, y)),
        ([x - y**2, y - z**2, z**2 - 2], (x, y, z)),
        ([x**2 + 1, y - x], (x, y)),
    ],
)
def test_shape_and_action_backends_agree(equations, variables):
    shape = polysolve(equations, variables, method="shape", digits=35)
    action = polysolve(equations, variables, method="action", digits=35)
    assert len(shape.roots) == len(action.roots)
    _match_roots(shape.roots, action.roots)


def test_action_and_triangular_agree_on_branching_system():
    equations = [x**2 - 1, y**2 - 1]
    action = polysolve(equations, (x, y), method="action", digits=35)
    triangular = polysolve(equations, (x, y), method="triangular", digits=35)
    assert len(action.roots) == len(triangular.roots) == 4
    _match_roots(action.roots, triangular.roots)


def test_randomized_shape_action_and_auto_agree_on_constructed_families():
    """Use deterministic exact systems where several independent backends apply."""
    import random

    rng = random.Random(20260821)
    for _ in range(10):
        degree = rng.choice((2, 3))
        constant = rng.choice((1, 2, 3, 5))
        shear = rng.randint(-3, 3)
        offset = rng.randint(-2, 2)
        equations = (
            x - shear * y - offset,
            y**degree - constant,
        )
        shape = polysolve(equations, (x, y), method="shape", digits=40)
        action = polysolve(equations, (x, y), method="action", digits=40)
        automatic = polysolve(equations, (x, y), method="auto", digits=40)

        assert len(shape.roots) == len(action.roots) == len(automatic.roots) == degree
        _match_roots(shape.roots, action.roots, tol=1e-12)
        _match_roots(shape.roots, automatic.roots, tol=1e-12)
        assert shape.max_relative_residual < 1e-25
        assert action.max_relative_residual < 1e-25
        assert automatic.max_relative_residual < 1e-25


def test_randomized_action_and_triangular_agree_on_decoupled_branching_systems():
    import random

    rng = random.Random(731)
    for _ in range(6):
        a = rng.choice((1, 2, 3, 5))
        b = rng.choice((1, 2, 3, 7))
        equations = (x**2 - a, y**2 - b)
        action = polysolve(equations, (x, y), method="action", digits=40)
        triangular = polysolve(
            equations,
            (x, y),
            method="triangular",
            digits=40,
        )
        assert len(action.roots) == len(triangular.roots) == 4
        _match_roots(action.roots, triangular.roots, tol=1e-12)


def test_common_regular_corpus_agrees_across_action_rur_and_homotopy():
    corpus = (
        ((x**2 - 1, y**2 - 4), (x, y), 4),
        ((x**2 + y**2 - 5, x - y), (x, y), 2),
        ((x**2 - 2, y - x - 1), (x, y), 2),
        ((x**3 - 1, y - x), (x, y), 3),
    )
    for equations, variables, expected_count in corpus:
        results = {
            method: polysolve(
                equations,
                variables,
                method=method,
                digits=32,
                recognize=False,
                homotopy_gamma_attempts=2,
                max_homotopy_paths=16,
            )
            for method in ("action", "rur", "homotopy")
        }
        assert all(len(result.roots) == expected_count for result in results.values())
        _match_roots(results["action"].roots, results["rur"].roots, tol=1e-11)
        _match_roots(results["action"].roots, results["homotopy"].roots, tol=1e-11)


def test_shape_joins_common_corpus_when_shape_position_is_available():
    equations = (x - y, y**3 - 2)
    results = {
        method: polysolve(
            equations,
            (x, y),
            method=method,
            digits=32,
            recognize=False,
            homotopy_gamma_attempts=2,
            max_homotopy_paths=8,
        )
        for method in ("shape", "action", "rur", "homotopy")
    }
    assert all(len(result.roots) == 3 for result in results.values())
    for method in ("action", "rur", "homotopy"):
        _match_roots(results["shape"].roots, results[method].roots, tol=1e-11)
