import os

import pytest
import sympy as sp

from algroots import polysolve
from algroots.numerical import flint_available

x, y = sp.symbols("x y")


@pytest.mark.parametrize("power", [10, 20, 30])
def test_exact_clustered_roots_remain_distinct(power):
    eps = sp.Rational(1, 10**power)
    result = polysolve(
        [(x - eps) * (x + eps)],
        (x,),
        method="shape",
        digits=max(45, power + 15),
        max_precision_digits=max(120, 2 * power + 40),
    )
    assert len(result.roots) == 2
    values = sorted(complex(sp.N(root[0], power + 10)).real for root in result.roots)
    assert values[0] < 0 < values[1]


@pytest.mark.flint
def test_action_precision_retry_after_one_controlled_isolation_failure(monkeypatch):
    if not flint_available():
        if os.environ.get("ALGROOTS_REQUIRE_FLINT") == "1":
            pytest.fail("python-flint required for precision-retry test")
        pytest.skip("python-flint unavailable")

    import algroots.solver as solver

    real_builder = solver._acb_separator_matrix
    calls = {"count": 0}

    def fail_once(columns, digits):
        calls["count"] += 1
        if calls["count"] == 1:
            raise ValueError("controlled insufficient-precision isolation failure")
        return real_builder(columns, digits)

    monkeypatch.setattr(solver, "_acb_separator_matrix", fail_once)
    result = polysolve(
        [x**2 - 2, y**2 - 3],
        (x, y),
        method="action",
        digits=35,
        max_precision_digits=200,
    )
    assert len(result.roots) == 4
    assert calls["count"] >= 2
    assert result.working_digits > 35


def test_invalid_precision_ceiling_is_rejected():
    with pytest.raises(ValueError):
        polysolve([x**2 - 2], (x,), digits=50, max_precision_digits=40)


def test_internal_root_deduplication_uses_requested_precision():
    from algroots.solver import RootDiagnostics, _deduplicate_records

    digits = 80
    delta = sp.Float(10, digits) ** -60
    roots = [(sp.Float(1, digits),), (sp.Float(1, digits) + delta,)]
    diagnostics = RootDiagnostics(0, 0, False, False, False)
    unique = _deduplicate_records([(root, diagnostics) for root in roots], digits)
    assert len(unique) == 2
