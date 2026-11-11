"""Precision ownership, restoration and high-precision midpoint conversion."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event

import mpmath as mp
import pytest
from flint import acb, ctx

from algroots.numerical import acb_mid_to_mpc, flint_dps


@pytest.mark.parametrize("imaginary", [False, True])
def test_midpoint_precision(imaginary):
    decimal = "1.2345678901234567890123456789012345678901234567890123456789"
    original = mp.mp.prec
    with flint_dps(90):
        value = acb(decimal, decimal if imaginary else "0")
    result = acb_mid_to_mpc(value, 75)
    with mp.workdps(90):
        expected = mp.mpc(decimal, decimal if imaginary else "0")
        assert abs(result - expected) < mp.mpf("1e-75")
    assert mp.mp.prec == original


@pytest.mark.parametrize("scope", ["flint", "mpmath"])
def test_thread_precision(scope):
    context = flint_dps
    if scope == "mpmath":
        from algroots._precision import working_precision

        context = working_precision
    original, mp_original = ctx.prec, mp.mp.prec
    first_ready, second_attempt, second_entered, release = (Event() for _ in range(4))

    def first():
        with context(80):
            first_ready.set()
            if not release.wait(5):
                raise TimeoutError("test did not release first precision scope")
            return ctx.dps if scope == "flint" else mp.mp.dps

    def second():
        if not first_ready.wait(5):
            raise TimeoutError("first precision scope did not start")
        second_attempt.set()
        with flint_dps(20):
            second_entered.set()
            return ctx.dps

    try:
        with ThreadPoolExecutor(2) as pool:
            left, right = pool.submit(first), pool.submit(second)
            assert second_attempt.wait(5)
            overlap = second_entered.wait(0.1)
            release.set()
            observed = left.result(), right.result()
        assert not overlap
        assert observed == (80, 20)
        assert ctx.prec == original
    finally:
        release.set()
        ctx.prec = original
        mp.mp.prec = mp_original


def test_nested_precision():
    from algroots._precision import working_precision

    bits, flint_bits = mp.mp.prec, ctx.prec
    with working_precision(70), flint_dps(80):
        assert mp.mp.dps == 70 and ctx.dps == 80
        with pytest.raises(RuntimeError, match="interrupt"), working_precision(30), flint_dps(40):
            assert mp.mp.dps == 30 and ctx.dps == 40
            raise RuntimeError("interrupt")
        assert mp.mp.dps == 70 and ctx.dps == 80
    assert (mp.mp.prec, ctx.prec) == (bits, flint_bits)


@pytest.mark.parametrize("part", ["nan", "inf"])
def test_nonfinite_midpoint(part):
    with pytest.raises(ValueError, match="nonfinite"):
        acb_mid_to_mpc(acb(part), 50)


def test_homotopy_coordinate_name():
    import sympy as sp

    from algroots import polysolve

    x = sp.Symbol("_algroots_homotopy_t", real=True)
    result = polysolve([x * x - 1], (x,), method="homotopy", recognize=False, presolve=False)
    assert len(result.roots) == 2
    assert sorted(round(float(point[0]), 12) for point in result.roots) == [-1, 1]
    assert max(abs(complex(point[0] ** 2 - 1)) for point in result.roots) < 1e-40


def test_homotopy_parameter():
    import sympy as sp

    from algroots.continuation import SympyHomotopy
    from algroots.total_degree_homotopy import TotalDegreeHomotopy

    x = sp.Symbol("x")
    homotopy = TotalDegreeHomotopy((x**2 - 2,), (x,), (2,), 1)
    assert homotopy.system().parameter == homotopy.parameter
    assert homotopy.system().expressions == homotopy.expressions()
    assert homotopy.expressions()[0].subs(homotopy.parameter, 1) == x**2 - 2
    with pytest.raises(ValueError, match="distinct"):
        SympyHomotopy([x**2 - 2], (x,), x)


def test_reused_homotopy_parameter():
    import sympy as sp

    from algroots.total_degree_homotopy import build_total_degree_homotopy

    x = sp.Symbol("x")
    first = build_total_degree_homotopy((x**2 - 2,), (x,))
    coordinate = first.parameter
    second = build_total_degree_homotopy((coordinate**2 - 2,), (coordinate,))
    assert second.parameter != coordinate
    assert second.expressions()[0].subs(second.parameter, 1) == coordinate**2 - 2


def test_compiled_homotopy_precision():
    import sympy as sp

    from algroots._precision import working_precision
    from algroots.continuation import SympyHomotopy

    x, t = sp.symbols("x t")
    for digits in (30, 80):
        system = SympyHomotopy((x**2 - t,), (x,), t)
        different = SympyHomotopy((x**2 + t,), (x,), t)
        with working_precision(digits):
            root = mp.sqrt(2)
            assert abs(system.evaluate((root,), 2)[0]) < mp.mpf(10) ** (-digits + 2)
            assert abs(system.jacobian((root,), 2)[0] - 2 * root) == 0
            assert system.parameter_derivative((root,), 2)[0] == -1
            assert abs(different.evaluate((root,), 2)[0] - 4) < mp.mpf(10) ** (-digits + 2)
