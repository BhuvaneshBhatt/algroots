"""Public discovery, grouped settings, and supervised mathematical results."""

import multiprocessing as mp
import os
import sys
import time

import pytest
import sympy as sp

from algroots import (
    ExecutionError,
    ExecutionLimits,
    ExecutionTimeout,
    GroebnerBackendUnavailable,
    NativeF4Limits,
    PrecisionOptions,
    QuotientAlgebra,
    SolverLimits,
    SolverOptions,
    algsolve,
    groebner_backends,
    groebner_basis,
    polysolve,
    run_supervised,
)

x, y = sp.symbols("x y")


def _delayed_reference():
    from unittest.mock import patch

    from sympy.polys.groebnertools import groebner

    def reference(*args, **kwargs):
        time.sleep(20)
        return groebner(*args, **kwargs)

    with patch("sympy.polys.groebnertools.groebner", reference):
        return groebner_basis([x**2 - 2], (x,), backend="native")


def _untransferable():
    return lambda: 1


def _allocate():
    return bytearray(1024**3)


def test_discovery():
    engines = groebner_backends()
    assert set(engines) == {"sympy", "native"}
    assert engines["sympy"].available
    assert engines["sympy"].limits_type is None
    assert engines["native"].limits_type is NativeF4Limits
    assert engines["native"].domains[:2] == ("QQ", "prime fields")
    assert "rational function fields over QQ" in engines["native"].domains
    assert engines["native"].max_variables == 64
    for name, info in engines.items():
        if not info.available:
            with pytest.raises(GroebnerBackendUnavailable, match="sympy"):
                groebner_basis([x**2 - 1], (x,), backend=name)


@pytest.mark.parametrize("backend", ["flint", "modular", "singular", "macaulay2"])
def test_domain_guidance(backend):
    with pytest.raises(ValueError, match="sympy or native"):
        groebner_basis([x**2 + 1], (x,), domain=sp.QQ.algebraic_field(sp.I), backend=backend)


def test_unknown_backend():
    with pytest.raises(ValueError, match="sympy or native"):
        groebner_basis([x], (x,), backend="missing")


@pytest.mark.parametrize("solver", [polysolve, algsolve])
def test_grouped_settings(solver):
    settings = SolverOptions(
        precision=PrecisionOptions(digits=35, recognize=False),
        limits=SolverLimits(max_action_dimension=8),
    )
    grouped = solver([x**2 - 2, y - x], (x, y), options=settings, presolve=False)
    flat = solver(
        [x**2 - 2, y - x],
        (x, y),
        digits=35,
        recognize=False,
        max_action_dimension=8,
        presolve=False,
    )
    assert grouped.roots == flat.roots
    assert grouped.variables == (x, y)
    with pytest.raises(ValueError, match="nondefault digits"):
        solver([x], (x,), digits=60, options=settings)
    with pytest.raises(TypeError, match="SolverOptions"):
        solver([x], (x,), options={})


@pytest.mark.parametrize(
    "constructor,kwargs",
    [
        (PrecisionOptions, {"digits": True}),
        (PrecisionOptions, {"verification_digits": 50}),
        (PrecisionOptions, {"max_precision_digits": 20}),
        (SolverLimits, {"maxsteps": 0}),
        (SolverOptions, {"precision": {}}),
        (ExecutionLimits, {"timeout": float("nan")}),
        (ExecutionLimits, {"timeout": True}),
        (ExecutionLimits, {"timeout": 1, "memory_mib": -1}),
    ],
)
def test_invalid_settings(constructor, kwargs):
    with pytest.raises((TypeError, ValueError)):
        constructor(**kwargs)


@pytest.mark.parametrize("order", ["lex", "grlex", "grevlex"])
@pytest.mark.parametrize(
    "domain",
    [
        sp.QQ,
        sp.GF(101),
        sp.QQ.algebraic_field(sp.I),
        sp.QQ.frac_field(sp.Symbol("t")),
        sp.QQ.algebraic_field(sp.I).frac_field(sp.Symbol("t")),
    ],
)
def test_supervised_basis(order, domain):
    basis = groebner_basis(
        [x**2 - y, y**2 - 1],
        (x, y),
        domain=domain,
        order=order,
        backend="sympy",
        execution=ExecutionLimits(20),
    )
    assert basis == groebner_basis(
        [x**2 - y, y**2 - 1], (x, y), domain=domain, order=order, backend="sympy"
    )
    assert basis.reduce(x**4 - 1)[1] == 0
    quotient = run_supervised(
        QuotientAlgebra.from_groebner_basis, basis, execution=ExecutionLimits(20)
    )
    assert quotient.groebner_basis == basis
    assert quotient.normal_form(x**4 - 1) == 0


def test_supervised_quotient():
    quotient = run_supervised(
        QuotientAlgebra.from_polynomials,
        [x**2 - y, y**2 - 1],
        (x, y),
        execution=ExecutionLimits(20),
    )
    assert quotient.dimension == 4
    assert quotient.normal_form(x**4 - 1) == 0


@pytest.mark.parametrize("solver", [polysolve, algsolve])
def test_supervised_solver(solver):
    settings = SolverOptions(
        precision=PrecisionOptions(recognize=False), execution=ExecutionLimits(20)
    )
    result = solver([x**2 - 1], (x,), options=settings)
    assert sorted(float(root[0]) for root in result.roots) == [-1, 1]


def test_deadline():
    existing = {p.pid for p in mp.active_children()}
    with pytest.raises(ExecutionTimeout, match="no result"):
        run_supervised(time.sleep, 20, execution=ExecutionLimits(0.2))
    assert {p.pid for p in mp.active_children()} == existing


@pytest.mark.skipif(
    not groebner_backends()["native"].available, reason="native F4 extension unavailable"
)
def test_verification_deadline():
    with pytest.raises(ExecutionTimeout):
        run_supervised(_delayed_reference, execution=ExecutionLimits(2))


def test_worker_errors():
    with pytest.raises(ValueError, match="sympy or native"):
        groebner_basis([x], (x,), backend="missing", execution=ExecutionLimits(20))
    with pytest.raises(ExecutionError, match="result must be pickleable"):
        run_supervised(_untransferable, execution=ExecutionLimits(20))


def test_transfer_inputs():
    with pytest.raises(TypeError, match="materialized inputs"):
        groebner_basis(
            (p for p in [x * x - 1]), (x,), execution=ExecutionLimits(20), backend="sympy"
        )


def test_bound_constant_transfer():
    t = sp.Symbol("t")
    alpha = sp.CRootOf(x**3 - x - 1, 0)
    domain = sp.QQ.algebraic_field(alpha).frac_field(t)
    value = alpha / (t + 1)
    result = groebner_basis(
        [x - value], (x,), domain=domain, execution=ExecutionLimits(20), backend="sympy"
    )
    assert result.domain == domain
    assert result.reduce(x)[1] == value


def test_settings_defaults():
    from dataclasses import fields
    from inspect import signature

    for solver in (polysolve, algsolve):
        defaults = signature(solver).parameters
        for group in (PrecisionOptions, SolverLimits):
            assert {setting.name: setting.default for setting in fields(group)} == {
                setting.name: defaults[setting.name].default for setting in fields(group)
            }


def test_memory_cap():
    if sys.platform.startswith("linux"):
        with pytest.raises((MemoryError, ExecutionError)):
            run_supervised(_allocate, execution=ExecutionLimits(20, memory_mib=256))
    else:
        with pytest.raises(NotImplementedError, match="Linux"):
            run_supervised(os.getpid, execution=ExecutionLimits(20, memory_mib=256))
