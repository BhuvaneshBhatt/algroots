"""Mathematical results, verified routing, and failure contracts of the public API."""

import pytest
import sympy as sp

from algroots import (
    GroebnerLimitError,
    GroebnerVerificationError,
    NativeF4Limits,
    QuotientAlgebra,
    algsolve,
    groebner_basis,
    polysolve,
)
from algroots.native_f4 import compiled_available

x, y = sp.symbols("x y")
SYSTEM = (x**2 - y, y**2 - 1)


@pytest.mark.parametrize("backend", ["sympy", "native"])
@pytest.mark.parametrize("order", ["lex", "grlex", "grevlex"])
def test_basis(backend, order):
    if backend == "native" and not compiled_available():
        pytest.skip("compiled kernel is not installed")
    basis = groebner_basis(iter(SYSTEM), (x, y), backend=backend, order=order)
    assert isinstance(basis, sp.GroebnerBasis)
    assert basis == sp.groebner(SYSTEM, x, y, domain=sp.QQ, order=order)
    assert basis.reduce(x**4 - 1)[1] == 0
    assert basis.reduce(x)[1] == x


@pytest.mark.parametrize(
    "generators, expected", [((), []), ((0,), []), ((1,), [1]), ((x, x - 1), [1])]
)
def test_degenerate(generators, expected):
    assert list(groebner_basis(iter(generators), (x,), backend="sympy")) == expected


@pytest.mark.parametrize("domain", [sp.QQ.algebraic_field(sp.I), sp.QQ.frac_field(sp.Symbol("t"))])
def test_fields(domain):
    coefficient = (
        domain.ext.as_expr() if domain.is_AlgebraicField else domain.to_sympy(domain.gens[0])
    )
    basis = groebner_basis((x - coefficient,), (x,), domain=domain, backend="sympy")
    assert basis.domain == domain
    assert sp.expand(basis.reduce(x**2)[1] - coefficient**2) == 0


@pytest.mark.parametrize("domain", [sp.GF(6), sp.RR, sp.ZZ])
def test_invalid_fields(domain):
    with pytest.raises(ValueError):
        groebner_basis(SYSTEM, (x, y), domain=domain, backend="sympy")


def test_limits():
    for value in (True, 1.5, 0):
        with pytest.raises(ValueError):
            NativeF4Limits(max_terms=value)
    for value in (True, 15, 1000000009):
        with pytest.raises(ValueError):
            NativeF4Limits(first_prime=value)
    with pytest.raises(ValueError, match="limits"):
        groebner_basis(SYSTEM, (x, y), limits=NativeF4Limits(), backend="sympy")
    with pytest.raises(TypeError, match="NativeF4Limits"):
        groebner_basis(SYSTEM, (x, y), backend="native", limits=object())


@pytest.mark.skipif(not compiled_available(), reason="native F4 extension unavailable")
def test_reject_wrong_ideal(monkeypatch):
    from algroots import groebner

    monkeypatch.setattr(
        groebner, "compute_groebner_basis", lambda inputs, ring, **kwargs: (ring.one,)
    )
    with pytest.raises(GroebnerVerificationError):
        groebner_basis(SYSTEM, (x, y), backend="native")


@pytest.mark.parametrize("entry", ["basis", "quotient", "solver", "algebraic", "rur"])
@pytest.mark.skipif(not compiled_available(), reason="native F4 extension unavailable")
def test_backend_routing(entry, monkeypatch):
    from algroots import groebner
    from algroots.rational_univariate import compute_rational_univariate_representation

    calls = []
    original = groebner.compute_groebner_basis

    def record(inputs, ring, **kwargs):
        calls.append(kwargs["backend"])
        return original(inputs, ring, **kwargs)

    monkeypatch.setattr(groebner, "compute_groebner_basis", record)
    if entry == "basis":
        assert list(groebner_basis(SYSTEM, (x, y), backend="native")) == list(
            sp.groebner(SYSTEM, x, y, domain=sp.QQ, order="grevlex")
        )
    elif entry == "quotient":
        quotient = QuotientAlgebra.from_polynomials(SYSTEM, (x, y), groebner_backend="native")
        assert quotient.dimension == 4
        assert quotient.normal_form(x**4 - 1) == 0
    elif entry == "rur":
        assert (
            compute_rational_univariate_representation(
                SYSTEM, (x, y), groebner_backend="native"
            ).solution_count
            == 4
        )
    else:
        solver = polysolve if entry == "solver" else algsolve
        result = solver(SYSTEM, (x, y), groebner_backend="native", recognize=False, presolve=False)
        assert len(result.roots) == 4
        for a, b in result.roots:
            assert abs(complex(a * a - b)) < 1e-40
            assert abs(complex(b * b - 1)) < 1e-40
    assert calls and set(calls) == {"native"}


@pytest.mark.skipif(not compiled_available(), reason="compiled kernel is not installed")
@pytest.mark.parametrize("limits", [NativeF4Limits(max_terms=1), NativeF4Limits(max_basis=1)])
def test_native_budget(limits):
    with pytest.raises(GroebnerLimitError):
        groebner_basis(SYSTEM, (x, y), domain=sp.GF(101), backend="native", limits=limits)


@pytest.mark.skipif(not compiled_available(), reason="compiled kernel is not installed")
def test_native_invariant_error(monkeypatch):
    _native_f4 = pytest.importorskip("algroots._native_f4")

    def broken(*args, **kwargs):
        raise RuntimeError("invariant failure")

    monkeypatch.setattr(_native_f4, "compute", broken)
    with pytest.raises(RuntimeError, match="invariant failure") as failure:
        groebner_basis(SYSTEM, (x, y), domain=sp.GF(101), backend="native")
    assert not isinstance(failure.value, GroebnerLimitError)


@pytest.mark.skipif(not compiled_available(), reason="native F4 extension unavailable")
def test_early_selection():
    with pytest.raises(ValueError, match="backend"):
        polysolve([x - 1], [x], groebner_backend="unknown", recognize=False)
    with pytest.raises(ValueError, match="homotopy"):
        polysolve(
            [x - 1], [x], method="homotopy", groebner_limits=NativeF4Limits(), recognize=False
        )


@pytest.mark.skipif(not compiled_available(), reason="compiled kernel is not installed")
def test_module_exception_ownership():
    import importlib.util

    from algroots import _native_f4

    spec = importlib.util.spec_from_file_location("_native_f4", _native_f4.__file__)
    other = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(other)
    with pytest.raises(other.LimitError):
        other.compute([{(2, 0): 1, (0, 1): 1}], 101, 2, max_terms=1)
    with pytest.raises(GroebnerLimitError):
        groebner_basis(
            SYSTEM, (x, y), domain=sp.GF(101), backend="native", limits=NativeF4Limits(max_terms=1)
        )


@pytest.mark.parametrize("backend", ["sympy"])
def test_native_ring_inputs(backend):
    from sympy.polys.rings import PolyRing

    from algroots.groebner import compute_groebner_basis

    ring = PolyRing((x, y), sp.QQ, "grevlex")
    equations = (x * y - 1, x * x - y)
    inputs = [ring.from_expr(p) for p in equations]
    before = [p.copy() for p in inputs]
    reference = sp.groebner(equations, x, y, domain=sp.QQ, order="grevlex")
    expected = [ring.from_expr(p.as_expr()) for p in reference.polys]
    result = compute_groebner_basis(inputs, ring, backend=backend)
    assert inputs == before
    assert {frozenset(p.items()) for p in result} == {frozenset(p.items()) for p in expected}


@pytest.mark.skipif(not compiled_available(), reason="compiled kernel is not installed")
@pytest.mark.parametrize("first_prime", [1000000007, 999999937])
def test_prime_range_budget(first_prime):
    with pytest.raises(GroebnerLimitError, match="prime"):
        groebner_basis(
            [x * x - 2], (x,), backend="native", limits=NativeF4Limits(first_prime=first_prime)
        )


@pytest.mark.skipif(not compiled_available(), reason="compiled kernel is not installed")
def test_error_lookup_failure():
    import subprocess
    import sys

    code = """
from algroots import _native_f4
del _native_f4.LimitError
try:
    _native_f4.compute([{(2,0):1,(0,1):1}],101,2,max_terms=1)
except AttributeError as error:
    if "LimitError" not in str(error):
        raise
else:
    raise RuntimeError("missing exception type was not reported")
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("parameter_field", [False, True])
@pytest.mark.parametrize("backend", ["sympy"])
def test_bound_coefficient_symbols(parameter_field, backend):
    t = sp.Symbol("t")
    alpha = sp.CRootOf(x**3 - x - 1, 0)
    field = sp.QQ.algebraic_field(alpha)
    coefficient = alpha
    if parameter_field:
        field = field.frac_field(t)
        coefficient = alpha / (t + 1)
    basis = groebner_basis([x - coefficient], (x,), domain=field, backend=backend)
    assert basis.domain == field
    assert basis.reduce(x)[1] == coefficient
    quotient = QuotientAlgebra.from_polynomials(
        [x - coefficient], (x,), domain=field, groebner_backend=backend
    )
    assert quotient.dimension == 1
    assert quotient.normal_form(x) == coefficient


def test_native_default():
    from algroots import GroebnerBackendUnavailable, groebner_backends

    reference = sp.groebner(SYSTEM, x, y, domain=sp.QQ, order="grevlex")
    if groebner_backends()["native"].available:
        assert groebner_basis(SYSTEM, (x, y)) == reference
    else:
        with pytest.raises(GroebnerBackendUnavailable, match="native"):
            groebner_basis(SYSTEM, (x, y))
    assert groebner_basis(SYSTEM, (x, y), backend="sympy") == reference


@pytest.mark.parametrize(
    "backend", ["flint", "modular", "f5b", "native-f4", "singular", "macaulay2"]
)
def test_removed_backends(backend):
    with pytest.raises(ValueError, match="sympy or native"):
        groebner_basis(SYSTEM, (x, y), backend=backend)
