"""Native construction defaults and explicit portable selection across public consumers."""

import inspect

import pytest
import sympy as sp

from algroots import (
    GroebnerBackendUnavailable,
    NativeF4Limits,
    QuotientAlgebra,
    algsolve,
    certify_groebner_basis,
    certify_rational_univariate_representation,
    groebner_basis,
    parametric_groebner_basis,
    polysolve,
)
from algroots.border_basis import compute_border_basis, compute_border_basis_linear
from algroots.native_f4 import compiled_available
from algroots.rational_univariate import (
    compute_rational_univariate_representation,
    solve_zero_dimensional_system_with_rur,
)
from algroots.verified_families import VerifiedGroebnerFamily

x, y, t = sp.symbols("x y t")
SYSTEM = (x**2 - y, y**2 - 1)


@pytest.mark.parametrize(
    "function, parameter",
    [
        (groebner_basis, "backend"),
        (polysolve, "groebner_backend"),
        (algsolve, "groebner_backend"),
        (QuotientAlgebra.from_polynomials, "groebner_backend"),
        (compute_rational_univariate_representation, "groebner_backend"),
        (solve_zero_dimensional_system_with_rur, "groebner_backend"),
        (compute_border_basis, "groebner_backend"),
        (compute_border_basis_linear, "groebner_backend"),
        (parametric_groebner_basis, "backend"),
        (certify_groebner_basis, "backend"),
        (certify_rational_univariate_representation, "backend"),
        (VerifiedGroebnerFamily.specialize, "backend"),
    ],
)
def test_signatures(function, parameter):
    assert inspect.signature(function).parameters[parameter].default == "native"


def _construct(entry, backend):
    basis_options = {} if backend is None else {"backend": backend}
    options = {} if backend is None else {"groebner_backend": backend}
    if entry == "basis":
        return tuple(groebner_basis(SYSTEM, (x, y), **basis_options))
    if entry == "quotient":
        algebra = QuotientAlgebra.from_polynomials(SYSTEM, (x, y), **options)
        return algebra.dimension, algebra.normal_form(x**4 - 1)
    if entry == "rur":
        result = compute_rational_univariate_representation(SYSTEM, (x, y), **options)
        return result.solution_count, tuple(result.normalized_coordinate_polynomials())
    if entry == "exact_roots":
        return solve_zero_dimensional_system_with_rur(SYSTEM, (x, y), real=False, **options)
    if entry in {"solver", "algebraic"}:
        solver = polysolve if entry == "solver" else algsolve
        roots = solver(SYSTEM, (x, y), presolve=False, recognize=False, **options).roots
        return sorted((tuple(complex(c) for c in root) for root in roots), key=repr)
    if entry in {"border", "linear_border"}:
        function = compute_border_basis if entry == "border" else compute_border_basis_linear
        result = function(SYSTEM, (x, y), **options)
        return result.order_ideal, result.border_polynomials
    if entry == "certificate":
        return tuple(certify_groebner_basis(SYSTEM, (x, y), **basis_options).verified_basis())
    if entry == "rur_certificate":
        proof = certify_rational_univariate_representation(SYSTEM, (x, y), **basis_options)
        return proof.verified_representation().solution_count
    proof = parametric_groebner_basis([t * x], (x,), (t,), **basis_options)
    return tuple(proof.prepare().specialize({t: 0}, **basis_options)), tuple(proof.basis)


@pytest.mark.parametrize(
    "entry",
    [
        "basis",
        "quotient",
        "rur",
        "exact_roots",
        "solver",
        "algebraic",
        "border",
        "linear_border",
        "certificate",
        "rur_certificate",
        "parameter",
    ],
)
def test_routing(entry, monkeypatch):
    from algroots import groebner

    calls = []
    original = groebner.compute_groebner_basis

    def record(inputs, ring, **options):
        calls.append(options["backend"])
        return original(inputs, ring, **options)

    monkeypatch.setattr(groebner, "compute_groebner_basis", record)
    portable = _construct(entry, "sympy")
    assert set(calls) <= {"sympy"}
    calls.clear()
    if compiled_available():
        assert _construct(entry, None) == portable
        assert calls and set(calls) == {"native"}
    else:
        with pytest.raises(GroebnerBackendUnavailable, match="native"):
            _construct(entry, None)
        assert calls == []


def test_homotopy_selection():
    result = polysolve([x - 1], (x,), method="homotopy", recognize=False, presolve=False)
    assert result.groebner_basis == ()
    assert len(result.roots) == 1
    assert abs(complex(result.roots[0][0] - 1)) < 1e-40
    with pytest.raises(ValueError, match="homotopy.*limits"):
        polysolve([x - 1], (x,), method="homotopy", groebner_limits=NativeF4Limits())


def test_profile_selection():
    from algroots.profiling import GroebnerProfiler

    with GroebnerProfiler() as profile:
        if compiled_available():
            groebner_basis(SYSTEM, (x, y))
        else:
            with pytest.raises(GroebnerBackendUnavailable):
                groebner_basis(SYSTEM, (x, y))
    assert profile.records[0]["backend"] == "native"
    assert profile.records[0]["completed"] == compiled_available()


def test_portable_certified_solver():
    result = polysolve(
        [x**2],
        (x,),
        groebner_backend="sympy",
        recognize=False,
        certify="required",
        root_mode="with_multiplicity",
    )
    assert result.multiplicities == (2,)
    assert tuple(result.iter_roots(with_multiplicity=True)) == ((0,), (0,))
    assert result.root_certifications[0].certificate.verify(groebner_backend="sympy")


def test_portable_specialization():
    field = sp.QQ.frac_field(t)
    algebra = QuotientAlgebra.from_polynomials(
        [t * x, y - x], (x, y), domain=field, groebner_backend="sympy"
    )
    with pytest.raises(ValueError, match="zero-dimensional"):
        algebra.specialize({t: 0}, groebner_backend="sympy")
    regular = algebra.specialize({t: 1}, groebner_backend="sympy")
    assert regular.dimension == 1
    assert regular.normal_form(x) == regular.normal_form(y) == 0
