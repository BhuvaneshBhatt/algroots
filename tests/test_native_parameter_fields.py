"""Reconstruct rational-function coefficients from native prime-field images."""

import pytest
import sympy as sp
from sympy.polys.rings import PolyRing

from algroots import GroebnerLimitError, NativeF4Limits, QuotientAlgebra, groebner_basis
from algroots.native_f4 import compiled_available
from algroots.native_fields import native_field_basis

pytestmark = pytest.mark.skipif(not compiled_available(), reason="native F4 is not installed")
x, y, t, u = sp.symbols("x y t u")


@pytest.mark.parametrize("order", ["lex", "grlex", "grevlex"])
@pytest.mark.parametrize(
    "equations",
    [
        [t * x, x * y],
        [t * x - 1, y - x * x],
        [(t + 1) * x - y, y * y - t],
        [x * x - (t * t + 1) / (t - 1), y - x],
        [t, x],
    ],
)
def test_parameter_basis(order, equations):
    domain = sp.QQ.frac_field(t)
    basis = groebner_basis(equations, (x, y), domain=domain, order=order, backend="native")
    reference = groebner_basis(equations, (x, y), domain=domain, order=order, backend="sympy")
    assert basis == reference
    assert all(basis.reduce(p)[1] == 0 for p in equations)


def test_multiple_parameters():
    domain = sp.QQ.frac_field(t, u)
    equations = [(t + u) * x - 1, y - t * x]
    basis = groebner_basis(equations, (x, y), domain=domain, backend="native")
    assert basis == groebner_basis(equations, (x, y), domain=domain, backend="sympy")
    quotient = QuotientAlgebra.from_polynomials(
        equations, (x, y), domain=domain, groebner_backend="native"
    )
    assert quotient.dimension == 1
    assert quotient.normal_form(x) == 1 / (t + u)


def test_native_images():
    ring = PolyRing((x,), sp.QQ.frac_field(t), "grevlex")
    result = native_field_basis(
        [ring.from_expr(t * x - 1)],
        ring,
        certify=True,
        limits=NativeF4Limits(collect_batch_metrics=True),
    )
    assert result.verified
    assert result.stats["native_calls"] >= 10
    assert len(result.primes) >= 2
    assert result.stats["images"]


def test_interpolation_budget():
    with pytest.raises(GroebnerLimitError, match="degree|point"):
        groebner_basis(
            [t * x - 1],
            (x,),
            domain=sp.QQ.frac_field(t),
            backend="native",
            limits=NativeF4Limits(max_parameter_degree=0, max_evaluation_points=3),
        )


def test_reconstruction_is_not_a_certificate(monkeypatch):
    import algroots.native_fields as fields
    from algroots import GroebnerVerificationError

    monkeypatch.setattr(
        fields, "_reconstruct", lambda signature, residues, modulus, model: (model.ring.one,)
    )
    monkeypatch.setattr(fields, "_fresh_check", lambda *args: True)
    with pytest.raises(GroebnerVerificationError, match="exact verification"):
        groebner_basis([t * x - 1], (x,), domain=sp.QQ.frac_field(t), backend="native")
