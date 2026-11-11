"""Validate installed wheel arithmetic and explicit native availability."""

import argparse
import importlib
import subprocess
import sys
from importlib.metadata import distribution
from pathlib import Path

import sympy as sp

from algroots import GroebnerBackendUnavailable, QuotientAlgebra, groebner_basis
from algroots.native_f4 import compiled_available


def main():
    """Check the chosen binary profile and verified QQ/GF(p) construction."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--tutorials", action="store_true")
    parser.add_argument("--extension-fields", action="store_true")
    parser.add_argument("--stable-abi", action="store_true")
    args = parser.parse_args()
    if args.stable_abi:
        if not args.native:
            parser.error("--stable-abi requires --native")
        metadata = distribution("algroots").read_text("WHEEL")
        tags = [
            line.split(": ", 1)[1] for line in metadata.splitlines() if line.startswith("Tag: ")
        ]
        if not tags or any(not tag.startswith("cp311-abi3-") for tag in tags):
            raise RuntimeError("installed native wheel does not declare cp311-abi3")
        for name in ("_native_f4", "_native_extension", "_ff_sparse"):
            module = importlib.import_module("algroots." + name)
            if not module.__file__.endswith((".abi3.so", ".pyd")):
                raise RuntimeError(f"{name} did not load a Stable-ABI extension")
        print("Stable ABI modules loaded on Python", sys.version.split()[0])
    if compiled_available() != args.native:
        raise RuntimeError("installed kernel availability differs from the build profile")
    x, y = sp.symbols("x y")
    equations = (x**2 - y, y**2 - 1)
    backend = "native" if args.native else "sympy"
    for domain in (sp.QQ, sp.GF(101)):
        result = groebner_basis(equations, (x, y), domain=domain, backend=backend)
        if result != sp.groebner(equations, x, y, domain=domain, order="grevlex"):
            raise ArithmeticError("installed wheel returned the wrong basis")
        quotient = QuotientAlgebra.from_polynomials(
            equations, (x, y), domain=domain, groebner_backend=backend
        )
        if quotient.dimension != 4 or quotient.normal_form(x**4 - 1) != 0:
            raise ArithmeticError("installed wheel returned the wrong quotient")
    if not args.native:
        try:
            groebner_basis(equations, (x, y), backend="native")
        except GroebnerBackendUnavailable:
            pass
        else:
            raise RuntimeError("fallback wheel accepted a missing native kernel")
    else:
        t = sp.Symbol("t")
        basis = groebner_basis([t * x - 1], (x,), domain=sp.QQ.frac_field(t), backend="native")
        if basis.reduce(t * x - 1)[1] != 0:
            raise ArithmeticError("installed native parameter-field result is incorrect")
    if args.extension_fields:
        from algroots import parametric_groebner_basis
        from algroots.backends import _extension_available

        if not _extension_available():
            raise RuntimeError("FLINT native extension was requested but is unavailable")
        t = sp.Symbol("t")
        for domain in (
            sp.QQ.algebraic_field(sp.sqrt(2)),
            sp.QQ.frac_field(t),
            sp.QQ.algebraic_field(sp.sqrt(2)).frac_field(t),
        ):
            coefficient = (
                sp.sqrt(2)
                if domain.is_AlgebraicField
                else t + sp.sqrt(2)
                if domain.domain.is_AlgebraicField
                else t
            )
            equations = (x * x - coefficient, y - x)
            result = groebner_basis(equations, (x, y), domain=domain, backend="native")
            reference = groebner_basis(equations, (x, y), domain=domain, backend="sympy")
            if result != reference:
                raise ArithmeticError(
                    "installed coefficient-field basis disagrees with the reference"
                )
        from algroots import NativeF4Limits
        from algroots.rational_univariate import compute_rational_univariate_representation

        u = sp.Symbol("u")
        field = sp.QQ.frac_field(t, u)
        sparse = groebner_basis(
            [(t - u) * x - (t * t + u)],
            (x,),
            domain=field,
            backend="native",
            limits=NativeF4Limits(parameter_interpolation="sparse"),
        )
        if sparse.reduce((t - u) * x - (t * t + u))[1] != 0:
            raise ArithmeticError("installed sparse rational interpolation failed")
        mixed = sp.QQ.algebraic_field(sp.sqrt(2)).frac_field(t)
        rur = compute_rational_univariate_representation(
            [x * x - t - sp.sqrt(2), y - x], (x, y), domain=mixed, groebner_backend="native"
        )
        coordinates = rur.normalized_coordinate_polynomials()
        identity = sp.Poly(
            coordinates[0].as_expr() ** 2 - t - sp.sqrt(2), rur.parameter, domain=mixed
        )
        if not identity.rem(rur.defining_polynomial).is_zero:
            raise ArithmeticError("installed mixed-field RUR failed")
        proof = parametric_groebner_basis([t * x], (x,), (t,), backend="native")
        if not proof.verify() or list(proof.specialize({t: 0})) != []:
            raise ArithmeticError("installed specialization lost an exceptional fiber")
    from algroots import AlgebraicFunctionField, certify_groebner_basis

    t, a, b = sp.symbols("t a b")
    field = AlgebraicFunctionField(sp.Poly(a * a - t, a, domain=sp.QQ.frac_field(t)))
    tower = field.extend(sp.Poly(b * b - a - 1, b, domain=field))
    prepared = certify_groebner_basis(
        [x * x - b, y - x], (x, y), domain=tower, backend=backend
    ).prepare()
    quotient = QuotientAlgebra.from_certificate(prepared)
    if quotient.dimension != 2 or quotient.normal_form(x * x - b) != 0:
        raise ArithmeticError("installed tower quotient or prepared certificate failed")
    embedding = field.specialize({t: 1}, roots={a: 1})
    if embedding(field.convert(1 / (a + 1))) != sp.Rational(1, 2):
        raise ArithmeticError("installed local specialization failed")
    if args.extension_fields:
        candidate = groebner_basis([x * x - b, y - x], (x, y), domain=tower, backend="native")
        if candidate != groebner_basis(
            [x * x - b, y - x], (x, y), domain=tower, certificate=prepared, backend="sympy"
        ):
            raise ArithmeticError("installed native tower basis failed")
    print("Installed wheel verified:", backend)
    if args.tutorials:
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("check_installed_examples.py")),
                "--optional-backends",
                *([] if args.native else ["--portable"]),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
