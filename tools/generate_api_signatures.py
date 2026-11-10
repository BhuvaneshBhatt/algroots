"""Generate public signatures and root exports from the source API."""

import argparse
import importlib
import inspect
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import algroots  # noqa: E402

REFERENCE = ROOT / "docs/api.md"
FUNCTIONS = {
    "VerifiedGroebnerFamily": "verified_families",
    "VerifiedGroebnerFamily.specialize": "verified_families",
    "VerifiedRationalUnivariateFamily": "verified_families",
    "VerifiedRationalUnivariateFamily.specialize_field": "verified_families",
    "GroebnerCertificate.prepare": "basis_certificates",
    "RationalUnivariateCertificate.prepare": "rur_certificates",
    "AlgebraicFunctionField.specializations": "algebraic_function_fields",
    "AlgebraicFunctionField.with_parameters": "algebraic_function_fields",
    "certify_groebner_basis": "basis_certificates",
    "certify_rational_univariate_representation": "rur_certificates",
    "AlgebraicFunctionField": "algebraic_function_fields",
    "AlgebraicFunctionField.extend": "algebraic_function_fields",
    "AlgebraicFunctionField.factor_univariate": "algebraic_function_fields",
    "AlgebraicFunctionField.specialize": "algebraic_function_fields",
    "QuotientAlgebra.from_certificate": "quotient",
    "parametric_groebner_basis": "parametric_groebner",
    "algsolve": "algebraic",
    "polysolve": "solver",
    "groebner_basis": "basis",
    "groebner_backends": "backends",
    "run_supervised": "execution",
    "algebraize_system": "algebraization",
    "recognize_system_roots": "recognition",
    "discover_monodromy_orbit": "monodromy",
    "compute_rational_univariate_representation": "rational_univariate",
    "solve_rur_representation": "rational_univariate",
    "solve_rur_points": "rational_univariate",
    "solve_zero_dimensional_system_with_rur": "rational_univariate",
    "compute_border_basis": "border_basis",
    "compute_border_basis_linear": "border_basis",
    "certify_isolated_root": "certification",
    "certify_root_box": "certification",
    "certify_numerical_roots": "certification",
    "deflate_isolated_root": "deflation",
    "cauchy_endgame": "endgames",
    "track_projective_path": "projective_tracking",
    "QuotientAlgebra.from_polynomials": "quotient",
    "QuotientAlgebra.from_groebner_basis": "quotient",
    "PrecisionOptions": "options",
    "SolverLimits": "options",
    "SolverOptions": "options",
    "ExecutionLimits": "execution",
}


def render_signature(name, module):
    """Strip annotations while retaining real parameter names and defaults."""
    value = importlib.import_module("algroots." + module)
    for component in name.split("."):
        value = getattr(value, component)
    signature = inspect.signature(value)
    signature = signature.replace(
        parameters=[
            p.replace(annotation=inspect.Parameter.empty) for p in signature.parameters.values()
        ],
        return_annotation=inspect.Signature.empty,
    )
    return f"{name}{signature}"


def render(text):
    """Update only marked signature blocks and the exported-name paragraph."""
    text = re.sub(
        r"<!-- api-exports -->.*?<!-- /api-exports -->",
        "<!-- api-exports -->\nThe root exports "
        + ", ".join(f"`{name}`" for name in algroots.__all__)
        + ".\n<!-- /api-exports -->",
        text,
        flags=re.S,
    )
    for name, module in FUNCTIONS.items():
        pattern = (
            r"(<!-- api-signature: "
            + re.escape(name)
            + r" -->\s*```(?:python|text)\n).*?(\n```\s*<!-- /api-signature -->)"
        )
        text, count = re.subn(
            pattern,
            lambda match, name=name, module=module: (
                match[1] + render_signature(name, module) + match[2]
            ),
            text,
            flags=re.S,
        )
        if count != 1:
            raise ValueError(f"expected one signature marker for {name}, found {count}")
    return text


def main():
    """Write the reference or fail if any generated signature is stale."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    previous = REFERENCE.read_text(encoding="utf-8")
    generated = render(previous)
    if args.check:
        if previous != generated:
            parser.exit(1, "docs/api.md is stale; run python tools/generate_api_signatures.py\n")
        print("Public signatures and exports are current")
    else:
        REFERENCE.write_text(generated, encoding="utf-8")


if __name__ == "__main__":
    main()
