"""Check mathematical contracts across an installed six-package release."""

import argparse
import importlib
import importlib.metadata
import json
import pickle
import sys
import tomllib
from pathlib import Path

import sympy as sp


def main():
    """Exercise exact consumers and reject source-tree or version contamination."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--allow-source", action="store_true")
    parser.add_argument(
        "--versions",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "ci/release-stack.toml",
    )
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    versions = tomllib.loads(args.versions.read_text())["versions"]
    origins = {}
    for name, expected in versions.items():
        module = importlib.import_module(name)
        actual = importlib.metadata.version(name)
        if actual != expected:
            raise RuntimeError(f"{name}: expected {expected}, installed {actual}")
        path = Path(module.__file__).resolve()
        if not args.allow_source and not path.is_relative_to(Path(sys.prefix).resolve()):
            raise RuntimeError(f"{name} was imported outside the installed environment: {path}")
        origins[name] = str(path)
    checks = []

    def require(condition, name):
        if not condition:
            raise ArithmeticError(name)
        checks.append(name)

    import funcprops as fp
    import symbopt as so
    from alggeom.ideals import PolynomialRing
    from probstats.algebraic import AlgebraicModel, critical_points
    from probstats.bayes.certification import certify_zero
    from semialg.algebraic_decomposition import (
        primary_decomposition,
        verify_primary_decomposition_certificate,
    )
    from semialg.algebraic_function_fields import MonogenicFunctionField, RationalFunctionField

    from algroots import (
        AlgebraicFunctionField,
        GroebnerBackendUnavailable,
        GroebnerLimitError,
        NativeF4Limits,
        QuotientAlgebra,
        certify_groebner_basis,
        certify_rational_univariate_representation,
        groebner_basis,
    )
    from algroots.native_f4 import compiled_available

    require(compiled_available() == args.native, "native profile")
    backend = "native" if args.native else "sympy"
    x, y, t, a, b, z = sp.symbols("x y t a b z")
    equations = [x * x - y, y * y - 1]
    for domain in [sp.QQ, sp.GF(101)]:
        basis = groebner_basis(equations, (x, y), domain=domain, backend=backend)
        reference = sp.groebner(equations, x, y, domain=domain, order="grevlex")
        require(basis == reference, "verified basis " + str(domain))
        ideal = PolynomialRing((x, y), domain).ideal(*equations, groebner_backend=backend)
        require(ideal.normal_form(x**4 - 1) == 0, "alggeom routing " + str(domain))
    quotient = QuotientAlgebra.from_polynomials([x**4, y**4], (x, y), groebner_backend=backend)
    require(quotient.trace_pairing.todok() == {(0, 0): 16}, "sparse quotient trace")
    require(
        quotient.characteristic_polynomial(x + y, z) == sp.Poly(z**16, z, domain=sp.QQ),
        "noncyclic characteristic multiplicity",
    )
    family = certify_rational_univariate_representation(
        [x * x - t, y - x], (x, y), parameters=(t,), backend=backend
    ).prepare()
    require(family.specialize({t: 0}).solution_count == 1, "exceptional RUR fiber")
    require(
        pickle.loads(pickle.dumps(family)).specialize({t: 2}).solution_count == 2,
        "prepared persistence",
    )
    field = AlgebraicFunctionField(sp.Poly(a * a - t, a, domain=sp.QQ.frac_field(t)))
    tower = field.extend(sp.Poly(b * b - a - 1, b, domain=field))
    proof = certify_groebner_basis(
        [x * x - b, y - x], (x, y), domain=tower, backend=backend
    ).prepare()
    require(QuotientAlgebra.from_certificate(proof).normal_form(x * x - b) == 0, "tower quotient")
    require(
        field.specialize({t: 1}, roots={a: 1})(field.convert(1 / (a + 1))) == sp.Rational(1, 2),
        "local removable pole",
    )
    require(len(tower.specializations({t: 4})) == 4, "explicit tower branches")
    sem_field = MonogenicFunctionField(RationalFunctionField((t,)), a, (-t, 0, 1))
    primary = primary_decomposition(
        [(x - a) ** 2 * (x + a) ** 3, y * y],
        (x, y),
        coefficient_field=sem_field,
        groebner_backend=backend,
    )
    require(
        sorted(c.degree for c in primary.components) == [4, 6], "semialg primary multiplicities"
    )
    require(verify_primary_decomposition_certificate(primary.certificate), "semialg primary replay")
    xr, yr = sp.symbols("xr yr", real=True)
    require(
        sp.simplify_logic(sp.Xor(fp.domain(sp.sqrt(xr - 1), xr), xr >= 1)) is sp.false,
        "funcprops domain",
    )
    require(
        sp.simplify(sp.Equivalent(fp.range(xr * xr, xr, yr), yr >= 0)) is sp.true, "funcprops image"
    )
    require(
        so.minimize((xr - 2) ** 2, variables=[xr], use_sos=False) == [0, {xr: 2}],
        "symbopt exact optimum",
    )
    require(
        so.minimize(xr, [xr > 0], variables=[xr], use_sos=False) == [0, {xr: 0}],
        "symbopt unattained infimum",
    )
    p = sp.symbols("p0:3")
    unavailable_checks = []
    try:
        critical = critical_points(AlgebraicModel(p), (2, 3, 5))
    except GroebnerBackendUnavailable:
        if args.native:
            raise
        unavailable_checks.append("probstats likelihood uses default native geometry")
        critical = None
    if critical is not None:
        require(
            critical.points
            == ({p[0]: sp.Rational(1, 5), p[1]: sp.Rational(3, 10), p[2]: sp.Rational(1, 2)},),
            "probstats exact likelihood",
        )
    require(certify_zero(sp.sin(xr)).is_zero is None, "unknown predicate preserved")
    if args.native:
        try:
            groebner_basis(
                [x * y - 1, y * y - x],
                (x, y),
                backend="native",
                limits=NativeF4Limits(max_basis=1),
            )
        except GroebnerLimitError:
            checks.append("native budget error")
        else:
            raise ArithmeticError("native budget was not enforced")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(
            {
                "versions": versions,
                "native": args.native,
                "origins": origins,
                "checks": checks,
                "unavailable_checks": unavailable_checks,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Verified {len(checks)} installed cross-package mathematical contracts")


if __name__ == "__main__":
    main()
