"""Exact coordinate semantics across coefficient parsing and real RUR solving."""

import pytest
import sympy as sp

from algroots import QuotientAlgebra, certify_rational_univariate_representation, polysolve
from algroots.certification import certify_isolated_root
from algroots.rational_univariate import (
    RationalUnivariateRepresentation,
    compute_rational_univariate_representation,
    solve_rur_points,
    solve_rur_representation,
)


@pytest.mark.parametrize("api", [solve_rur_representation, solve_rur_points])
@pytest.mark.parametrize("equations", [0, 1, 2])
def test_real_coordinates(api, equations):
    x, y = sp.symbols("x y")
    systems = [
        ([x - sp.I, y + sp.I], ()),
        ([x - sp.I, (y + sp.I) ** 2 - 1], ()),
        ([x * (x - sp.I), y + (1 + sp.I) * x], ((0, 0),)),
    ]
    source, expected = systems[equations]
    representation = compute_rational_univariate_representation(source, (x, y))
    result = api(representation, real=True)
    coordinates = tuple(p.coordinates for p in result) if api is solve_rur_points else result
    assert coordinates == expected
    all_points = api(representation, real=False)
    assert len(all_points) == representation.solution_count


@pytest.mark.parametrize("api", [solve_rur_representation, solve_rur_points])
def test_complex_separator(api):
    x, t = sp.symbols("x t")
    field = sp.QQ.algebraic_field(sp.I)
    representation = RationalUnivariateRepresentation(
        (x,),
        t,
        sp.Poly(t**2 + 1, t, domain=field),
        sp.Poly(1, t, domain=field),
        (sp.Poly(-sp.I * t, t, domain=field),),
        sp.I * x,
        ((0,), (1,)),
        2,
        2,
    )
    result = api(representation, real=True)
    coordinates = tuple(p.coordinates for p in result) if api is solve_rur_points else result
    assert set(coordinates) == {(-1,), (1,)}


@pytest.mark.parametrize("explicit_domain", [False, True])
@pytest.mark.parametrize("algebraic_number", [False, True])
def test_bound_coefficient_quotient(explicit_domain, algebraic_number):
    x = sp.Symbol("x")
    alpha = sp.CRootOf(x**3 - x - 1, 0)
    coefficient = sp.AlgebraicNumber(alpha) if algebraic_number else alpha
    domain = sp.QQ.algebraic_field(alpha) if explicit_domain else None
    quotient = QuotientAlgebra.from_polynomials([x - coefficient], (x,), domain=domain)
    assert quotient.dimension == 1
    assert quotient.domain.convert(quotient.normal_form(x)) == quotient.domain.convert(alpha)
    assert quotient.normal_form(x - coefficient) == 0


@pytest.mark.parametrize("method", ["auto", "action", "rur", "shape", "triangular"])
@pytest.mark.parametrize("presolve", [True, False])
@pytest.mark.parametrize("backend", ["native", "sympy"])
def test_bound_coefficient_solver(method, presolve, backend):
    x = sp.Symbol("x")
    alpha = sp.CRootOf(x**3 - x - 1, 0)
    result = polysolve(
        [x**2 - alpha],
        (x,),
        method=method,
        presolve=presolve,
        recognize=False,
        groebner_backend=backend,
    )
    assert len(result.roots) == 2
    assert result.max_relative_residual < sp.Float("1e-30")
    for point in result.roots:
        assert abs(complex(sp.N(point[0] ** 2 - alpha, 45))) < 1e-30


def test_bound_coefficient_certificate():
    x = sp.Symbol("x")
    alpha = sp.CRootOf(x**3 - x - 1, 0)
    certificate = certify_isolated_root([x - alpha], (x,), (alpha,))
    assert certificate.point == (alpha,)
    assert certificate.multiplicity == certificate.quotient_dimension == 1
    assert certificate.verify()


@pytest.mark.parametrize("coefficient", [sp.sqrt(2), sp.I])
def test_number_field_rur_certificate(coefficient):
    x = sp.Symbol("x")
    field = sp.QQ.algebraic_field(coefficient)
    certificate = certify_rational_univariate_representation(
        [x**2 - coefficient], (x,), domain=field
    )
    assert certificate.verify()
    assert certificate.prepare().verified_representation().solution_count == 2


@pytest.mark.parametrize("complex_root", [False, True])
def test_rootof_ball_arithmetic(complex_root):
    from flint import acb_poly

    from algroots.numerical import flint_dps, sympy_to_acb

    x = sp.Symbol("x")
    root = sp.CRootOf(x**3 - x - 1, 1 if complex_root else 0)
    with flint_dps(100):
        references = acb_poly([-1, -1, 0, 1]).roots(tol="1e-85")
        reference = (
            next(r for r in references if r.imag < 0)
            if complex_root
            else next(r for r in references if r.imag.contains(0))
        )
        cases = [
            (root, reference),
            (root**2, reference**2),
            (1 / (1 + root), 1 / (1 + reference)),
            (sp.AlgebraicNumber(root), reference),
            (sp.sqrt(root), reference.sqrt()),
        ]
        for value, expected in cases:
            enclosure = sympy_to_acb(value, 70)
            assert enclosure.is_finite()
            assert enclosure.overlaps(expected)
            assert (enclosure - expected).contains(0)


@pytest.mark.parametrize("digits", [30, 70])
def test_rootof_homotopy(digits):
    from algroots._precision import working_precision
    from algroots.continuation import SympyHomotopy
    from algroots.numerical import sympy_to_mpc

    x, t = sp.symbols("x t")
    alpha = sp.CRootOf(x**3 - x - 1, 0)
    system = SympyHomotopy((x**2 - alpha * t,), (x,), t)
    with working_precision(digits):
        root = sympy_to_mpc(sp.sqrt(alpha), digits + 5)
        assert abs(system.evaluate((root,), 1)[0]) < 10 ** (-digits + 2)
        assert abs(system.jacobian((root,), 1)[0] - 2 * root) < 10 ** (-digits + 2)
        assert abs(
            system.parameter_derivative((root,), 1)[0] + sympy_to_mpc(alpha, digits)
        ) < 10 ** (-digits + 2)


def test_rootof_homotopy_solver():
    x = sp.Symbol("x")
    alpha = sp.CRootOf(x**3 - x - 1, 0)
    result = polysolve([x**2 - alpha], (x,), method="homotopy", presolve=False, recognize=False)
    assert len(result.roots) == 2
    assert result.max_relative_residual < sp.Float("1e-25")
    for point in result.roots:
        assert abs(complex(sp.N(point[0] ** 2 - alpha, 45))) < 1e-25
