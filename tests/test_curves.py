"""Normalized genus and ground-field rationality are separate exact claims."""

from dataclasses import replace

import pytest
import sympy as sp

from algroots.curves import CurveComputationLimitError, CurveError, PlaneCurve
from algroots.function_fields import coefficient_field
from algroots.quotient import QuotientAlgebra

x, y, t, u, v, w = sp.symbols("x y t u v w")


@pytest.mark.parametrize(
    "f,genus",
    [
        (y * y - x**3, 0),
        (y * y - x**3 + x, 1),
        (x**4 + y**4 - 1, 3),
        (y**3 - x**4, 0),
        (y * y - x**5, 0),
        (y * y - x**6 + x, 2),
        (y * y - (x * x - 2) ** 2 * (x + 3), 0),
        (y * y - (x**3 - 2) ** 2 * (x + 1), 0),
        (y * y - (x * x + 1) ** 2 * (x**3 - x), 1),
    ],
)
def test_genus_of_projective_normalization(f, genus):
    C = PlaneCurve.from_polynomial(f, (x, y))
    N = C.normalization()
    assert N.genus == genus and N.verify()
    assert N.genus == N.arithmetic_genus - N.delta_sum == len(N.canonical_adjoints)
    assert not replace(N, genus=genus + 1).verify()


def test_infinitely_near_points_and_conjugate_singularities_are_counted():
    N = PlaneCurve.from_polynomial(y * y - x**5, (x, y)).normalization()
    assert N.arithmetic_genus == 6 and N.delta_sum == 6
    assert any(p.depth > 0 for p in N.singularities)
    assert {p.chart for p in N.singularities} == {"affine", "infinity_y"}
    conjugate = PlaneCurve.from_polynomial(
        y * y - (x * x - 2) ** 2 * (x + 3), (x, y)
    ).normalization()
    assert any(p.residue_degree == 2 and p.chart == "affine" for p in conjugate.singularities)
    bad = replace(N.singularities[0], multiplicity=3)
    assert not replace(N, singularities=(bad, *N.singularities[1:])).verify()


def test_cusp_birational_maps_and_exceptional_charts():
    R = PlaneCurve.from_polynomial(y * y - x**3, (x, y)).rational_parametrization(t)
    assert R.rational and R.verify()
    P = R.parametrization
    assert P.affine_coordinates == (t * t, t**3) and P.inverse == y / x
    assert P.exceptional_parameter_polynomial == sp.Poly(t, t, domain=sp.QQ)
    assert P.point_at_infinity == (0, 1, 0)
    a, b = sp.symbols("a b")
    assert P.homogeneous_coordinates(a, b) == (a * a * b, a**3, b**3)
    assert not replace(P, projective_coordinates=(t**4, t**6, sp.Integer(1))).verify()
    assert not replace(P, inverse=x).verify()


@pytest.mark.parametrize("f", [x * x + y * y - 1, x * y - 1, y - x * x, x + y + 1, x - 2])
def test_lines_and_conics_have_verified_maps(f):
    R = PlaneCurve.from_polynomial(f, (x, y)).rational_parametrization(t)
    assert R.rational and R.verify()
    assert R.parametrization.verify()


def test_circle_parameter_infinity_retains_omitted_affine_point():
    R = PlaneCurve.from_polynomial(x * x + y * y - 1, (x, y)).rational_parametrization(t)
    P = R.parametrization
    assert P.point_at_infinity == (0, -1, 1)
    assert P.inverse_guard == sp.Ne(x, 0)
    assert sp.expand(P.affine_coordinates[0] ** 2 + P.affine_coordinates[1] ** 2 - 1).cancel() == 0
    C = PlaneCurve.from_polynomial(y * y - x**3, (x, y))
    assert not replace(C.rational_parametrization(t), parametrization=P).verify()


def test_positive_genus_rejection_is_not_a_point_search_failure():
    R = PlaneCurve.from_polynomial(y * y - x**3 + x, (x, y)).rational_parametrization(t)
    assert (
        R.status == "positive_genus" and not R.rational and R.parametrization is None and R.verify()
    )


def test_genus_zero_without_qq_point_has_explicit_quadratic_extension():
    C = PlaneCurve.from_polynomial(x * x + y * y - 3, (x, y))
    R = C.rational_parametrization(t)
    assert C.genus == 0 and R.status == "no_ground_field_point" and not R.rational
    assert R.parametrization is None and R.splitting_field == sp.QQ.algebraic_field(sp.sqrt(3))
    assert R.verify()
    extended = C.rational_parametrization(t, extend_constants=True)
    assert (
        extended.status == "rational_over_extension" and not extended.rational and extended.verify()
    )
    assert extended.parametrization.domain == extended.splitting_field
    empty_real = PlaneCurve.from_polynomial(x * x + y * y + 1, (x, y)).rational_parametrization(t)
    assert empty_real.status == "no_ground_field_point" and empty_real.verify()
    assert empty_real.splitting_field.ext.as_expr().is_real is False


@pytest.mark.parametrize("constant,status", [(2, "rational"), (3, "no_ground_field_point")])
def test_singular_quartic_descends_to_conic_before_point_decision(constant, status):
    C = PlaneCurve.from_polynomial(x * x + (y - x * x) ** 2 - constant, (x, y))
    R = C.rational_parametrization(t)
    assert C.genus == 0 and R.status == status and R.conic_model.verify() and R.verify()
    bad = replace(R.conic_model, inverse=(sp.Integer(0), sp.Integer(0)))
    assert not bad.verify()
    if constant == 3:
        extended = C.rational_parametrization(t, extend_constants=True)
        assert extended.status == "rational_over_extension" and extended.verify()


def test_number_field_extension_witness():
    K = sp.QQ.algebraic_field(sp.sqrt(2))
    C = PlaneCurve.from_polynomial(x * x + y * y - 2, (x, y), domain=K)
    R = C.rational_parametrization(t, point=(sp.sqrt(2), 0))
    assert R.rational and R.verify()
    with pytest.raises(CurveError, match="automatic conic-point"):
        C.rational_parametrization(t)
    D = PlaneCurve.from_polynomial(x * x + y * y - sp.sqrt(2), (x, y), domain=K)
    extension = D.rational_parametrization(t, extend_constants=True)
    assert extension.status == "rational_over_extension" and extension.verify()
    assert extension.splitting_field.mod.degree() == 4


def test_field_view_preserves_exceptional_original_fibres():
    C = PlaneCurve.from_polynomial(x * y * y + y + 1, (x, y))
    A = C.function_field()
    B = A.specialize({x: 0})
    assert B.dimension == 1 and B.normal_form(y) == -1


def test_coefficient_plane_model_and_generator_embedding_replay():
    A = QuotientAlgebra.from_polynomials([y * y - x], (y,), domain=sp.QQ.frac_field(x))
    E = coefficient_field([y * y, y**3, y * y + y**3], ambient=A, generator_symbols=(u, v, w))
    assert E.genus == 0 and E.curve_model().verify()
    P = E.rational_parametrization(t)
    assert P.verify() and A.normal_form(P.generator - y) == 0
    assert P.coefficient_images == (t * t, t**3, t * t + t**3)
    assert P.to_source(y + 1) == t + 1 and P.to_ambient(t + 1) == y + 1
    assert not replace(P.model, generator_images=(sp.Integer(0),) * 3).verify()


def test_coefficient_fields_preserve_genus_and_point_obstructions():
    for relation, status, genus in [
        (y * y - x**3 + x, "positive_genus", 1),
        (y * y + x * x - 3, "no_ground_field_point", 0),
    ]:
        A = QuotientAlgebra.from_polynomials([relation], (y,), domain=sp.QQ.frac_field(x))
        E = coefficient_field([x, y], ambient=A, generator_symbols=(u, v))
        assert E.genus == genus and E.rationality(t).status == status
        with pytest.raises(CurveError, match=status):
            E.rational_parametrization(t)
    C = coefficient_field([1], ambient=A)
    with pytest.raises(CurveError, match="transcendence degree"):
        C.curve_model()


@pytest.mark.parametrize("f", [x * x - 2 * y * y, x * x + y * y])
def test_geometric_reducibility_is_not_silently_given_a_genus(f):
    C = PlaneCurve.from_polynomial(f, (x, y))
    with pytest.raises(CurveError, match="geometrically integral"):
        C.normalization()


@pytest.mark.parametrize(
    "f", [sp.Integer(0), sp.Integer(1), x * y, (x + y) ** 2, 1 / x + y, x + sp.Float(0.5)]
)
def test_invalid_curve_equations(f):
    with pytest.raises(ValueError):
        PlaneCurve.from_polynomial(f, (x, y))


def test_explicit_limits_and_invalid_map_requests():
    C = PlaneCurve.from_polynomial(y * y - x**5, (x, y))
    with pytest.raises(CurveComputationLimitError, match="max_steps"):
        C.normalization(max_steps=1)
    with pytest.raises(CurveComputationLimitError, match="singular locus"):
        C.normalization(max_dimension=1)
    for bad in (0, -1, True):
        with pytest.raises(CurveError):
            C.normalization(max_steps=bad)
    with pytest.raises(CurveError, match="fresh Symbol"):
        C.rational_parametrization(x)
    with pytest.raises(CurveError, match="point"):
        PlaneCurve.from_polynomial(y * y - x**3, (x, y)).rational_parametrization(t, point=(0, 0))
    with pytest.raises(CurveError, match="not on"):
        PlaneCurve.from_polynomial(x * x + y * y - 1, (x, y)).rational_parametrization(
            t, point=(2, 0)
        )


def test_proper_coefficient_field_does_not_adjoin_the_ambient_parameter():
    A = QuotientAlgebra.from_polynomials([y * y - x * y + x**3], (y,), domain=sp.QQ.frac_field(x))
    E = coefficient_field([y], ambient=A, generator_symbols=(u,))
    P = E.rational_parametrization(t)
    assert E.genus == 0 and P.verify() and P.generator == y
    with pytest.raises(CurveError, match="does not belong"):
        P.to_source(x)


def test_genus_under_coordinate_changes_and_relative_number_constants():
    transformed = PlaneCurve.from_polynomial((y + x) ** 2 - (x - 2 * y + 1) ** 3, (x, y))
    assert transformed.genus == 0 and transformed.rational_parametrization(t).verify()
    K = sp.QQ.algebraic_field(sp.sqrt(2))
    curve = PlaneCurve.from_polynomial(
        y * y - (x * x - sp.sqrt(2)) ** 2 * (x + 1), (x, y), domain=K
    )
    normalization = curve.normalization()
    assert normalization.genus == 0 and normalization.verify()
    assert any(p.residue_degree == 2 for p in normalization.singularities)


def test_integral_over_qq_can_split_geometrically():
    curve = PlaneCurve.from_polynomial(x**2 + y**2, (x, y), domain=sp.QQ)
    assert curve.polynomial == sp.Poly(x**2 + y**2, x, y, domain=sp.QQ).monic()
    with pytest.raises(CurveError, match="geometrically integral"):
        curve.normalization()
    with pytest.raises(CurveError, match="geometrically integral"):
        _ = curve.genus
