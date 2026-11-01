"""Multivariate GRS02 decisions, field maps, and membership with exact oracles."""

from dataclasses import replace

import pytest
import sympy as sp

from algroots.errors import QuotientAlgebraError
from algroots.function_fields import (
    LurothGeneratorResult,
    RationalFunctionSubfield,
    coefficient_field,
    luroth_generator,
)

x, y, z, t, u, v, w = sp.symbols("x y z t u v w")


def assert_maps(result):
    """Check recorded identities directly, independently of the gcd verifier."""
    assert result.verify()
    if not result.rational:
        return
    for original, image in zip(result.generators, result.coefficient_images, strict=True):
        assert sp.cancel(image.subs(t, result.generator) - original) == 0
    inverse = result.generator_expression.xreplace(
        dict(zip(result.generator_symbols, result.generators, strict=True))
    )
    assert sp.cancel(inverse - result.generator) == 0


@pytest.mark.parametrize("h", [x + y, x / y, (x * y + 1) / (x - y)])
def test_coprime_powers_recover_the_full_generated_field(h):
    result = luroth_generator([h**2, h**3], variables=(x, y), parameter=t, generator_symbols=(u, v))
    assert isinstance(result, LurothGeneratorResult)
    assert result.rational and result.status == "rational"
    assert result.transcendence_degree == result.transcendence_degree_lower_bound == 1
    assert_maps(result)
    assert result.contains(h) and not result.contains(x)
    assert result.express(h).verify() and result.express(x).verify()
    assert result.express(1 / (h + 1)).verify()
    assert not hasattr(result, "extension_degree")
    assert not hasattr(result, "minimal_ambient_polynomial")
    assert result.genus == 0
    parametrization = result.rational_parametrization()
    assert parametrization.verify()
    assert parametrization.to_ambient(parametrization.to_source(h)) == sp.cancel(h)
    assert parametrization.inverse_guard == sp.Ne(sp.denom(result.generator_expression), 0)


@pytest.mark.parametrize("h", [x + y, x / y])
def test_proper_subfield_does_not_silently_adjoin_an_algebraic_element(h):
    result = luroth_generator([h**2, h**4], variables=(x, y), parameter=t)
    assert_maps(result)
    assert sp.cancel(result.generator - h**2) == 0
    assert result.contains(h**2) and not result.contains(h)
    assert result.express(h).verify()
    assert result.express(1 / (h**2 + 1)).expression == 1 / (t + 1)
    with pytest.raises(QuotientAlgebraError, match="does not belong"):
        result.rational_parametrization().to_source(h)


def test_grs02_example_3_1_in_three_variables():
    # Expanded numerators/denominators in Example 3.1 equal these expressions.
    a, b = y * (x**2 - z), x - z
    f1 = (a**2 + b**2) / (a * b)
    f2 = a**2 / (b**2 + a * b)
    result = luroth_generator([f1, f2], variables=(x, y, z), parameter=t, generator_symbols=(u, v))
    assert_maps(result)
    h = a / b
    assert result.contains(h) and not result.contains(x)
    # The published generator and the returned one differ at most by a Möbius map.
    coordinate = result.express(h).expression
    numerator, denominator = sp.fraction(coordinate)
    assert max(sp.degree(numerator, t), sp.degree(denominator, t)) == 1
    assert result.express(h).verify()


def test_three_inputs_are_needed_to_generate_the_full_rational_field():
    h = x + y
    # Every pair of these powers generates a proper subfield, but gcd(6,10,15)=1.
    result = luroth_generator(
        [h**6, h**10, h**15], variables=(x, y), parameter=t, generator_symbols=(u, v, w)
    )
    assert_maps(result)
    assert result.contains(h) and result.express(h).verify()


@pytest.mark.parametrize("proper", [False, True])
def test_nonpower_outer_functions_and_proper_invariant_subfield(proper):
    h = (x * y + z) / (y + 1)
    values = [h + 1 / h, h**2 + 1 / h**2] if proper else [h**2 + h, h**2 - h]
    result = luroth_generator(values, variables=(x, y, z), parameter=t)
    assert_maps(result)
    assert result.contains(h) == (not proper)
    assert result.express(h).verify()


@pytest.mark.parametrize("values", [[], [0, 2, sp.Rational(2, 3)]])
def test_constant_field_has_no_spurious_transcendental_generator(values):
    result = luroth_generator(values, variables=(x, y), parameter=t)
    assert_maps(result)
    assert result.status == "constant" and not result.rational
    assert result.transcendence_degree == result.transcendence_degree_lower_bound == 0
    assert result.gcd_polynomial is result.generator is result.generator_expression is None
    assert result.contains(3) and not result.contains(x / y)
    assert result.express(x).verify() and result.express(3).verify()
    assert result.to_ambient(sp.Rational(2, 3)) == sp.Rational(2, 3)
    with pytest.raises(QuotientAlgebraError):
        result.to_ambient(t)
    with pytest.raises(QuotientAlgebraError, match="transcendence degree one"):
        result.rational_parametrization()
    with pytest.raises(QuotientAlgebraError):
        _ = result.genus


@pytest.mark.parametrize("values", [[x, y], [x, y, z], [x * x, x * y], [x / y, y / z]])
def test_unit_gcd_certifies_only_a_lower_bound_not_an_exact_dimension(values):
    result = luroth_generator(values, variables=(x, y, z), parameter=t)
    assert_maps(result)
    assert result.status == "higher_transcendence_degree" and not result.rational
    assert result.transcendence_degree is None
    assert result.transcendence_degree_lower_bound == 2
    assert result.gcd_polynomial.as_expr() == 1
    assert result.generator is result.generator_expression is result.coefficient_images is None
    # Independent characteristic-zero oracle: the Jacobian rank is at least two.
    jacobian = sp.Matrix(values).jacobian((x, y, z))
    assert jacobian.rank() >= 2
    for operation in (lambda: result.contains(x), lambda: result.to_ambient(t)):
        with pytest.raises(QuotientAlgebraError, match="greater than one"):
            operation()


def test_number_field_constants_and_nonmonic_rational_poles():
    a = sp.sqrt(2)
    constants = sp.QQ.algebraic_field(a)
    h = (a * x + y) / (x - 2 * y)
    result = luroth_generator([h**2, h**3, a], variables=(x, y), constants=constants, parameter=t)
    assert_maps(result)
    assert result.constants == constants and result.contains(h)
    assert result.express(a).expression == a and result.express(a).verify()
    assert result.express(1 / (h + a)).verify()
    for query in (a * h + 2 / h, 1 / (h**2 + a), (h**2 + a) / (a * h - 1)):
        assert result.contains(query) and result.express(query).verify()
    assert not result.contains(x)
    constant = luroth_generator([a], variables=(x, y), constants=constants, parameter=t)
    assert constant.verify() and constant.status == "constant" and constant.contains(a)


def test_constant_and_redundant_inputs_cancellation_and_unused_variables():
    h = (x + y) / (x - y)
    values = [2, h**2, h**3, 0, h**2, (x**2 - y**2) / (x - y)]
    # The last input is x+y and makes this field have transcendence degree two.
    assert luroth_generator(values, variables=(x, y, z)).status == "higher_transcendence_degree"
    result = luroth_generator(values[:-1], variables=(x, y, z), parameter=t)
    assert_maps(result)
    assert result.contains(h) and not result.contains(z)
    polynomial = luroth_generator([(x**2 - y**2) / (x - y)], variables=(x, y, z), parameter=t)
    assert polynomial.generator == x + y
    assert polynomial.coefficient_images == (t,) and polynomial.verify()


@pytest.mark.parametrize("variables", [(x, y), (y, x), (z, y, x)])
def test_variable_order_and_birational_ambient_changes_preserve_the_field(variables):
    h = (x + y) / (x - y)
    result = luroth_generator([h**2, h**3], variables=variables, parameter=t)
    assert_maps(result)
    assert result.contains(h) and not result.contains(x)
    transformed = h.xreplace({x: x / (y + 1), y: y + 1})
    changed = luroth_generator([transformed**2, transformed**3], variables=variables, parameter=t)
    assert_maps(changed)
    assert changed.contains(transformed) and not changed.contains(x)


def test_coefficient_field_dispatch_and_legacy_one_variable_result():
    multivariate = coefficient_field(
        [(x / y) ** 2, (x / y) ** 3], ambient=sp.QQ.frac_field(x, y), parameter=t
    )
    assert isinstance(multivariate, LurothGeneratorResult) and multivariate.verify()
    assert multivariate.contains(x / y) and not multivariate.contains(x)
    constant = coefficient_field([], ambient=sp.QQ.frac_field(x, y))
    assert constant.status == "constant" and constant.verify()
    with pytest.raises(QuotientAlgebraError, match="inspect luroth_generator"):
        coefficient_field([x, y], ambient=sp.QQ.frac_field(x, y))
    univariate_field = coefficient_field([x**2], ambient=sp.QQ.frac_field(x), parameter=t)
    assert (
        isinstance(univariate_field, RationalFunctionSubfield)
        and univariate_field.extension_degree == 2
    )
    direct = luroth_generator([x**2], variables=(x,), parameter=t)
    assert direct.verify() and not direct.contains(x) and direct.contains(x**2)
    assert not hasattr(direct, "extension_degree")


def test_generator_inputs_are_consumed_once_and_polys_are_supported():
    result = luroth_generator(
        (sp.Poly(f, x, y) for f in ((x + y) ** 2, (x + y) ** 3)),
        variables=(s for s in (x, y)),
        parameter=t,
    )
    assert_maps(result)
    assert result.contains(x + y)


@pytest.mark.parametrize(
    "options",
    [
        {"variables": ()},
        {"variables": (x, x)},
        {"variables": (x, 1)},
        {"variables": (x, sp.Symbol("n", commutative=False))},
        {"variables": (x, y), "parameter": x},
        {"variables": (x, y), "parameter": x + 1},
        {"variables": (x, y), "generator_symbols": (x,)},
        {"variables": (x, y), "parameter": t, "generator_symbols": (t,)},
        {"variables": (x, y), "generator_symbols": (u, v)},
        {"variables": (x, y), "generator_symbols": (sp.Symbol("n", commutative=False),)},
        {"variables": (x, y), "constants": sp.GF(7)},
        {"variables": (x, y), "constants": sp.EX},
        {"variables": (x, y), "constants": sp.QQ.frac_field(z)},
    ],
)
def test_invalid_domain_and_coordinate_contracts(options):
    with pytest.raises(QuotientAlgebraError):
        luroth_generator([x + y], **options)


@pytest.mark.parametrize("value", [x + z, x + sp.sqrt(2), sp.sin(x), x + 0.5, sp.oo, sp.zoo])
def test_nonrational_inexact_and_undeclared_inputs_are_rejected(value):
    with pytest.raises(QuotientAlgebraError):
        luroth_generator([value], variables=(x, y))


def test_query_domain_and_map_contracts():
    result = luroth_generator([x / y], variables=(x, y), parameter=t)
    for invalid in (z, x + 0.5, sp.sqrt(2), sp.sin(x)):
        with pytest.raises(QuotientAlgebraError):
            result.express(invalid)
    for invalid in (x, u, t + 0.5, sp.sin(t)):
        with pytest.raises(QuotientAlgebraError):
            result.to_ambient(invalid)
    assert result.express(0).expression == 0
    assert result.express((x**2 - y**2) / (x * y - y**2)).expression == t + 1


def test_certificate_replay_rejects_altered_maps_and_decisions():
    result = luroth_generator(
        [(x / y) ** 2, (x / y) ** 3], variables=(x, y), parameter=t, generator_symbols=(u, v)
    )
    for changes in (
        {"generator": x},
        {"generator": 2},
        {"generator_expression": u},
        {"generator_expression": 1 / (v**2 - u**3)},
        {"coefficient_images": (t**2, t**3 + 1)},
        {"coefficient_images": (t**2,)},
        {"coefficient_images": (t**2, z)},
        {"gcd_polynomial": result.gcd_polynomial * 2},
        {"status": "constant"},
        {"status": "unknown"},
        {"comparison_symbols": (x, z)},
        {"comparison_symbols": (z, z)},
        {"generator_symbols": (u, u)},
        {"parameter": x},
        {"constants": sp.QQ.algebraic_field(sp.sqrt(2))},
    ):
        assert not replace(result, **changes).verify(), changes
    member = result.express(x / y)
    assert not replace(member, expression=t + 1).verify()
    assert not replace(result.express(x), expression=t).verify()
    constant = luroth_generator([2], variables=(x, y), parameter=t)
    assert not replace(constant, generator=x).verify()
    negative = luroth_generator([x, y], variables=(x, y), parameter=t)
    assert not replace(negative, status="rational", generator=x).verify()
    assert not replace(negative, coefficient_images=()).verify()


def test_integer_and_gaussian_constant_domains_are_normalized():
    integer = luroth_generator([x + y], variables=(x, y), constants=sp.ZZ)
    assert integer.constants == sp.QQ and integer.verify()
    gaussian = luroth_generator([sp.I * x / y], variables=(x, y), constants=sp.QQ_I)
    assert gaussian.constants == sp.QQ.algebraic_field(sp.I)
    assert gaussian.verify() and gaussian.contains(x / y)
