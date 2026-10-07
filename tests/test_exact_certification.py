from types import SimpleNamespace

import pytest
import sympy as sp

from algroots import PolynomialSystemRoots
from algroots.errors import ExactCertificationError
from algroots.recognition import _recognize_exact_roots
from algroots.solver import polysolve


class _IntegerPolynomial:
    def __init__(self, coefficients):
        self._coefficients = tuple(coefficients)

    def coeffs(self):
        return self._coefficients


def _recognized(coefficients, *, certified=True):
    return SimpleNamespace(
        polynomial=_IntegerPolynomial(coefficients),
        certified=certified,
    )


def _recognize_result(result, recognizer, *, require_certified):
    count = len(result.roots)
    return _recognize_exact_roots(
        result.roots,
        result.equations,
        result.variables,
        (result.precision_digits,) * count,
        (result.verification_digits,) * count,
        lambda value, digits: recognizer(value),
        require_certified=require_certified,
    )


def test_solver_result_retains_original_equations() -> None:
    x, y = sp.symbols("x y")
    result = polysolve((sp.Eq(x, y), y**2 - 2), (x, y))

    assert result.equations == (x - y, y**2 - 2)


def test_exact_joint_certification_for_recognized_roots() -> None:
    x, y = sp.symbols("x y")
    result = polysolve((x - y, y**2 - 2), (x, y), digits=60)

    recognized = _recognize_result(
        result,
        lambda value: _recognized((-2, 0, 1)),
        require_certified=True,
    )

    assert len(recognized) == 2
    assert all(item.scalar_certified for item in recognized)
    assert all(item.jointly_certified for item in recognized)
    assert all(item.certified for item in recognized)
    assert all(item.equation_values == (0, 0) for item in recognized)
    for item in recognized:
        exact_x, exact_y = item.exact_coordinates
        assert sp.simplify(exact_x - exact_y) == 0
        assert sp.simplify(exact_y**2 - 2) == 0


def test_joint_certification_detects_false_scalar_combination() -> None:
    x, y = sp.symbols("x y")
    near_sqrt2 = sp.N(sp.sqrt(2), 30)
    nearby_rational = sp.Rational(14142136, 10_000_000)
    result = PolynomialSystemRoots(
        roots=((near_sqrt2, sp.N(nearby_rational, 30)),),
        variables=(x, y),
        equations=(x - y,),
        groebner_basis=(x - y,),
        precision_digits=30,
        working_digits=40,
        verification_digits=6,
        max_relative_residual=sp.Float("1e-8"),
        method="test",
    )

    recognitions = iter(
        (
            _recognized((-2, 0, 1)),
            _recognized((-14_142_136, 10_000_000)),
        )
    )
    recognized = _recognize_result(
        result,
        lambda value: next(recognitions),
        require_certified=False,
    )[0]

    assert recognized.scalar_certified
    assert not recognized.jointly_certified
    assert not recognized.certified
    assert recognized.equation_values[0] != 0


def test_required_joint_certification_rejects_false_combination() -> None:
    x, y = sp.symbols("x y")
    near_sqrt2 = sp.N(sp.sqrt(2), 30)
    nearby_rational = sp.Rational(14142136, 10_000_000)
    result = PolynomialSystemRoots(
        roots=((near_sqrt2, sp.N(nearby_rational, 30)),),
        variables=(x, y),
        equations=(x - y,),
        groebner_basis=(x - y,),
        precision_digits=30,
        working_digits=40,
        verification_digits=6,
        max_relative_residual=sp.Float("1e-8"),
        method="test",
    )

    recognitions = iter(
        (
            _recognized((-2, 0, 1)),
            _recognized((-14_142_136, 10_000_000)),
        )
    )
    with pytest.raises(ExactCertificationError, match="complete scalar and joint"):
        _recognize_result(
            result,
            lambda value: next(recognitions),
            require_certified=True,
        )


def test_uncertified_scalar_prevents_full_certification() -> None:
    x = sp.symbols("x")
    result = polysolve((x**2 - 2,), (x,), digits=50)

    recognized = _recognize_result(
        result,
        lambda value: _recognized((-2, 0, 1), certified=False),
        require_certified=False,
    )

    assert all(item.jointly_certified for item in recognized)
    assert all(not item.scalar_certified for item in recognized)
    assert all(not item.certified for item in recognized)


def test_joint_certification_with_algebraic_coefficients() -> None:
    x, y = sp.symbols("x y")
    result = polysolve(
        (x - sp.sqrt(2) * y, y**2 - 3),
        (x, y),
        digits=60,
    )

    def recognize(value):
        magnitude = abs(complex(sp.N(value, 17)))
        if magnitude > 2:
            return _recognized((-6, 0, 1))
        return _recognized((-3, 0, 1))

    recognized = _recognize_result(
        result,
        recognize,
        require_certified=True,
    )

    assert len(recognized) == 2
    assert all(item.certified for item in recognized)
    assert all(item.equation_values == (0, 0) for item in recognized)


def test_exact_certification_supports_rootof_degree_five() -> None:
    x = sp.symbols("x")
    result = polysolve((x**5 - x - 1,), (x,), digits=60)

    recognized = _recognize_result(
        result,
        lambda value: _recognized((-1, -1, 0, 0, 0, 1)),
        require_certified=True,
    )

    assert len(recognized) == 5
    assert all(item.certified for item in recognized)
    assert all(item.equation_values == (0,) for item in recognized)


def test_joint_certification_uses_original_radical_equation() -> None:
    from algroots import algsolve

    x = sp.symbols("x")
    result = algsolve((sp.sqrt(x) - (x - 2),), (x,), digits=50)

    recognized = _recognize_result(
        result,
        lambda value: _recognized((-4, 1)),
        require_certified=True,
    )

    assert len(recognized) == 1
    item = recognized[0]
    assert item.exact_coordinates == (sp.Integer(4),)
    assert item.equation_values == (0,)
    assert item.certified
