import pytest
import sympy as sp

pytestmark = pytest.mark.recognition


def test_recognize_system_roots_with_algrecognize() -> None:
    pytest.importorskip("flint")
    pytest.importorskip("algrecognize")

    from algroots import polysolve
    from algroots.recognition import recognize_system_roots

    x, y = sp.symbols("x y")
    result = polysolve((x - y, y**2 - 2), (x, y), digits=70, recognize=False)
    recognized = recognize_system_roots(result, max_degree=2, require_certified=True)

    assert len(recognized) == 2
    assert all(len(item.coordinates) == 2 for item in recognized)
    assert all(item.scalar_certified for item in recognized)
    assert all(item.jointly_certified for item in recognized)
    assert all(item.certified for item in recognized)
    assert all(item.equation_values == (0, 0) for item in recognized)
    for item in recognized:
        for coordinate in item.coordinates:
            assert tuple(int(c) for c in coordinate.polynomial.coeffs()) == (-2, 0, 1)


def test_monodromy_recognition_uses_per_root_precision(monkeypatch) -> None:
    import sys
    import types

    import mpmath as mp

    from algroots import recognition as recognition_module
    from algroots.monodromy import MonodromyOrbitResult, MonodromyRootInfo

    x = sp.symbols("x")
    info = (
        MonodromyRootInfo(40, 20, mp.mpf("1e-30"), True),
        MonodromyRootInfo(90, 55, mp.mpf("1e-65"), True),
    )
    orbit = MonodromyOrbitResult(
        roots=((sp.Integer(1),), (sp.Integer(2),)),
        variables=(x,),
        equations=((x - 1) * (x - 2),),
        loops_completed=1,
        paths_tracked=2,
        new_roots_per_loop=(1,),
        perturbations=((1 + 0j,),),
        all_paths_successful=True,
        root_info=info,
    )

    seen_digits: list[int] = []

    class FakePoly:
        def __init__(self, root: int) -> None:
            self.root = root

        def coeffs(self):
            return (-self.root, 1)

    class FakeRecognition:
        def __init__(self, root: int) -> None:
            self.polynomial = FakePoly(root)
            self.certified = True

    class FakeCtx:
        prec = 53

    def fake_to_flint(value, digits):
        seen_digits.append(digits)
        return int(value)

    fake_alg = types.ModuleType("algrecognize")
    fake_alg.algrecognize = lambda value, **kwargs: FakeRecognition(value)
    fake_flint = types.ModuleType("flint")
    fake_flint.ctx = FakeCtx()
    monkeypatch.setitem(sys.modules, "algrecognize", fake_alg)
    monkeypatch.setitem(sys.modules, "flint", fake_flint)
    monkeypatch.setattr(recognition_module, "_to_flint_number", fake_to_flint)

    recognized = recognition_module.recognize_system_roots(
        orbit,
        max_degree=1,
        require_certified=True,
    )

    assert seen_digits == [20, 55]
    assert [item.exact_coordinates for item in recognized] == [(1,), (2,)]
    assert all(item.certified for item in recognized)


def test_monodromy_recognition_does_not_change_completeness(monkeypatch) -> None:
    import sys
    import types

    import mpmath as mp

    from algroots import recognition as recognition_module
    from algroots.monodromy import MonodromyOrbitResult, MonodromyRootInfo

    x = sp.symbols("x")
    orbit = MonodromyOrbitResult(
        roots=((sp.Integer(1),),),
        variables=(x,),
        equations=(x**2 - 1,),
        loops_completed=1,
        paths_tracked=1,
        new_roots_per_loop=(0,),
        perturbations=((1 + 0j,),),
        all_paths_successful=True,
        stopping_reason="max_loops",
        completeness_basis="none",
        root_info=(MonodromyRootInfo(50, 30, mp.mpf("1e-40"), True),),
    )

    class FakePoly:
        def coeffs(self):
            return (-1, 1)

    class FakeRecognition:
        polynomial = FakePoly()
        certified = True

    class FakeCtx:
        prec = 53

    fake_alg = types.ModuleType("algrecognize")
    fake_alg.algrecognize = lambda value, **kwargs: FakeRecognition()
    fake_flint = types.ModuleType("flint")
    fake_flint.ctx = FakeCtx()
    monkeypatch.setitem(sys.modules, "algrecognize", fake_alg)
    monkeypatch.setitem(sys.modules, "flint", fake_flint)
    monkeypatch.setattr(recognition_module, "_to_flint_number", lambda value, digits: value)

    recognized = recognition_module.recognize_system_roots(
        orbit,
        max_degree=1,
        require_certified=True,
    )

    assert recognized[0].certified
    assert orbit.stopping_reason == "max_loops"
    assert orbit.completeness_basis == "none"


def test_real_recognition_handles_rational_and_complex_algebraic_roots() -> None:
    """Exercise actual algrecognize/python-flint on simple exact root families."""
    pytest.importorskip("flint")
    pytest.importorskip("algrecognize")

    from algroots import polysolve
    from algroots.recognition import recognize_system_roots

    x = sp.symbols("x")

    rational = polysolve((3 * x - 2,), (x,), digits=60, recognize=False)
    rational_recognized = recognize_system_roots(
        rational,
        max_degree=1,
        require_certified=True,
    )
    assert len(rational_recognized) == 1
    assert rational_recognized[0].exact_coordinates == (sp.Rational(2, 3),)
    assert rational_recognized[0].certified

    complex_result = polysolve((x**2 + 1,), (x,), digits=70, recognize=False)
    complex_recognized = recognize_system_roots(
        complex_result,
        max_degree=2,
        require_certified=True,
    )
    assert {sp.simplify(item.exact_coordinates[0]) for item in complex_recognized} == {
        sp.I,
        -sp.I,
    }
    assert all(item.certified for item in complex_recognized)


def test_real_recognition_respects_degree_bound_failure() -> None:
    pytest.importorskip("flint")
    pytest.importorskip("algrecognize")

    from algroots import polysolve
    from algroots.errors import ExactCertificationError
    from algroots.recognition import recognize_system_roots

    x = sp.symbols("x")
    result = polysolve((x**2 - 2,), (x,), digits=70, recognize=False)

    # A degree-one search cannot certify sqrt(2). algrecognize versions may
    # either fail to return a certified scalar relation or reject the search;
    # the public API must not return a certified system root.
    try:
        recognized = recognize_system_roots(
            result,
            max_degree=1,
            require_certified=True,
        )
    except (ExactCertificationError, ValueError, ArithmeticError):
        return
    assert not all(item.certified for item in recognized)


def test_exact_root_selection_rejects_nearby_ambiguous_roots() -> None:
    from algroots.recognition import ExactCertificationError, _exact_root_from_result

    q = 10**20

    class CloseRootPoly:
        def coeffs(self):
            # (x - 1) * (q*x - (q + 1)) in ascending coefficient order.
            return (q + 1, -(2 * q + 1), q)

    class Recognition:
        polynomial = CloseRootPoly()
        certified = True

    with pytest.raises(ExactCertificationError, match="uniquely identify"):
        _exact_root_from_result(
            Recognition(),
            sp.Float("1.000000000000000000005", 80),
            digits=70,
            verification_digits=30,
        )


def test_exact_root_selection_resolves_nearby_roots_with_sufficient_precision() -> None:
    from algroots.recognition import _exact_root_from_result

    q = 10**20

    class CloseRootPoly:
        def coeffs(self):
            return (q + 1, -(2 * q + 1), q)

    class Recognition:
        polynomial = CloseRootPoly()
        certified = True

    exact = _exact_root_from_result(
        Recognition(),
        sp.Float("1.00000000000000000001", 100),
        digits=90,
        verification_digits=60,
    )
    assert sp.simplify(exact - sp.Rational(q + 1, q)) == 0


def test_require_certified_rejects_scalar_uncertified_result(monkeypatch) -> None:
    import sys
    import types

    from algroots import polysolve
    from algroots import recognition as recognition_module
    from algroots.errors import ExactCertificationError

    x = sp.symbols("x")
    result = polysolve((x - 1,), (x,), digits=40)

    class FakePoly:
        def coeffs(self):
            return (-1, 1)

    class FakeRecognition:
        polynomial = FakePoly()
        certified = False

    class FakeCtx:
        prec = 53

    fake_alg = types.ModuleType("algrecognize")
    fake_alg.algrecognize = lambda value, **kwargs: FakeRecognition()
    fake_flint = types.ModuleType("flint")
    fake_flint.ctx = FakeCtx()
    monkeypatch.setitem(sys.modules, "algrecognize", fake_alg)
    monkeypatch.setitem(sys.modules, "flint", fake_flint)
    monkeypatch.setattr(recognition_module, "_to_flint_number", lambda value, digits: value)

    with pytest.raises(ExactCertificationError, match="complete scalar and joint"):
        recognition_module.recognize_system_roots(
            result,
            max_degree=1,
            require_certified=True,
        )


def test_core_solver_attempts_recognition_by_default(monkeypatch) -> None:
    from algroots import polysolve
    from algroots import recognition as recognition_module

    x = sp.symbols("x")
    sentinel = (object(),)
    seen = {}

    def fake_recognize(result, *, max_degree, require_certified=False, **kwargs):
        seen["max_degree"] = max_degree
        seen["require_certified"] = require_certified
        return sentinel

    monkeypatch.setattr(recognition_module, "recognize_system_roots", fake_recognize)
    result = polysolve((x - 1,), (x,), digits=30)

    assert result.recognition_attempted
    assert result.recognized_roots is sentinel
    assert result.recognition_error is None
    assert seen == {"max_degree": 8, "require_certified": True}


def test_core_solver_can_disable_automatic_recognition(monkeypatch) -> None:
    from algroots import polysolve
    from algroots import recognition as recognition_module

    x = sp.symbols("x")

    def should_not_run(*args, **kwargs):
        raise AssertionError("recognition should be disabled")

    monkeypatch.setattr(recognition_module, "recognize_system_roots", should_not_run)
    result = polysolve((x - 1,), (x,), digits=30, recognize=False)

    assert not result.recognition_attempted
    assert result.recognized_roots is None
    assert result.recognition_error is None


def test_automatic_recognition_failure_preserves_numerical_result(monkeypatch) -> None:
    from algroots import polysolve
    from algroots import recognition as recognition_module
    from algroots.errors import ExactCertificationError

    x = sp.symbols("x")

    def fail_recognition(*args, **kwargs):
        raise ExactCertificationError("no certified relation in budget")

    monkeypatch.setattr(recognition_module, "recognize_system_roots", fail_recognition)
    result = polysolve((x - 1,), (x,), digits=30)

    assert len(result.roots) == 1
    assert result.recognition_attempted
    assert result.recognized_roots is None
    assert "no certified relation in budget" in result.recognition_error


def test_algebraic_solver_recognizes_only_final_projected_result(monkeypatch) -> None:
    from algroots import algsolve
    from algroots import recognition as recognition_module

    x = sp.symbols("x")
    calls = []

    def fake_recognize(result, *, max_degree, require_certified=False, **kwargs):
        calls.append((result.variables, max_degree, require_certified))
        return tuple(object() for _ in result.roots)

    monkeypatch.setattr(recognition_module, "recognize_system_roots", fake_recognize)
    result = algsolve((sp.sqrt(x) - 2,), (x,), digits=30)

    assert result.recognition_attempted
    assert len(calls) == 1
    assert calls[0][0] == (x,)
    assert calls[0][1:] == (8, True)


def test_monodromy_attempts_recognition_by_default(monkeypatch) -> None:
    from algroots import recognition as recognition_module
    from algroots.monodromy import discover_monodromy_orbit

    x = sp.symbols("x")
    sentinel = (object(),)

    monkeypatch.setattr(
        recognition_module,
        "recognize_system_roots",
        lambda result, *, max_degree, require_certified=False, **kwargs: sentinel,
    )
    result = discover_monodromy_orbit(
        (x - 1,),
        (x,),
        ((1,),),
        expected_root_count=1,
        max_loops=1,
    )

    assert result.recognition_attempted
    assert result.recognized_roots is sentinel
    assert result.recognition_error is None


def test_algebraic_monodromy_recognizes_projected_original_result(monkeypatch) -> None:
    from algroots import recognition as recognition_module
    from algroots.monodromy import discover_monodromy_orbit

    x = sp.symbols("x")
    equation = sp.sqrt(x) - (x - 2)
    observed = {}
    sentinel = (object(),)

    def fake_recognize(result, *, max_degree, require_certified=False, **kwargs):
        observed["variables"] = result.variables
        observed["equations"] = result.equations
        observed["tracking_variables"] = result.tracking_variables
        observed["max_degree"] = max_degree
        observed["require_certified"] = require_certified
        return sentinel

    monkeypatch.setattr(recognition_module, "recognize_system_roots", fake_recognize)
    result = discover_monodromy_orbit(
        (equation,),
        (x,),
        ((4,),),
        expected_root_count=1,
        max_loops=1,
    )

    assert result.recognition_attempted
    assert result.recognized_roots is sentinel
    assert observed["variables"] == (x,)
    assert observed["equations"] == (equation,)
    assert len(observed["tracking_variables"]) == 2
    assert observed["max_degree"] == 8
    assert observed["require_certified"] is True
