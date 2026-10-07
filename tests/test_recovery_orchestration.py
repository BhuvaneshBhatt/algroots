import pytest
import sympy as sp

from algroots import HomotopyRecoveryOptions, polysolve
from algroots.continuation import PathTrackingError
from algroots.errors import HomotopySolveError

x, y = sp.symbols("x y")


@pytest.mark.parametrize(
    "equations,variables,multiplicity",
    [((x**2,), (x,), 2), ((x**3,), (x,), 3), ((x**2, y**2), (x, y), 4), ((x - 2,), (x,), 1)],
)
def test_certified_high_level_recovery(equations, variables, multiplicity):
    result = polysolve(
        equations, variables, method="homotopy", homotopy_recovery=True, recognize=False, digits=30
    )
    assert len(result.roots) == 1
    assert result.completeness.status == "certified"
    assert result.root_certifications[0].certificate.multiplicity == multiplicity
    assert result.root_certifications[0].certificate.verify()
    assert result.recovery_records
    if multiplicity > 1:
        assert result.root_deflations[0].regular
        assert result.root_deflations[0].verify()


def test_automatic_chart_fallback_through_infinity(monkeypatch):
    import algroots.recovery as recovery

    monkeypatch.setattr(recovery, "gamma_candidates", lambda *args: (sp.Integer(1),))

    def fail(*args, **kwargs):
        raise PathTrackingError("require projective recovery")

    monkeypatch.setattr(recovery, "cauchy_endgame", fail)
    result = polysolve(
        (-x - 1,), (x,), method="homotopy", homotopy_recovery=True, recognize=False, digits=30
    )
    assert abs(result.roots[0][0] + 1) < sp.Rational(1, 10**25)
    assert result.completeness.status == "certified"
    assert any(r.chart_switches >= 1 for r in result.recovery_records)


def test_projective_fallback_when_endgame_fails(monkeypatch):
    import algroots.recovery as recovery

    def fail(*args, **kwargs):
        raise PathTrackingError("forced endgame failure")

    monkeypatch.setattr(recovery, "cauchy_endgame", fail)
    result = polysolve(
        (x - 2,), (x,), method="homotopy", homotopy_recovery=True, recognize=False, digits=30
    )
    assert "projective_charts" in result.recovery_records[0].stages
    assert result.root_certifications[0].status == "certified"


def test_exhausted_limits_never_return_incomplete_roots(monkeypatch):
    import algroots.recovery as recovery

    monkeypatch.setattr(recovery, "gamma_candidates", lambda *args: (sp.Integer(1),))
    with pytest.raises(HomotopySolveError) as caught:
        polysolve(
            (-x - 1,),
            (x,),
            method="homotopy",
            homotopy_recovery=True,
            recognize=False,
            recovery_options=HomotopyRecoveryOptions(path_steps=1, max_segments=1),
        )
    assert caught.value.recovery_records
    assert "certified 0/1" in str(caught.value)


@pytest.mark.parametrize(
    "options",
    [dict(max_paths=0), dict(path_steps=False), dict(endgame_samples=4), dict(endgame_levels=1)],
)
def test_invalid_recovery_options(options):
    with pytest.raises(ValueError):
        HomotopyRecoveryOptions(**options)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(homotopy_recovery=True),
        dict(recovery_options=HomotopyRecoveryOptions()),
        dict(homotopy_recovery=1),
        dict(method="homotopy", homotopy_recovery=True, homotopy_parallel=True),
    ],
)
def test_invalid_recovery_route(kwargs):
    with pytest.raises(ValueError):
        polysolve((x - 1,), (x,), recognize=False, **kwargs)


def test_deflation_limit_reported_without_losing_endpoint_proof():
    result = polysolve(
        (x**4,),
        (x,),
        method="homotopy",
        homotopy_recovery=True,
        recognize=False,
        recovery_options=HomotopyRecoveryOptions(max_deflation_stages=1),
    )
    assert result.root_certifications[0].certificate.multiplicity == 4
    assert not result.root_deflations[0].regular
    assert result.root_deflations[0].stopping_reason == "stage_limit"


def test_recovery_reuses_one_original_groebner(monkeypatch):
    original = sp.groebner
    calls = []

    def counted(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(sp, "groebner", counted)
    result = polysolve(
        (x**2,), (x,), method="homotopy", homotopy_recovery=True, recognize=False, digits=30
    )
    assert len(calls) == 1
    assert result.root_deflations[0].regular


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(recovery_options={}),
        dict(homotopy_gamma_attempts=0),
        dict(max_homotopy_paths=0),
        dict(max_precision_digits=30),
    ],
)
def test_recovery_invalid_global_controls(kwargs):
    with pytest.raises((ValueError, TypeError)):
        polysolve(
            (x**2,),
            (x,),
            method="homotopy",
            homotopy_recovery=True,
            recognize=False,
            digits=30,
            **kwargs,
        )


def test_algebraic_recovery_limitation_is_explicit():
    with pytest.raises(HomotopySolveError, match="rational coefficients"):
        polysolve(
            (x - sp.sqrt(2),), (x,), method="homotopy", homotopy_recovery=True, recognize=False
        )


def test_recovery_with_paths_at_infinity_accounts_for_finite_roots():
    result = polysolve(
        (x * y - 1, y - 1),
        (x, y),
        method="homotopy",
        homotopy_recovery=True,
        recognize=False,
        digits=30,
    )
    assert len(result.roots) == 1 and result.completeness.status == "certified"
    assert result.root_certifications[0].certificate.point == (1, 1)
    assert len(result.recovery_records) == 2
    assert any(record.outcome == "unresolved" for record in result.recovery_records)


def test_recovery_retains_high_precision_endpoint_proposals(monkeypatch):
    from types import SimpleNamespace

    import mpmath as mp

    import algroots.recovery as recovery

    delta = sp.Rational(1, 10**25)

    def prefix(system, start, **kwargs):
        with mp.workdps(80):
            value = mp.mpc(1) + (mp.mpf("1e-25") if start[0] == -1 else 0)
        return SimpleNamespace(success=True, endpoint=(value,))

    monkeypatch.setattr(recovery, "track_path", prefix)
    monkeypatch.setattr(
        recovery,
        "cauchy_endgame",
        lambda system, start, **kwargs: SimpleNamespace(
            converged=True, endpoint=start, cycle_number=1
        ),
    )
    result = polysolve(
        ((x - 1) * (x - 1 - delta),),
        (x,),
        method="homotopy",
        homotopy_recovery=True,
        recognize=False,
        digits=60,
    )
    assert len(result.roots) == 2
    assert {a.certificate.point for a in result.root_certifications} == {
        (sp.Integer(1),),
        (1 + delta,),
    }
