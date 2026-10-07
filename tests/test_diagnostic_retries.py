from types import SimpleNamespace

import mpmath as mp
import pytest
import sympy as sp

from algroots import HomotopyRecoveryOptions, polysolve
from algroots.errors import HomotopySolveError

x = sp.Symbol("x")


def _install_proposals(monkeypatch, *, fail_mode="certificate"):
    import algroots.recovery as recovery

    calls = []

    def prefix(system, start, options, **kwargs):
        calls.append((start[0], options.initial_digits, options.residual_digits))
        negative = start[0] == -1
        value = -1 if negative else (1 if options.initial_digits >= 100 else 2)
        if fail_mode == "continuation" and not negative and options.initial_digits < 100:
            return SimpleNamespace(success=False, message="minimum step/maximum precision reached")
        with mp.workdps(options.initial_digits):
            endpoint = (mp.mpc(value),)
        return SimpleNamespace(success=True, endpoint=endpoint, final_digits=options.initial_digits)

    monkeypatch.setattr(recovery, "track_path", prefix)
    monkeypatch.setattr(
        recovery,
        "cauchy_endgame",
        lambda system, start, **kwargs: SimpleNamespace(
            converged=True, endpoint=start, cycle_number=1
        ),
    )
    monkeypatch.setattr(
        recovery,
        "track_projective_path",
        lambda *args, **kwargs: SimpleNamespace(
            success=False,
            classification="unresolved",
            message="precision failure",
            chart_switches=(),
        ),
    )
    return calls


@pytest.mark.parametrize("mode", ["certificate", "continuation"])
def test_retry_only_unproved_path_and_raise_precision(monkeypatch, mode):
    calls = _install_proposals(monkeypatch, fail_mode=mode)
    result = polysolve(
        (x**2 - 1,),
        (x,),
        method="homotopy",
        homotopy_recovery=True,
        recognize=False,
        digits=40,
        guard_digits=20,
    )
    assert len(result.roots) == 2 and result.completeness.status == "certified"
    assert len(calls) == 3
    assert [c[1] for c in calls] == [60, 60, 120]
    assert calls[-1][2] > calls[0][2]
    assert len(result.recovery_records) == 3
    retried = result.recovery_records[-1]
    assert retried.retry_round == 1 and retried.working_digits == 120
    assert retried.retry_reason == (
        "endpoint_certification_failed" if mode == "certificate" else "continuation_unresolved"
    )
    assert retried.certification_status == "certified"
    assert sum(r.retry_disposition == "certified_skip" for r in result.recovery_records) == 2


def test_precision_cap_does_not_repeat_identical_failed_work(monkeypatch):
    calls = _install_proposals(monkeypatch)
    with pytest.raises(HomotopySolveError) as caught:
        polysolve(
            (x**2 - 1,),
            (x,),
            method="homotopy",
            homotopy_recovery=True,
            recognize=False,
            digits=40,
            guard_digits=20,
            max_precision_digits=60,
            homotopy_gamma_attempts=1,
            recovery_options=HomotopyRecoveryOptions(max_retry_rounds=4),
        )
    assert len(calls) == 2
    assert len(caught.value.certified_endpoints) == 1
    assert len(caught.value.recovery_records) == 2


def test_retry_can_be_disabled(monkeypatch):
    calls = _install_proposals(monkeypatch)
    with pytest.raises(HomotopySolveError):
        polysolve(
            (x**2 - 1,),
            (x,),
            method="homotopy",
            homotopy_recovery=True,
            recognize=False,
            digits=40,
            guard_digits=20,
            homotopy_gamma_attempts=1,
            recovery_options=HomotopyRecoveryOptions(max_retry_rounds=0),
        )
    assert len(calls) == 2


def test_structural_budget_does_not_trigger_precision_retry(monkeypatch):
    import algroots.recovery as recovery

    calls = []

    def prefix(*args, **kwargs):
        calls.append(1)
        return SimpleNamespace(success=False, message="maximum path steps exceeded")

    monkeypatch.setattr(recovery, "track_path", prefix)
    monkeypatch.setattr(
        recovery,
        "track_projective_path",
        lambda *args, **kwargs: SimpleNamespace(
            success=False,
            classification="unresolved",
            message="projective segment budget exhausted",
            chart_switches=(),
        ),
    )
    with pytest.raises(HomotopySolveError) as caught:
        polysolve(
            (x - 1,),
            (x,),
            method="homotopy",
            homotopy_recovery=True,
            recognize=False,
            homotopy_gamma_attempts=1,
        )
    assert len(calls) == 1
    assert caught.value.recovery_records[0].retry_disposition == "structural_limit_or_infinity"


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(max_retry_rounds=-1),
        dict(max_retry_rounds=True),
        dict(retry_precision_growth=1),
        dict(retry_precision_growth=2.5),
    ],
)
def test_retry_options_validation(kwargs):
    with pytest.raises(ValueError):
        HomotopyRecoveryOptions(**kwargs)


def test_retries_share_exact_extraction_kernels(monkeypatch):
    import algroots.certification as certification

    calls = _install_proposals(monkeypatch)
    original = certification.rur_from_quotient
    extractions = []

    def counted(*args, **kwargs):
        extractions.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(certification, "rur_from_quotient", counted)
    result = polysolve(
        (x**2 - 1,),
        (x,),
        method="homotopy",
        homotopy_recovery=True,
        recognize=False,
        digits=40,
        guard_digits=20,
    )
    assert len(calls) == 3 and len(extractions) == 1
    assert any("0 distinct roots" in r.message for r in result.recovery_records)


def test_new_gamma_does_not_assume_old_path_endpoint_identity(monkeypatch):
    import algroots.recovery as recovery

    calls = []

    def prefix(system, start, **kwargs):
        calls.append(start)
        value = -1 if len(calls) <= 2 else (1 if start[0] == 1 else -1)
        return SimpleNamespace(success=True, endpoint=(mp.mpc(value),))

    monkeypatch.setattr(recovery, "gamma_candidates", lambda *args: (sp.S.One, -sp.S.One))
    monkeypatch.setattr(recovery, "track_path", prefix)
    monkeypatch.setattr(
        recovery,
        "cauchy_endgame",
        lambda system, start, **kwargs: SimpleNamespace(
            converged=True, endpoint=start, cycle_number=1
        ),
    )
    result = polysolve(
        (x**2 - 1,), (x,), method="homotopy", homotopy_recovery=True, recognize=False
    )
    assert len(calls) == 4 and len(result.roots) == 2
    assert result.homotopy_gamma_attempts == 2
