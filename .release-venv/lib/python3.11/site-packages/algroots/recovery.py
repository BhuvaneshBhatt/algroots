"""Bounded high-level continuation recovery with exact finite-root accounting."""

from dataclasses import dataclass, replace
from time import perf_counter

import sympy as sp

from .continuation import PathTrackerOptions, PathTrackingError, track_path
from .endgames import cauchy_endgame, projective_homotopy
from .errors import HomotopySolveError, PolynomialSystemError, SystemSolveLimitError
from .projective_tracking import track_projective_path
from .quotient import QuotientAlgebra
from .total_degree_homotopy import build_total_degree_homotopy, gamma_candidates


@dataclass(frozen=True)
class HomotopyRecoveryOptions:
    max_paths: int = 64
    gamma_attempts: int = 2
    max_quotient_dimension: int = 128
    path_steps: int = 600
    max_segments: int = 256
    max_chart_switches: int = 32
    endgame_samples: int = 16
    endgame_cycles: int = 4
    endgame_levels: int = 3
    max_deflation_stages: int = 4
    max_deflation_equations: int = 128
    max_retry_rounds: int = 1
    retry_precision_growth: int = 2

    def __post_init__(self):
        for name, value in self.__dict__.items():
            minimum = (
                0 if name == "max_retry_rounds" else 2 if name == "retry_precision_growth" else 1
            )
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.endgame_samples < 8 or self.endgame_levels < 2:
            raise ValueError("endgame_samples >= 8 and endgame_levels >= 2 required")


@dataclass(frozen=True)
class PathRecoveryRecord:
    gamma_attempt: int
    path_index: int
    stages: tuple
    outcome: str
    message: str = ""
    chart_switches: int = 0
    cycle_number: int | None = None
    phase_seconds: tuple = ()
    retry_round: int = 0
    working_digits: int = 0
    retry_reason: str = ""
    certification_status: str = "not_attempted"
    retry_disposition: str = ""


def solve_recovered_homotopy(
    equations,
    variables,
    *,
    digits,
    guard_digits,
    verification_digits,
    maxsteps,
    max_solutions,
    max_paths,
    seed,
    options=None,
    max_precision_digits=None,
    certification_max_dimension=128,
    certification_max_refinements=128,
    certification_max_box_attempts=16,
    gamma_attempts=4,
):
    from .certification import _certify_numerical_roots
    from .deflation import _deflate_certificate
    from .numerical import sympy_to_mpc
    from .solver import (
        CompletenessEvidence,
        PolynomialSystemRoots,
        SolveCostDiagnostics,
        _validate_roots,
    )

    opts = HomotopyRecoveryOptions() if options is None else options
    if not isinstance(opts, HomotopyRecoveryOptions):
        raise TypeError("recovery_options must be HomotopyRecoveryOptions")
    cap = min(opts.max_quotient_dimension, certification_max_dimension)
    quotient = QuotientAlgebra.from_polynomials(equations, variables, max_dimension=cap)
    if quotient.domain != sp.QQ:
        raise HomotopySolveError(
            "bounded certified recovery currently requires rational coefficients"
        )
    expected = quotient.geometric_solution_count
    if expected > max_solutions:
        raise SystemSolveLimitError("geometric count exceeds max_solutions")
    work_digits = digits + guard_digits
    max_digits = max_precision_digits or max(80, work_digits * 2)
    if max_digits < work_digits:
        raise ValueError("max_precision_digits must cover recovery working precision")
    tracker = PathTrackerOptions(
        initial_digits=work_digits,
        max_digits=max_digits,
        residual_digits=min(24, verification_digits),
        max_steps=opts.path_steps,
    )
    base = PolynomialSystemRoots(
        (),
        variables,
        tuple(equations),
        tuple(p.as_expr() for p in quotient.groebner_basis.polys),
        digits,
        work_digits,
        verification_digits,
        sp.Integer(0),
        "recovered_homotopy",
        quotient_dimension=quotient.dimension,
        total_multiplicity=quotient.dimension,
        geometric_solution_count=expected,
        is_radical=quotient.dimension == expected,
        has_multiple_roots=quotient.dimension > expected,
        solver_variables=variables,
    )
    if expected == 0:
        return replace(
            base,
            root_certifications=(),
            root_deflations=(),
            completeness=CompletenessEvidence("certified", "unit_ideal", 0, 0, 0),
            cost_diagnostics=SolveCostDiagnostics(),
        )
    problem = build_total_degree_homotopy(
        equations, variables, max_paths=min(opts.max_paths, max_paths)
    )
    records = []
    certified = {}
    proofs = ()
    all_proofs = []
    certification_context = {}
    starts = tuple(problem.iter_start_points())
    for gamma_index, gamma in enumerate(
        gamma_candidates(seed, min(opts.gamma_attempts, gamma_attempts)), 1
    ):
        system = problem.with_gamma(gamma).system()
        # Successful endpoint proofs survive every retry and gamma change. Path
        # indices can only be skipped within one gamma: changing gamma changes
        # their correspondence to endpoints.
        pending = dict.fromkeys(range(len(starts)), "")
        projective = None
        previous_digits = work_digits
        for retry_round in range(opts.max_retry_rounds + 1):
            retry_digits = (
                work_digits
                if retry_round == 0
                else min(max_digits, previous_digits * opts.retry_precision_growth)
            )
            if retry_round and retry_digits == previous_digits:
                break
            previous_digits = retry_digits
            retry_tracker = replace(
                tracker,
                initial_digits=retry_digits,
                residual_digits=min(
                    max_digits - 1,
                    max(tracker.residual_digits, min(retry_digits - 8, 24 * 2**retry_round)),
                ),
            )
            candidates = []
            candidate_paths = []
            record_indices = {}
            reasons = {}
            for path_index, reason in pending.items():
                start = starts[path_index]
                path_started = perf_counter()
                phase_started = path_started
                stage_seconds = {}
                stages = []
                messages = []
                switches = 0
                cycle = None
                candidate = None
                retryable = True
                final_digits = retry_digits
                try:
                    stages.append("affine_prefix")
                    prefix = track_path(system, start, t_end=0.9, options=retry_tracker)
                    final_digits = max(final_digits, getattr(prefix, "final_digits", retry_digits))
                    if not prefix.success:
                        raise HomotopySolveError(prefix.message)
                    stage_seconds["affine_prefix"] = perf_counter() - phase_started
                    phase_started = perf_counter()
                    stages.append("cauchy_endgame")
                    end = cauchy_endgame(
                        system,
                        prefix.endpoint,
                        samples=opts.endgame_samples,
                        max_cycle=opts.endgame_cycles,
                        levels=opts.endgame_levels,
                        options=retry_tracker,
                    )
                    stage_seconds["cauchy_endgame"] = perf_counter() - phase_started
                    cycle = end.cycle_number
                    if end.converged:
                        candidate = tuple(
                            sp.Float(v.real, final_digits) + sp.I * sp.Float(v.imag, final_digits)
                            for v in end.endpoint
                        )
                    else:
                        messages.append("endgame did not stabilize")
                except (
                    PolynomialSystemError,
                    PathTrackingError,
                    ValueError,
                    ArithmeticError,
                ) as exc:
                    messages.append(str(exc))
                if stages[-1] not in stage_seconds:
                    stage_seconds[stages[-1]] = perf_counter() - phase_started
                if candidate is None:
                    phase_started = perf_counter()
                    try:
                        stages.append("projective_charts")
                        if projective is None:
                            projective = projective_homotopy(
                                system.expressions,
                                variables,
                                system.parameter,
                                patch=(*([0] * len(variables)), 1),
                            )
                        path = track_projective_path(
                            projective,
                            projective.lift(start),
                            options=retry_tracker,
                            max_segments=opts.max_segments,
                            max_chart_switches=opts.max_chart_switches,
                        )
                        switches = len(path.chart_switches)
                        final_digits = max(
                            final_digits, getattr(path, "final_digits", retry_digits)
                        )
                        if path.success and path.classification == "finite":
                            candidate = tuple(
                                sp.Float(v.real, final_digits)
                                + sp.I * sp.Float(v.imag, final_digits)
                                for v in path.affine_endpoint()
                            )
                        else:
                            messages.append(path.classification + ": " + path.message)
                            retryable = path.classification != "numerically_near_infinity"
                            retryable &= not any(
                                text in path.message
                                for text in (
                                    "segment budget",
                                    "chart switch budget",
                                    "maximum path steps",
                                )
                            )
                    except (
                        PolynomialSystemError,
                        PathTrackingError,
                        ValueError,
                        ArithmeticError,
                    ) as exc:
                        messages.append(str(exc))
                if stages[-1] not in stage_seconds:
                    stage_seconds[stages[-1]] = perf_counter() - phase_started
                stage_seconds["total"] = perf_counter() - path_started
                if candidate is not None:
                    candidates.append(candidate)
                    candidate_paths.append(path_index)
                else:
                    if retryable:
                        reasons[path_index] = "continuation_unresolved"
                record_indices[path_index] = len(records)
                records.append(
                    PathRecoveryRecord(
                        gamma_index,
                        path_index,
                        tuple(stages),
                        "candidate" if candidate is not None else "unresolved",
                        "; ".join(messages),
                        switches,
                        cycle,
                        tuple(stage_seconds.items()),
                        retry_round,
                        final_digits,
                        reason,
                        retry_disposition="raise_precision"
                        if candidate is None and retryable
                        else "structural_limit_or_infinity"
                        if candidate is None
                        else "await_certificate",
                    )
                )
            if candidates:
                provisional = replace(base, roots=tuple(candidates))
                proofs = _certify_numerical_roots(
                    provisional,
                    quotient=quotient,
                    max_quotient_dimension=cap,
                    max_refinements=certification_max_refinements,
                    max_box_attempts=certification_max_box_attempts,
                    context=certification_context,
                    neighbour_points=tuple(
                        tuple(sp.N(c, retry_digits) for c in point) for point in certified
                    ),
                )
                all_proofs.extend(proofs)
                for path_index, proof in zip(candidate_paths, proofs, strict=True):
                    index = record_indices[path_index]
                    records[index] = replace(
                        records[index],
                        certification_status=proof.status,
                        retry_disposition="certified_skip"
                        if proof.certificate is not None
                        else "raise_precision",
                        message="; ".join(
                            part for part in (records[index].message, proof.error) if part
                        ),
                    )
                    if proof.certificate is not None:
                        certified[proof.certificate.point] = proof
                    else:
                        reasons[path_index] = "endpoint_certification_failed"
            if len(certified) == expected:
                break
            # Only failed paths/candidates are retracked. A new gamma below must
            # consider every start again; no cross-gamma path identity is assumed.
            pending = reasons
            if not pending:
                break
        if len(certified) == expected:
            break
    if len(certified) != expected:
        error = HomotopySolveError(
            f"bounded recovery certified {len(certified)}/{expected} finite roots; "
            f"{len(records)} path attempts; limits or unresolved endpoints remain"
        )
        error.recovery_records = tuple(records)
        error.certification_attempts = tuple(all_proofs)
        error.certified_endpoints = tuple(certified.values())
        raise error
    ordered = sorted(certified.values(), key=lambda a: sp.sstr(a.certificate.point))
    candidates = [tuple(sp.N(c, work_digits) for c in proof.certificate.point) for proof in ordered]
    roots, diagnostics, residual = _validate_roots(
        candidates,
        tuple(equations),
        variables,
        digits=digits,
        work_digits=work_digits,
        verification_digits=verification_digits,
        maxsteps=maxsteps,
        expected_count=expected,
        error_type=HomotopySolveError,
    )
    # Validation sorts roots; associate proofs through their certified exact point.
    aligned = []
    unused = set(range(len(candidates)))
    for root in roots:
        index = min(
            unused,
            key=lambda i: max(
                abs(sympy_to_mpc(a - b, work_digits))
                for a, b in zip(root, candidates[i], strict=True)
            ),
        )
        unused.remove(index)
        aligned.append(ordered[index])
    deflations = []
    for proof in aligned:
        certificate = proof.certificate
        deflations.append(
            _deflate_certificate(
                certificate,
                max_stages=opts.max_deflation_stages,
                max_added_equations=opts.max_deflation_equations,
            )
            if certificate.is_singular
            else None
        )
    return replace(
        base,
        roots=roots,
        diagnostics=diagnostics,
        max_relative_residual=residual,
        root_certifications=tuple(aligned),
        root_deflations=tuple(deflations),
        recovery_records=tuple(records),
        homotopy_paths_total=len(records),
        homotopy_paths_succeeded=sum(r.outcome == "candidate" for r in records),
        homotopy_paths_failed=sum(r.outcome == "unresolved" for r in records),
        homotopy_gamma_attempts=gamma_index,
        cost_diagnostics=SolveCostDiagnostics(
            normal_form_statistics=tuple(quotient.normal_form_cache_info.items()),
            quotient_operation_statistics=tuple(quotient.operation_diagnostics.items()),
        ),
        completeness=CompletenessEvidence(
            "certified",
            "exact_endpoint_certificates_and_finite_quotient_count",
            expected,
            len(roots),
            quotient.dimension,
            (
                "All finite roots accounted for by exact endpoint proofs; numerical paths and infinity are not certified.",
            ),
        ),
    )
