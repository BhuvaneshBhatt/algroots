"""Numerical projective continuation with adaptive coordinate-chart switching."""

import math
from dataclasses import dataclass, replace

import mpmath as mp
import sympy as sp

from .continuation import PathTrackerOptions, track_path
from .endgames import ProjectiveHomotopy
from .numerical import number_to_mpc


@dataclass(frozen=True)
class ChartSwitch:
    parameter: float
    previous_patch: tuple
    next_patch: tuple
    point: tuple


@dataclass(frozen=True)
class ProjectivePathResult:
    endpoint: tuple
    final_t: float
    success: bool
    patch: tuple
    segments: tuple
    chart_switches: tuple
    classification: str
    message: str = ""
    status: str = "numerical"
    limitations: tuple = (
        "Numerical chart tracking is not a certified path enclosure.",
        "A small homogenizing coordinate is not a proof of infinity.",
    )

    def affine_endpoint(self):
        if self.classification != "finite":
            raise ValueError("endpoint is not numerically resolved as finite")
        return tuple(v / self.endpoint[-1] for v in self.endpoint[:-1])


def _coordinate_patch(size, index):
    return tuple(sp.Integer(i == index) for i in range(size))


def _rechart(projective, patch):
    from .continuation import SympyHomotopy

    variables = projective.system.variables
    expressions = (
        *projective.system.expressions[:-1],
        sum(c * v for c, v in zip(patch, variables, strict=True)) - 1,
    )
    return replace(
        projective,
        system=SympyHomotopy(expressions, variables, projective.system.parameter),
        patch=patch,
    )


def track_projective_path(
    projective,
    start,
    *,
    t_start=0,
    t_end=1,
    options=None,
    segment_size=0.05,
    min_segment=1e-8,
    switch_ratio=0.5,
    max_segments=1000,
    max_chart_switches=128,
):
    """Track a homogeneous point, switching to its largest usable coordinate.

    The original homogeneous equations are retained in every chart. Successful
    short segments trigger scale-based switching with hysteresis; a failed
    segment can resume from its last accepted point in a different chart.
    Otherwise the segment shrinks. Actual projective singularities, very rapid
    chart changes or exhausted budgets return an explicit unresolved result.
    Start coordinates are homogenous: use ``projective.lift(affine_point)``.
    """
    if not isinstance(projective, ProjectiveHomotopy):
        raise TypeError("projective must be a ProjectiveHomotopy")
    opts = options or PathTrackerOptions()
    if not all(
        math.isfinite(float(v)) for v in (t_start, t_end, segment_size, min_segment, switch_ratio)
    ):
        raise ValueError("tracking controls must be finite")
    if not 0 < min_segment <= segment_size or not 0 < switch_ratio < 1:
        raise ValueError("invalid segment bounds or switching ratio")
    for value in (max_segments, max_chart_switches):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("tracking budgets must be positive integers")
    if len(start) != len(projective.system.variables):
        raise ValueError("homogeneous start point dimension does not match")
    segments, switches = [], []
    chart = projective
    with mp.workdps(opts.max_digits + 10):
        point = tuple(number_to_mpc(v, opts.max_digits + 5) for v in start)
        if any(not mp.isfinite(v) for v in point) or max(abs(v) for v in point) == 0:
            raise ValueError("homogeneous point must be finite and nonzero")
        current, target = mp.mpf(str(t_start)), mp.mpf(str(t_end))
        direction = 1 if target >= current else -1
        switch_budget_exhausted = False
        step = mp.mpf(str(segment_size))
        minimum = mp.mpf(str(min_segment))

        def finish(success, message=""):
            scale = max(abs(v) for v in point)
            finite = abs(point[-1]) / scale > mp.power(10, -opts.residual_digits // 2)
            classification = (
                "finite"
                if success and finite
                else ("numerically_near_infinity" if success else "unresolved")
            )
            return ProjectivePathResult(
                point,
                float(current),
                success,
                chart.patch,
                tuple(segments),
                tuple(switches),
                classification,
                message,
            )

        def switch_if_needed(force=False):
            nonlocal point, chart, switch_budget_exhausted
            index = max(range(len(point)), key=lambda i: abs(point[i]))
            next_patch = _coordinate_patch(len(point), index)
            current_index = next(
                (i for i in range(len(point)) if chart.patch == _coordinate_patch(len(point), i)),
                None,
            )
            if next_patch == chart.patch:
                return False
            if (
                not force
                and current_index is not None
                and abs(point[current_index]) > switch_ratio * abs(point[index])
            ):
                return False
            if len(switches) >= max_chart_switches:
                switch_budget_exhausted = True
                return False
            normalized = tuple(v / point[index] for v in point)
            switches.append(ChartSwitch(float(current), chart.patch, next_patch, normalized))
            chart = _rechart(chart, next_patch)
            point = normalized
            return True

        # Normalize into a coordinate chart before the first seed correction.
        switch_if_needed(force=True)
        if not switches:
            denominator = sum(c * v for c, v in zip(chart.patch, point, strict=True))
            if denominator == 0:
                return finish(False, "initial patch is unavailable and switch budget was exhausted")
            point = tuple(v / denominator for v in point)
        # Even a zero-length request must correct/validate the initial point.
        seed = track_path(
            chart.system, point, t_start=float(current), t_end=float(current), options=opts
        )
        segments.append(seed)
        if not seed.success:
            return finish(False, "initial projective seed is not resolved: " + seed.message)
        point = tuple(seed.endpoint)
        attempts = 0
        while direction * (target - current) > 0:
            if attempts >= max_segments:
                return finish(False, "projective segment budget exhausted")
            switch_if_needed()
            if switch_budget_exhausted:
                return finish(False, "projective chart switch budget exhausted")
            # Bound the coordinate motion predicted in the current chart. This
            # reduces the risk of a long segment jumping across a chart pole while
            # both endpoints happen to be well scaled.
            try:
                tangent = mp.lu_solve(
                    chart.system.jacobian(point, current),
                    -chart.system.parameter_derivative(point, current),
                )
                speed = max((abs(v) for v in tangent), default=mp.mpf(0))
            except (ZeroDivisionError, ValueError):
                speed = mp.inf
            motion_step = (
                (1 - switch_ratio) * max(abs(v) for v in point) / (2 * speed) if speed > 0 else step
            )
            trial_step = min(step, motion_step, abs(target - current))
            if trial_step < minimum and abs(target - current) > minimum:
                if switch_if_needed(force=True):
                    continue
                return finish(False, "projective tangent requires a segment below min_segment")
            next_t = current + direction * trial_step
            if float(next_t) == float(current):
                return finish(False, "parameter increment is below the tracker's resolution")
            path = track_path(
                chart.system, point, t_start=float(current), t_end=float(next_t), options=opts
            )
            segments.append(path)
            attempts += 1
            if path.success:
                point = tuple(path.endpoint)
                # The tracker takes binary64 parameter endpoints. Record the
                # parameter it actually reached, rather than a higher-precision
                # planning sum that could leave an untrackable sub-ulp remainder.
                current = target if path.final_t == float(target) else mp.mpf(str(path.final_t))
                step = min(mp.mpf(str(segment_size)), step * opts.step_growth)
                continue
            # An accepted prefix is a valid numerical checkpoint. Avoid treating
            # an unaccepted Newton trial as a projective path point.
            checkpoint = mp.mpf(str(path.final_t))
            if (
                path.accepted_steps
                and direction * (checkpoint - current) > 0
                and all(mp.isfinite(v) for v in path.endpoint)
            ):
                point = tuple(path.endpoint)
                current = checkpoint
                if switch_if_needed(force=True):
                    continue
            elif switch_if_needed(force=True):
                continue
            step /= 2
            if step < minimum:
                reason = (
                    "chart switch budget exhausted"
                    if len(switches) >= max_chart_switches
                    else path.message
                )
                return finish(False, "projective tracking could not resolve a segment: " + reason)
        switch_if_needed()
        return finish(True)
