"""Numerical Cauchy endgames and fixed-chart projective contiunation.

These APIs produce numerical evidence only. They do not certify
multiplicity, path completeness, or projective chart coverage.
"""

from dataclasses import dataclass, replace

import mpmath as mp
import sympy as sp

from .continuation import PathTrackerOptions, PathTrackingError, SympyHomotopy, track_path


@dataclass(frozen=True)
class EndgameResult:
    endpoint: tuple
    converged: bool
    cycle_number: int | None
    radii: tuple
    estimates: tuple
    error_estimate: object
    endpoint_residual: object = mp.inf
    status: str = "numerical"
    limitations: tuple = ("No rigourous endpoint enclosure or multiplicity certification.",)
    certificate: object = None
    deflation: object = None


class _Segment:
    def __init__(self, system, start, end):
        self.system = system
        self.variables = system.variables
        self.start, self.delta = start, end - start

    def evaluate(self, values, t):
        return self.system.evaluate(values, self.start + self.delta * t)

    def jacobian(self, values, t):
        return self.system.jacobian(values, self.start + self.delta * t)

    def parameter_derivative(self, values, t):
        return self.delta * self.system.parameter_derivative(values, self.start + self.delta * t)


def cauchy_endgame(
    system,
    start,
    *,
    target=1,
    radius=0.1,
    samples=32,
    max_cycle=8,
    levels=4,
    tolerance=None,
    options=None,
    certify=False,
    certification_box=None,
    deflate=False,
    max_quotient_dimension=256,
):
    """Estimate an isolated finite endpoint by cycle-aware Cauchy averaging.

    ``start`` must lie on the path at ``target-radius``. Circular parameter
    continuation visits complete Puiseux cycles before averaging. Failure to
    close a cycle or stabilize shrinking-radius averages returns no certificate.
    This method assumes an analytic homotopy with no other branch points inside
    the sampled disk. It cannot infer that assumption from numerical samples.
    """
    for name, value in (("samples", samples), ("max_cycle", max_cycle), ("levels", levels)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
    if not 0 < radius < 1 or samples < 8 or max_cycle < 1 or levels < 2:
        raise ValueError("invalid radius, sample count, cycle limit, or levels")
    if not isinstance(certify, bool) or not isinstance(deflate, bool):
        raise TypeError("certify and deflate must be Boolean")
    if deflate and not certify:
        raise ValueError("deflate requires certify=True")
    if certification_box is not None and not certify:
        raise ValueError("certification_box requires certify=True")
    if certify:
        from .certification import _positive_integer

        _positive_integer(max_quotient_dimension, "max_quotient_dimension")
    exact_target = sp.sympify(target)
    if certify and (exact_target.has(sp.Float) or exact_target.is_algebraic is not True):
        raise ValueError("endpoint certification requires an exact algebraic target parameter")
    opts = options or PathTrackerOptions()
    with mp.workdps(opts.max_digits):
        tol = (
            mp.mpf(str(tolerance))
            if tolerance is not None
            else mp.power(10, -opts.residual_digits // 2)
        )
        if tol <= 0:
            raise ValueError("tolerance must be positive")
        values = tuple(start)
        radii, estimates = [], []
        cycle_number = None
        error = mp.inf
        r = mp.mpf(str(radius))
        target = mp.mpc(target)
        for level in range(levels):
            base = target - r
            initial = tuple(values)
            circle_values = []
            previous = base
            closed = False
            for cycle in range(1, max_cycle + 1):
                for index in range(1, samples + 1):
                    next_t = target - r * mp.exp(2j * mp.pi * index / samples)
                    path = track_path(_Segment(system, previous, next_t), values, options=opts)
                    if not path.success:
                        raise PathTrackingError("Cauchy circle tracking failed: " + path.message)
                    values = tuple(path.endpoint)
                    circle_values.append(values)
                    previous = next_t
                scale = max(1, *(abs(v) for v in initial))
                if max(abs(a - b) for a, b in zip(values, initial, strict=True)) <= tol * scale:
                    cycle_number = cycle
                    closed = True
                    break
            if not closed:
                return EndgameResult(
                    tuple(values), False, None, tuple(radii), tuple(estimates), mp.inf
                )
            estimate = tuple(
                sum(row[i] for row in circle_values) / len(circle_values)
                for i in range(len(values))
            )
            radii.append(r)
            estimates.append(estimate)
            if len(estimates) > 1:
                error = max(abs(a - b) for a, b in zip(estimates[-2], estimate, strict=True))
                if error <= tol * max(1, *(abs(v) for v in estimate)):
                    residual = max(
                        (abs(v) for v in system.evaluate(estimate, target)), default=mp.mpf(0)
                    )
                    if residual <= tol:
                        result = EndgameResult(
                            estimate,
                            True,
                            cycle_number,
                            tuple(radii),
                            tuple(estimates),
                            error,
                            residual,
                        )
                        if certify:
                            result = _certify_endgame(
                                result,
                                system,
                                exact_target,
                                certification_box,
                                deflate,
                                max_quotient_dimension,
                                opts,
                            )
                        return result
            if level + 1 < levels:
                next_r = r / 2
                path = track_path(
                    _Segment(system, target - r, target - next_r), values, options=opts
                )
                if not path.success:
                    raise PathTrackingError("Cauchy radial tracking failed: " + path.message)
                values = tuple(path.endpoint)
                r = next_r
        return EndgameResult(
            estimates[-1], False, cycle_number, tuple(radii), tuple(estimates), error
        )


def _mp_rational(value):
    sign, mantissa, exponent, _ = mp.mpf(value)._mpf_
    return sp.Integer(-1 if sign else 1) * mantissa * sp.Integer(2) ** exponent


def _certify_endgame(result, system, target, box, deflate, max_dimension, options):
    from .certification import RationalComplexBox, RootCertificationError, certify_root_box
    from .deflation import deflate_isolated_root

    if not isinstance(system, SympyHomotopy):
        raise RootCertificationError("certification requires an explicit SympyHomotopy")
    centers = tuple((_mp_rational(mp.re(v)), _mp_rational(mp.im(v))) for v in result.endpoint)
    if box is None:
        width = sp.Rational(1, 10 ** max(4, options.residual_digits // 2))
        box = RationalComplexBox(
            tuple((r - width, r + width, i - width, i + width) for r, i in centers)
        )
    elif not isinstance(box, RationalComplexBox):
        box = RationalComplexBox(tuple(box))
    if len(box.bounds) != len(centers) or not all(
        b[0] < r < b[1] and b[2] < i < b[3] for b, (r, i) in zip(box.bounds, centers, strict=True)
    ):
        raise RootCertificationError(
            "the numerical endpoint is not inside the requested certification box"
        )
    equations = tuple(sp.expand(e.subs(system.parameter, target)) for e in system.expressions)
    certificate = certify_root_box(
        equations, system.variables, box, max_quotient_dimension=max_dimension
    )
    deflation = (
        deflate_isolated_root(
            equations, system.variables, certificate.point, max_quotient_dimension=max_dimension
        )
        if deflate
        else None
    )
    return replace(
        result,
        certificate=certificate,
        deflation=deflation,
        status="certified_endpoint",
        limitations=(
            "Exact certificate proves one target root in the recorded box; the numerical path is not certified.",
        ),
    )


@dataclass(frozen=True)
class ProjectiveHomotopy:
    system: SympyHomotopy
    affine_variables: tuple
    homogenizing_variable: object
    patch: tuple
    limitations: tuple = (
        "Fixed chart only; paths crossing the patch hyperplane require chart switching.",
        "Numerical infinity detection is not proof of affine completeness.",
    )

    def lift(self, affine_point):
        if len(affine_point) != len(self.affine_variables):
            raise ValueError("point dimension does not match")
        point = (*affine_point, sp.Integer(1))
        scale = sum(c * x for c, x in zip(self.patch, point, strict=True))
        if scale == 0:
            raise ValueError("point lies outside the fixed projective chart")
        return tuple(x / scale for x in point)

    def dehomogenize(self, point):
        if len(point) != len(self.system.variables):
            raise ValueError("point dimension does not match")
        if point[-1] == 0:
            raise ValueError("point is at infinity")
        return tuple(x / point[-1] for x in point[:-1])


def projective_homotopy(expressions, variables, parameter, *, patch):
    """Homogenize an analytic polynomial homotopy and impose a fixed linear patch.

    Track the returned ``system`` with ``track_path``. A patch with a nonzero
    affine component can follow some paths through affine infinity. Automatic
    chart switching and projective endpoint certification remain unsupported.
    """
    variables = tuple(variables)
    expressions = tuple(expressions)
    patch = tuple(map(sp.sympify, patch))
    if len(expressions) != len(variables) or len(patch) != len(variables) + 1:
        raise ValueError("square affine system and n+1 patch coefficients required")
    if not any(patch) or any(c.free_symbols or c.is_finite is not True for c in patch):
        raise ValueError("patch must be a nonzero constant vector")
    z = sp.Dummy("projective_z")
    homogeneous = []
    for expression in expressions:
        polynomial = sp.Poly(expression, *variables)
        degree = polynomial.total_degree()
        homogeneous.append(
            sum(
                coefficient
                * z ** (degree - sum(powers))
                * sp.prod(v**p for v, p in zip(variables, powers, strict=True))
                for powers, coefficient in polynomial.terms()
            )
        )
    all_variables = (*variables, z)
    homogeneous.append(sum(c * v for c, v in zip(patch, all_variables, strict=True)) - 1)
    return ProjectiveHomotopy(
        SympyHomotopy(homogeneous, all_variables, parameter), variables, z, patch
    )
