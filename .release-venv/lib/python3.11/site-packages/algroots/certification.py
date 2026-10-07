"""Exact isolated-root certificates and rational complex-box isolation.

The proof substrate is a characteristic-zero finite quotient, not a small
numerical residual. No perturbed-system or numerical path certificate is implied.
"""

from dataclasses import dataclass

import sympy as sp
from sympy.polys.matrices import DomainMatrix

from ._validation import validate_integer as _positive_integer
from ._validation import validate_variables
from .errors import PolynomialSystemError
from .quotient import QuotientAlgebra
from .rational_univariate.construction import rur_from_quotient
from .solver import _normalize_equations


class RootCertificationError(PolynomialSystemError):
    """The requested exact certificate could not be established."""


@dataclass(frozen=True)
class RationalComplexBox:
    """Open coordinate box: each entry is (real_lo, real_hi, imag_lo, imag_hi)."""

    bounds: tuple

    def __post_init__(self):
        bounds = tuple(tuple(map(sp.sympify, row)) for row in self.bounds)
        if not bounds or any(len(row) != 4 for row in bounds):
            raise ValueError("one four-endpoint rectangle is required per coordinate")
        if any(v.is_Rational is not True for row in bounds for v in row):
            raise ValueError("box endpoints must be exact rationals")
        if any(row[0] >= row[1] or row[2] >= row[3] for row in bounds):
            raise ValueError("rectangle endpoints must be strictly ordered")
        object.__setattr__(self, "bounds", bounds)


@dataclass(frozen=True)
class IsolatedRootCertificate:
    equations: tuple
    variables: tuple
    point: tuple
    groebner_basis: tuple
    quotient_dimension: int
    geometric_solution_count: int
    jacobian_rank: int
    multiplicity: int
    separator_coefficients: tuple
    separator_value: object
    characteristic_polynomial: object
    box: RationalComplexBox | None = None
    status: str = "certified"
    proof_basis: tuple = (
        "exact_equation_membership",
        "finite_quotient_isolation",
        "exact_jacobian_rank",
        "separating_action_local_multiplicity",
    )

    @property
    def is_singular(self):
        return self.jacobian_rank < len(self.variables)

    def verify(self, *, max_quotient_dimension=256, max_refinements=64):
        """Replay from the original equations, detecting altered certificate fields."""
        try:
            if self.box is None:
                replay = certify_isolated_root(
                    self.equations,
                    self.variables,
                    self.point,
                    max_quotient_dimension=max_quotient_dimension,
                )
            else:
                replay = certify_root_box(
                    self.equations,
                    self.variables,
                    self.box,
                    max_quotient_dimension=max_quotient_dimension,
                    max_refinements=max_refinements,
                )
            return self == replay
        except (PolynomialSystemError, ValueError, TypeError, NotImplementedError):
            return False


def _point_field(quotient, point):
    if len(point) != len(quotient.variables):
        raise RootCertificationError("point dimension does not match")
    point = tuple(map(sp.sympify, point))
    if any(v.free_symbols or v.has(sp.Float) or v.is_algebraic is not True for v in point):
        raise RootCertificationError("point must have exact algebraic coordinates, with no floats")
    extensions = [v for v in point if v.is_Rational is not True]
    roots = set().union(*(v.atoms(sp.CRootOf) for v in point))
    if len(roots) == 1:
        root = next(iter(roots))
        parameter = sp.Dummy("point_field_t")
        try:
            maps = [sp.Poly(v.xreplace({root: parameter}), parameter) for v in point]
            if all(
                coefficient.is_Rational
                for polynomial in maps
                for coefficient in polynomial.all_coeffs()
            ):
                # RUR coordinates share one primitive root. Rebuilding a
                # primitive element from every coordinate is unnecessary.
                extensions = [root]
        except sp.PolynomialError:
            pass
    if quotient.domain.is_AlgebraicField:
        extensions.insert(0, quotient.domain.ext.as_expr())
    field = sp.QQ.algebraic_field(*extensions) if extensions else sp.QQ
    return point, field


def _field_value(value, field):
    if field.is_AlgebraicField and isinstance(field.ext.as_expr(), sp.CRootOf):
        root = field.ext.as_expr()
        parameter = sp.Dummy("field_value_t")
        try:
            polynomial = sp.Poly(value.xreplace({root: parameter}), parameter, domain=sp.QQ)
            result = field.zero
            for coefficient in polynomial.all_coeffs():
                result = result * field.unit + field.from_sympy(coefficient)
            return result
        except sp.polys.polyerrors.BasePolynomialError:
            pass
    return field.from_sympy(value)


def _evaluate(expression, variables, point, field, values=None):
    polynomial = sp.Poly(expression, *variables, domain=field)
    if values is None:
        values = tuple(_field_value(v, field) for v in point)
    result = field.zero
    for powers, coefficient in polynomial.terms():
        term = field.from_sympy(coefficient)
        for value, power in zip(values, powers, strict=True):
            term *= value**power
        result += term
    return result


def _jacobian_data(equations, variables, point, field, values=None):
    jacobian = sp.Matrix(equations).jacobian(variables)
    rows = [
        [_evaluate(entry, variables, point, field, values) for entry in row]
        for row in jacobian.tolist()
    ]
    matrix = DomainMatrix(rows, (len(equations), len(variables)), field)
    return jacobian, matrix


def _certificate(quotient, equations, point, *, separator=None, box=None, characteristic=None):
    point, field = _point_field(quotient, point)
    values = tuple(_field_value(v, field) for v in point)
    if any(_evaluate(e, quotient.variables, point, field, values) != field.zero for e in equations):
        raise RootCertificationError("the exact candidate does not satisfy every input equation")
    _, jacobian = _jacobian_data(equations, quotient.variables, point, field, values)
    if separator is None:
        name = "_certificate_t"
        while name in {str(v) for v in quotient.variables}:
            name = "_" + name
        separator = quotient.separating_element(sp.Symbol(name))
    parameter = separator.defining_polynomial.gens[0]
    if characteristic is None:
        characteristic = sp.Poly(
            quotient.linear_combination_matrix(separator.coefficients)
            .charpoly(parameter)
            .as_expr(),
            parameter,
            domain=quotient.domain,
        )
    alpha = sum(c * v for c, v in zip(separator.coefficients, point, strict=True))
    derivative = characteristic
    multiplicity = 0
    while _evaluate(derivative.as_expr(), (parameter,), (alpha,), field) == field.zero:
        multiplicity += 1
        derivative = derivative.diff()
    if not 1 <= multiplicity <= quotient.dimension:
        raise RootCertificationError("separating action failed to establish local multiplicity")
    return IsolatedRootCertificate(
        tuple(equations),
        quotient.variables,
        point,
        tuple(p.as_expr() for p in quotient.groebner_basis.polys),
        quotient.dimension,
        separator.geometric_solution_count,
        jacobian.rank(),
        multiplicity,
        separator.coefficients,
        alpha,
        characteristic,
        box,
    )


def certify_isolated_root(equations, variables, point, *, max_quotient_dimension=256):
    """Prove exact membership, isolation, Jacobian rank and local multiplicity.

    Accepts exact rational/algebraic coefficients and coordinates. The whole
    input ideal must be zero-dimensional. This deliberately refuses floating
    candidates and positive-dimensional ideals, even if a local point looks isolated.
    """
    _positive_integer(max_quotient_dimension, "max_quotient_dimension")
    variables = validate_variables(variables)
    equations = _normalize_equations(equations, variables)
    quotient = QuotientAlgebra.from_polynomials(
        equations, variables, max_dimension=max_quotient_dimension
    )
    if quotient.dimension == 0:
        raise RootCertificationError("the unit ideal has no root to certify")
    return _certificate(quotient, equations, tuple(point))


def _interval_product(a, b):
    products = [u * v for u in a for v in b]
    return min(products), max(products)


def _rectangle_product(a, b):
    ac, bd = _interval_product(a[:2], b[:2]), _interval_product(a[2:], b[2:])
    ad, bc = _interval_product(a[:2], b[2:]), _interval_product(a[2:], b[:2])
    return ac[0] - bd[1], ac[1] - bd[0], ad[0] + bc[0], ad[1] + bc[1]


def _polynomial_rectangle(polynomial, rectangle):
    value = (sp.Integer(0),) * 4
    for coefficient in polynomial.all_coeffs():
        value = _rectangle_product(value, rectangle)
        value = (value[0] + coefficient, value[1] + coefficient, value[2], value[3])
    return value


def _root_rectangle(root, refinement):
    if root.is_Rational:
        return root, root, sp.Integer(0), sp.Integer(0)
    if not isinstance(root, sp.CRootOf):
        raise RootCertificationError(
            "rational-box proof requires an isolated rational-polynomial root"
        )
    # CRootOf's isolating intervals/rectangles carry exact rational endpoints.
    interval = root._get_interval().refine_size(sp.Rational(1, 2**refinement))
    if root.is_real:
        return sp.Rational(interval.a), sp.Rational(interval.b), sp.Integer(0), sp.Integer(0)
    return tuple(sp.Rational(v) for v in (interval.ax, interval.bx, interval.ay, interval.by))


def certify_root_box(equations, variables, box, *, max_quotient_dimension=256, max_refinements=64):
    """Prove exactly one geometric root lies in an open rational complex box.

    Currently restricted to rational coefficients. The separator image must
    contain exactly one RUR parameter root, counted by exact complex root
    isolation. Rational rectangle arithmetic proves every coordinate lies
    strictly inside the input box. A failed/coarse box is rejected, not guessed.
    """
    _positive_integer(max_quotient_dimension, "max_quotient_dimension")
    _positive_integer(max_refinements, "max_refinements")
    variables = validate_variables(variables)
    equations = _normalize_equations(equations, variables)
    box = box if isinstance(box, RationalComplexBox) else RationalComplexBox(tuple(box))
    if len(box.bounds) != len(variables):
        raise RootCertificationError("box dimension does not match")
    quotient = QuotientAlgebra.from_polynomials(
        equations, variables, max_dimension=max_quotient_dimension
    )
    return _certify_box_from_quotient(quotient, equations, box, max_refinements=max_refinements)


def _certify_box_from_quotient(
    quotient,
    equations,
    box,
    *,
    max_refinements=64,
    separator=None,
    representation=None,
    characteristic=None,
    parameter_roots=None,
):
    """Internal batched proof path; the quotient is built from these equations."""
    variables = quotient.variables
    if quotient.domain != sp.QQ:
        raise RootCertificationError(
            "rational-box isolation currently requires rational coefficients"
        )
    if quotient.dimension == 0:
        raise RootCertificationError("the box cannot contain a root of the unit ideal")
    name = "_certificate_t"
    while name in {str(v) for v in variables}:
        name = "_" + name
    parameter = sp.Symbol(name)
    if separator is None:
        separator = quotient.separating_element(parameter)
    else:
        parameter = separator.defining_polynomial.gens[0]
    image = [sp.Integer(0)] * 4
    for c, bounds in zip(separator.coefficients, box.bounds, strict=True):
        image = [
            u + v for u, v in zip(image, _rectangle_product((c, c, 0, 0), bounds), strict=True)
        ]
    count = separator.defining_polynomial.count_roots(
        image[0] + sp.I * image[2], image[1] + sp.I * image[3]
    )
    if count != 1:
        raise RootCertificationError(
            f"separator image contains {count} distinct roots; tighten the box"
        )
    if representation is None:
        representation = rur_from_quotient(quotient, parameter)
    if representation.separating_linear_form != separator.expression:
        raise RootCertificationError("RUR separator changed during isolation")
    maps = representation.normalized_coordinate_polynomials()
    if parameter_roots is None:
        parameter_roots = representation.defining_polynomial.all_roots(radicals=False)
    for root in parameter_roots:
        for refinement in range(1, max_refinements + 1):
            rectangle = _root_rectangle(root, refinement)
            if (
                rectangle[1] < image[0]
                or rectangle[0] > image[1]
                or rectangle[3] < image[2]
                or rectangle[2] > image[3]
            ):
                break
            enclosures = tuple(_polynomial_rectangle(poly, rectangle) for poly in maps)
            if all(
                bound[0] < enclosed[0]
                and enclosed[1] < bound[1]
                and bound[2] < enclosed[2]
                and enclosed[3] < bound[3]
                for bound, enclosed in zip(box.bounds, enclosures, strict=True)
            ):
                point = tuple(sp.expand(poly.as_expr().subs(parameter, root)) for poly in maps)
                return _certificate(
                    quotient,
                    equations,
                    point,
                    separator=separator,
                    box=box,
                    characteristic=characteristic,
                )
    raise RootCertificationError(
        "coordinate enclosure could not prove a root strictly inside the box"
    )


@dataclass(frozen=True)
class RootCertificationAttempt:
    """Aligned per-root proof result; failures are never numerical certificates."""

    status: str
    certificate: IsolatedRootCertificate | None = None
    error: str | None = None
    box_attempts: int = 0


def certify_numerical_roots(
    result, *, max_quotient_dimension=128, max_refinements=128, required=False, max_box_attempts=16
):
    return _certify_numerical_roots(
        result,
        max_quotient_dimension=max_quotient_dimension,
        max_refinements=max_refinements,
        required=required,
        max_box_attempts=max_box_attempts,
    )


def _certify_numerical_roots(
    result,
    *,
    max_quotient_dimension=128,
    max_refinements=128,
    required=False,
    max_box_attempts=16,
    quotient=None,
    context=None,
    neighbour_points=(),
):
    """Automatically construct rational boxes and batch exact endpoint proofs.

    All roots share one original-system quotient, separator, RUR and
    characteristic polynomial. This certifies endpoints, never numerical paths.
    """
    _positive_integer(max_quotient_dimension, "max_quotient_dimension")
    _positive_integer(max_refinements, "max_refinements")
    _positive_integer(max_box_attempts, "max_box_attempts")
    if not isinstance(required, bool):
        raise ValueError("required must be Boolean")
    if not result.roots:
        return ()
    try:
        if quotient is None:
            quotient = QuotientAlgebra.from_polynomials(
                result.equations, result.variables, max_dimension=max_quotient_dimension
            )
        elif quotient.dimension > max_quotient_dimension:
            raise RootCertificationError("quotient exceeds certification dimension budget")
        if quotient.domain != sp.QQ:
            raise RootCertificationError(
                "automatic box certification requires rational coefficients"
            )
        cached = (
            context.get("kernels")
            if context is not None and context.get("quotient") is quotient
            else None
        )
        if cached is not None:
            separator, representation, parameter_roots, characteristic = cached
        else:
            name = "_certificate_t"
            while name in {str(v) for v in result.variables}:
                name = "_" + name
            parameter = sp.Symbol(name)
            separator = quotient.separating_element(parameter)
            representation = rur_from_quotient(quotient, parameter)
            parameter_roots = representation.defining_polynomial.all_roots(radicals=False)
            characteristic = sp.Poly(
                quotient.linear_combination_matrix(separator.coefficients)
                .charpoly(parameter)
                .as_expr(),
                parameter,
                domain=quotient.domain,
            )
            if context is not None:
                context.clear()
                context.update(
                    quotient=quotient,
                    kernels=(separator, representation, parameter_roots, characteristic),
                )
    except (PolynomialSystemError, ValueError, TypeError, NotImplementedError) as exc:
        if required:
            raise RootCertificationError(f"automatic certification unavailable: {exc}") from exc
        return tuple(RootCertificationAttempt("unavailable", error=str(exc)) for _ in result.roots)
    radius = sp.Rational(
        1, 10 ** max(6, min(12, result.verification_digits, result.precision_digits // 2))
    )
    attempts = []
    centers = tuple(
        tuple((sp.Rational(sp.re(c)), sp.Rational(sp.im(c))) for c in root) for root in result.roots
    )
    neighbours = centers + tuple(
        tuple((sp.Rational(sp.re(c)), sp.Rational(sp.im(c))) for c in point)
        for point in neighbour_points
    )
    for index, _root in enumerate(result.roots):
        # Neighbour distances only propose a box. Exact separator counts and
        # coordinate isolation remain the sole proof of membership/uniqueness.
        distances = [
            max(
                max(abs(a - c), abs(b - d))
                for (a, b), (c, d) in zip(centers[index], other, strict=True)
            )
            for j, other in enumerate(neighbours)
            if j != index
        ]
        widths = [radius * max(1, abs(re), abs(im)) for re, im in centers[index]]
        if (
            len(neighbours) == separator.geometric_solution_count
            and distances
            and min(distances) > 0
        ):
            widths = [min(width, min(distances) / 8) for width in widths]
        error = None
        for attempt in range(1, max_box_attempts + 1):
            box = RationalComplexBox(
                tuple(
                    (re - width, re + width, im - width, im + width)
                    for (re, im), width in zip(centers[index], widths, strict=True)
                )
            )
            try:
                certificate = _certify_box_from_quotient(
                    quotient,
                    result.equations,
                    box,
                    max_refinements=max_refinements,
                    separator=separator,
                    representation=representation,
                    characteristic=characteristic,
                    parameter_roots=parameter_roots,
                )
                attempts.append(
                    RootCertificationAttempt("certified", certificate, box_attempts=attempt)
                )
                break
            except (PolynomialSystemError, ValueError, TypeError, NotImplementedError) as exc:
                error = str(exc)
                # Tighten on ambiguous images; widen a zero-count image to
                # compensate for an overly optimistic numerical proposal.
                factor = 2 if "contains 0 distinct roots" in error else sp.Rational(1, 16)
                widths = [width * factor for width in widths]
        else:
            if required:
                raise RootCertificationError(f"automatic root certification failed: {error}")
            attempts.append(
                RootCertificationAttempt("failed", error=error, box_attempts=max_box_attempts)
            )
    return tuple(attempts)
