"""Adjoint pencils, anticanonical descent, and exact rational conics."""

import sympy as sp
from sympy.solvers.diophantine.diophantine import diop_ternary_quadratic

from ._curve_constants import ConstantCoordinates, extension_root
from ._curve_normalization import U, V, W, _kernel, _sub, adjoint_basis
from .curves import ConicModel, CurveError, CurveParametrization, CurveRationalityResult, PlaneCurve
from .domains import field_element, polynomial, polynomial_domain
from .function_fields import coefficient_field
from .quotient import QuotientAlgebra
from .subfields import field_normal_form


def validate_point(curve, point, domain):
    """Check exact projective coordinates, curve membership and smoothness."""
    if point is None or len(point) not in (2, 3):
        raise CurveError("point must have two affine or three projective coordinates")
    values = tuple(sp.sympify(c) for c in point)
    if any(c.has(sp.Float) or c.free_symbols for c in values):
        raise CurveError("point coordinates must be exact constants")
    values = tuple(domain.to_sympy(domain.convert(c)) for c in values)
    if len(values) == 2:
        values = (*values, sp.Integer(1))
    if not any(values):
        raise CurveError("the zero vector is not a projective point")
    x, y = curve.variables
    value = curve.homogeneous_polynomial.subs(
        dict(zip((x, y, W), values, strict=True)), simultaneous=True
    )
    if domain.convert(value) != domain.zero:
        raise CurveError("the supplied point is not on the curve")
    return values


def conic_point(curve):
    """Complete QQ conic decision, using exact ternary-quadratic descent."""
    if curve.domain != sp.QQ:
        raise CurveError(
            "automatic conic-point decisions require QQ; supply a point or request a constants extension"
        )
    # SymPy returns coordinates in ordered-symbol order, so use private names
    # with an explicit lexical order, independent of the user's variable names.
    a, b, c = sp.symbols("_curve_conic_0 _curve_conic_1 _curve_conic_2", integer=True)
    x, y = curve.variables
    Q = sp.Poly(_sub(curve.homogeneous_polynomial, {x: a, y: b, W: c}), a, b, c, domain=sp.QQ)
    _, Q = Q.clear_denoms(convert=True)
    solution = diop_ternary_quadratic(Q.as_expr())
    if solution is None:
        raise CurveError("the exact conic solver did not return a decision")
    if solution == (None, None, None):
        return None
    return validate_point(curve, solution, sp.QQ)


def _smooth_extension_point(curve):
    """Find a finite closed smooth point; the discriminant bounds the search."""
    x, y = curve.variables
    if curve.polynomial.degree(y) == 0:
        raise CurveError("a nonvertical model is required for a closed-point search")
    f = curve.polynomial.as_expr()
    discriminant = sp.Poly(sp.resultant(f, sp.diff(f, y), y), x, domain=curve.domain)
    leading = polynomial(f, (y,), curve.domain.frac_field(x)).LC()
    bound = int(discriminant.degree()) + int(sp.degree(leading, x)) + 1
    for a in range(bound + 1):
        fibre = sp.Poly(f.subs(x, a), y, domain=curve.domain)
        if fibre.degree() < 1 or fibre.gcd(fibre.diff()).degree() > 0:
            continue
        factors = sorted(fibre.factor_list()[1], key=lambda item: item[0].degree())
        E, root = extension_root(factors[0][0])
        return E, (sp.Integer(a), root, sp.Integer(1))
    raise ArithmeticError("closed-point search exhausted its discriminant degree bound")


def _make_parametrization(curve, domain, t, coordinates, inverse):
    field = polynomial_domain(domain.frac_field(t))
    coordinates = tuple(field.to_sympy(field_element(c, field)) for c in coordinates)
    denominator = sp.Poly(1, t, domain=domain)
    for c in coordinates:
        denominator = denominator.lcm(sp.Poly(sp.fraction(c)[1], t, domain=domain))
    polys = [
        sp.Poly(
            field.to_sympy(field_element(c, field) * field_element(denominator.as_expr(), field)),
            t,
            domain=domain,
        )
        for c in coordinates
    ]
    polys.append(denominator)
    gcd = polys[0].gcd(polys[1]).gcd(polys[2])
    values = tuple(p.exquo(gcd).as_expr() for p in polys)
    result = CurveParametrization(curve, domain, t, values, inverse)
    if not result.verify():
        raise ArithmeticError("constructed parametrization failed its birational identities")
    return result


def _parametrize_conic(curve, t, point, domain):
    point = validate_point(curve, point, domain)
    pivot = next(i for i, c in enumerate(point) if c != 0)
    others = tuple(i for i in range(3) if i != pivot)
    a, b = (point[i] / point[pivot] for i in others)
    s = sp.Dummy("conic_line_distance")
    line = [sp.Integer(1)] * 3
    line[others[0]] = a + s
    line[others[1]] = b + t * s
    x, y = curve.variables
    expression = curve.homogeneous_polynomial.subs(
        dict(zip((x, y, W), line, strict=True)), simultaneous=True
    )
    poly = polynomial(expression, (s,), polynomial_domain(domain.frac_field(t)))
    if poly.nth(0) != 0 or poly.degree() != 2:
        raise CurveError("a smooth irreducible conic is required")
    root = -poly.nth(1) / poly.nth(2)
    values = tuple(sp.cancel(c.subs(s, root)) for c in line)
    original = (x, y, sp.Integer(1))
    inverse = sp.cancel(
        (original[others[1]] - b * original[pivot]) / (original[others[0]] - a * original[pivot])
    )
    return _make_parametrization(
        curve, domain, t, (values[0] / values[2], values[1] / values[2]), inverse
    )


def _point_chart(curve, point, E):
    point = validate_point(curve, point, E)
    pivot = next(i for i, c in enumerate(point) if c != 0)
    other = tuple(i for i in range(3) if i != pivot)
    image = [sp.Integer(1)] * 3
    image[other[0]] = U + point[other[0]] / point[pivot]
    image[other[1]] = V + point[other[1]] / point[pivot]
    x, y = curve.variables
    local = _sub(curve.homogeneous_polynomial, dict(zip((x, y, W), image, strict=True)))
    zero = {U: 0, V: 0}
    if E.convert(sp.diff(local, V).subs(zero)) == 0:
        if E.convert(sp.diff(local, U).subs(zero)) == 0:
            raise CurveError("a supplied point must be smooth on the plane model")
        local = _sub(local, {U: V, V: U})
        image = [_sub(c, {U: V, V: U}) for c in image]
    return local, tuple(image)


def _adjoint_generator(curve, normalization, point=None):
    basis = adjoint_basis(curve, normalization.singularities, curve.degree - 2)
    if len(basis) != curve.degree - 1:
        raise ArithmeticError("genus-zero adjoint dimension is inconsistent")
    E = curve.domain
    if curve.degree > 3:
        E, point = _smooth_extension_point(curve) if point is None else (E, point)
        local, image = _point_chart(curve, point, E)
        order = curve.degree - 3
        # Formal implicit series V(U), truncated just far enough for the jets.
        series = sp.Integer(0)
        derivative = E.convert(sp.diff(local, V).subs({U: 0, V: 0}))
        for k in range(1, order):
            coefficient = sp.Poly(sp.expand(local.subs(V, series)), U, domain=E).nth(k)
            coefficient = E.to_sympy(-E.convert(coefficient) / derivative)
            series += coefficient * U**k
        x, y = curve.variables
        local_forms = [
            _sub(H, dict(zip((x, y, W), image, strict=True))).subs(V, series) for H in basis
        ]
        rows = [tuple(sp.Poly(H, U, domain=E).nth(k) for H in local_forms) for k in range(order)]
        kernel = _kernel(rows, len(basis), E)
        basis = tuple(
            sp.expand(sum(c * H for c, H in zip(row, basis, strict=True))) for row in kernel
        )
    elif point is not None:
        _point_chart(curve, point, E)
    if len(basis) != 2:
        raise ArithmeticError("adjoint point jets did not produce a degree-one pencil")
    return E, sp.cancel(basis[1].subs(W, 1) / basis[0].subs(W, 1))


def _from_generator(curve, t, domain, generator):
    A = curve.function_field(domain=domain)
    image = coefficient_field([generator], ambient=A, constants=domain, generator_symbols=(t,))
    coordinates = tuple(image.express(v).expression for v in curve.variables)
    if any(c is None for c in coordinates):
        raise ArithmeticError("adjoint pencil did not generate the entire function field")
    return _make_parametrization(curve, domain, t, coordinates, generator)


class _RelativeFunctions:
    """Expand E(C) as a constant extension of K(C), including rational coefficients."""

    def __init__(self, curve, E):
        self.curve = curve
        self.constants = ConstantCoordinates(E, curve.domain)
        self.ambient = curve.function_field(domain=E)
        self.y = self.ambient.variables[0]
        self.x = self.ambient.domain.symbols[0]
        self.s = sp.Dummy("relative_constant")
        self.extension = QuotientAlgebra.from_polynomials(
            [self.constants.polynomial(self.s)], (self.s,), domain=curve.domain.frac_field(self.x)
        )

    def _polynomial(self, expression):
        poly = sp.Poly(expression, self.x, domain=self.constants.extension)
        return sum(
            sum(c * self.s**i for i, c in enumerate(self.constants.coordinates(value)))
            * self.x ** degree[0]
            for degree, value in poly.terms()
        )

    def coordinates(self, expression):
        expression = field_normal_form(self.ambient, expression)
        result = [sp.Integer(0)] * self.constants.degree
        for degree, coefficient in polynomial(expression, (self.y,), self.ambient.domain).terms():
            numerator, denominator = sp.fraction(sp.cancel(coefficient))
            relative = field_normal_form(
                self.extension, self._polynomial(numerator) / self._polynomial(denominator)
            )
            relative = polynomial(relative, (self.s,), self.extension.domain)
            for i in range(self.constants.degree):
                result[i] += relative.nth(i) * self.y ** degree[0]
        return tuple(sp.cancel(c) for c in result)

    def base_element(self, expression):
        coordinates = self.coordinates(expression)
        if any(c != 0 for c in coordinates[1:]):
            raise ArithmeticError("anticanonical map did not descend to the ground constants")
        return coordinates[0]


def _conic_from_generator(curve, E, generator):
    """Descend the three regular vector fields of P1 to the ground curve.

    In a rational parameter t they are (1,t,t**2)d/dt. In the x coordinate their
    coefficients are (1,t,t**2)/(dt/dx). Constant descent is exact linear algebra.
    The double-zero vector fields give the ground-field anticanonical conic.
    """
    relative = _RelativeFunctions(curve, E)
    A = relative.ambient
    x, y = relative.x, relative.y
    f = curve.polynomial.as_expr()
    derivative = sp.diff(generator, x) - sp.diff(generator, y) * sp.diff(f, x) / sp.diff(f, y)
    functions = [field_normal_form(A, generator**i / derivative) for i in range(3)]
    m = relative.constants.degree
    theta = relative.constants.theta
    columns = [
        relative.coordinates(function * theta**j) for function in functions for j in range(m)
    ]
    rows = []
    Kx = polynomial_domain(curve.domain.frac_field(x))
    for k in range(1, m):
        for power in range(A.dimension):
            values = [polynomial(column[k], (y,), Kx).nth(power) for column in columns]
            denominator = sp.Poly(1, x, domain=curve.domain)
            for value in values:
                denominator = denominator.lcm(
                    sp.Poly(sp.fraction(sp.cancel(value))[1], x, domain=curve.domain)
                )
            polys = [
                sp.Poly(
                    Kx.to_sympy(
                        field_element(value, Kx) * field_element(denominator.as_expr(), Kx)
                    ),
                    x,
                    domain=curve.domain,
                )
                for value in values
            ]
            degree = max((p.degree() for p in polys), default=0)
            if degree is sp.S.NegativeInfinity:
                continue
            rows.extend(tuple(p.nth(j) for p in polys) for j in range(int(degree) + 1))
    kernel = _kernel(rows, 3 * m, curve.domain)
    if len(kernel) != 3:
        raise ArithmeticError("anticanonical descent did not have dimension three")
    vectors = [
        tuple(sum(row[i * m + j] * theta**j for j in range(m)) for i in range(3)) for row in kernel
    ]
    matrix = sp.Matrix.hstack(*(sp.Matrix(v) for v in vectors)).to_DM(domain=E)
    inverse = matrix.inv().to_Matrix()
    image = inverse * sp.Matrix([generator**2, -2 * generator, 1])
    denominator = next(c for c in reversed(image) if field_normal_form(A, c) != 0)
    affine = [relative.base_element(c / denominator) for c in image]
    # Remove the coordinate identically one to choose an affine conic chart.
    pivot = next(i for i, c in reversed(list(enumerate(affine))) if c == 1)
    forward = tuple(c for i, c in enumerate(affine) if i != pivot)
    u, v = sp.Dummy("conic_u"), sp.Dummy("conic_v")
    base_ambient = curve.function_field()
    field = coefficient_field(
        forward, ambient=base_ambient, constants=curve.domain, generator_symbols=(u, v)
    )
    if len(field.relations) != 1 or sp.Poly(field.relations[0], u, v).total_degree() != 2:
        raise ArithmeticError("anticanonical image is not a plane conic")
    conic = PlaneCurve.from_polynomial(field.relations[0], (u, v), domain=curve.domain)
    inverse = tuple(field.express(var).expression for var in curve.variables)
    if any(c is None for c in inverse):
        raise ArithmeticError("anticanonical conic map is not birational")
    result = ConicModel(curve, conic, forward, inverse)
    if not result.verify():
        raise ArithmeticError("conic model failed birational replay")
    return result


def _compose_conic(model, parametrization, t):
    substitutions = dict(
        zip(model.conic.variables, parametrization.affine_coordinates, strict=True)
    )
    coordinates = tuple(c.subs(substitutions, simultaneous=True) for c in model.inverse)
    inverse = parametrization.inverse.subs(
        dict(zip(model.conic.variables, model.forward, strict=True)), simultaneous=True
    )
    return _make_parametrization(model.source, parametrization.domain, t, coordinates, inverse)


def _conic_result(normalization, model, t, extend_constants, point=None):
    curve = model.conic
    if point is not None:
        point = validate_point(curve, point, curve.domain)
    elif curve.domain == sp.QQ:
        point = conic_point(curve)
    elif not extend_constants:
        raise CurveError(
            "automatic conic-point decisions require QQ; supply a point or request a constants extension"
        )
    if point is not None:
        param = _compose_conic(model, _parametrize_conic(curve, t, point, curve.domain), t)
        return CurveRationalityResult(normalization, "rational", param, model)
    E, point = _smooth_extension_point(curve)
    if curve.domain == E:
        # A found witness is valid over any supported constants, including when
        # no automatic number-field decision procedure was requested.
        param = _compose_conic(model, _parametrize_conic(curve, t, point, E), t)
        return CurveRationalityResult(normalization, "rational", param, model)
    param = None
    if extend_constants:
        param = _compose_conic(model, _parametrize_conic(curve, t, point, E), t)
    status = "rational_over_extension" if param is not None else "no_ground_field_point"
    return CurveRationalityResult(normalization, status, param, model, E, point)


def rational_parametrization(curve, t, point, extend_constants, max_steps, max_dimension):
    """Construct a checked rational parametrization with its inverse map."""
    t = sp.Dummy("curve_parameter") if t is None else t
    if not isinstance(t, sp.Symbol) or t in curve.variables:
        raise CurveError("parameter must be a fresh Symbol")
    if not isinstance(extend_constants, bool):
        raise CurveError("extend_constants must be Boolean")
    normalization = curve.normalization(max_steps=max_steps, max_dimension=max_dimension)
    if normalization.genus > 0:
        return CurveRationalityResult(normalization, "positive_genus")
    if curve.degree == 1:
        x, y = curve.variables
        f = curve.polynomial
        a, b = f.coeff_monomial(x), f.coeff_monomial(y)
        if b != 0:
            coordinates = (t, -(a * t + f.TC()) / b)
            inverse = x
        else:
            coordinates = (-f.TC() / a, t)
            inverse = y
        if point is not None:
            validate_point(curve, point, curve.domain)
        return CurveRationalityResult(
            normalization,
            "rational",
            _make_parametrization(curve, curve.domain, t, coordinates, inverse),
        )
    if curve.degree == 2:
        # Keep an explicit identity conic model, so negative decisions carry the
        # same replayable obstruction interface as higher-degree plane models.
        u, v = sp.Dummy("conic_u"), sp.Dummy("conic_v")
        conic = PlaneCurve.from_polynomial(
            curve.polynomial.as_expr().xreplace(dict(zip(curve.variables, (u, v), strict=True))),
            (u, v),
            domain=curve.domain,
        )
        model = ConicModel(curve, conic, curve.variables, (u, v))
        return _conic_result(normalization, model, t, extend_constants, point)
    E, generator = _adjoint_generator(curve, normalization, point)
    if curve.domain == E:
        return CurveRationalityResult(
            normalization, "rational", _from_generator(curve, t, E, generator)
        )
    model = _conic_from_generator(curve, E, generator)
    return _conic_result(normalization, model, t, extend_constants)
