"""Projective plane-curve resolution and adjoint linear systems."""

import sympy as sp

from ._curve_constants import ConstantCoordinates, extension_root
from .errors import QuotientAlgebraError
from .quotient import QuotientAlgebra
from .rational_univariate.construction import rur_from_quotient

# Shared identities keep replay records comparable; none can capture user Symbols.
U, V, W = (sp.Dummy(n) for n in ("curve_local_u", "curve_local_v", "curve_homogenizer"))


def _sub(expression, values):
    return sp.expand(expression.subs(values, simultaneous=True))


def _polynomial(expression, domain):
    return sp.Poly(expression, U, V, domain=domain).monic().as_expr()


def _order(expression, domain):
    poly = sp.Poly(expression, U, V, domain=domain)
    return min(sum(monomial) for monomial in poly.monoms())


def _exceptional_points(expression, domain, exceptional=U):
    """Closed singular points on an exceptional line, one per conjugacy orbit."""
    variable = V if exceptional == U else U
    equations = [expression, sp.diff(expression, U), sp.diff(expression, V)]
    gcd = sp.Poly(0, variable, domain=domain)
    for eq in equations:
        gcd = gcd.gcd(sp.Poly(eq.subs(exceptional, 0), variable, domain=domain))
    if gcd.is_zero:
        raise ArithmeticError("strict transform contains an exceptional component")
    for factor, _ in gcd.sqf_part().factor_list()[1]:
        E, root = extension_root(factor)
        yield E, root, int(factor.degree())


def _affine_singularities(curve, max_dimension):
    x, y = curve.variables
    f = curve.polynomial.as_expr()
    try:
        A = QuotientAlgebra.from_polynomials(
            [f, sp.diff(f, x), sp.diff(f, y)],
            (x, y),
            domain=curve.domain,
            max_dimension=max_dimension,
        )
    except QuotientAlgebraError as exc:
        from .curves import CurveComputationLimitError

        raise CurveComputationLimitError(
            f"could not construct finite singular locus: {exc}"
        ) from exc
    if A.dimension == 0:
        return
    T = sp.Dummy("singular_separator")
    n = A.geometric_solution_count
    rur = rur_from_quotient(A, T, max_separating_attempts=n * (n - 1) // 2 + 1)
    for factor, _ in rur.defining_polynomial.factor_list()[1]:
        E, root = extension_root(factor)
        denominator = E.convert(rur.coordinate_denominator.as_expr().subs(T, root))
        values = tuple(
            E.to_sympy(E.convert(p.as_expr().subs(T, root)) / denominator)
            for p in rur.coordinate_numerators
        )
        yield E, values, int(factor.degree())


def _initial_points(curve, max_dimension):
    x, y = curve.variables
    f = curve.polynomial.as_expr()
    for E, (a, b), weight in _affine_singularities(curve, max_dimension):
        yield (
            _polynomial(_sub(f, {x: U + a, y: V + b}), E),
            (U + a, V + b, sp.Integer(1)),
            E,
            weight,
            "affine",
        )
    # Disjoint infinity charts: [1:a:0], then the remaining [0:1:0].
    F = curve.homogeneous_polynomial
    chart = _sub(F, {x: 1, y: V, W: U})
    for E, a, weight in _exceptional_points(chart, curve.domain):
        yield (
            _polynomial(_sub(chart, {V: V + a}), E),
            (sp.Integer(1), V + a, U),
            E,
            weight,
            "infinity_x",
        )
    chart = _sub(F, {x: U, y: 1, W: V})
    if all(eq.subs({U: 0, V: 0}) == 0 for eq in (chart, sp.diff(chart, U), sp.diff(chart, V))):
        yield _polynomial(chart, curve.domain), (U, sp.Integer(1), V), curve.domain, 1, "infinity_y"


def resolve(curve, max_steps, max_dimension):
    from .curves import CurveComputationLimitError, ResolutionPoint

    records = []

    def visit(f, projective, multiplier, E, weight, chart, depth):
        m = _order(f, E)
        if m < 2:
            return
        if len(records) >= max_steps:
            raise CurveComputationLimitError("singular-point resolution exceeded max_steps")
        records.append(ResolutionPoint(f, projective, multiplier, E, weight, m, depth, chart))
        substitution = {U: U, V: U * V}
        transform = _polynomial(sp.cancel(_sub(f, substitution) / U**m), E)
        image = tuple(_sub(c, substitution) for c in projective)
        factor = _sub(multiplier, substitution) * U ** (m - 1)
        for L, root, relative_degree in _exceptional_points(transform, E):
            shift = {V: V + root}
            visit(
                _polynomial(_sub(transform, shift), L),
                tuple(_sub(c, shift) for c in image),
                _sub(factor, shift),
                L,
                weight * relative_degree,
                chart,
                depth + 1,
            )
        # The vertical tangent direction is the only point not in the first chart.
        substitution = {U: U * V, V: V}
        transform = _polynomial(sp.cancel(_sub(f, substitution) / V**m), E)
        if all(
            eq.subs({U: 0, V: 0}) == 0
            for eq in (transform, sp.diff(transform, U), sp.diff(transform, V))
        ):
            image = tuple(_sub(c, substitution) for c in projective)
            factor = _sub(multiplier, substitution) * V ** (m - 1)
            visit(transform, image, factor, E, weight, chart, depth + 1)

    for f, image, E, weight, chart in _initial_points(curve, max_dimension):
        visit(f, image, sp.Integer(1), E, weight, chart, 0)
    return tuple(records)


def _kernel(rows, size, domain):
    if not rows:
        return tuple(tuple(sp.Integer(i == j) for i in range(size)) for j in range(size))
    matrix = sp.Matrix(rows).to_DM(domain=domain)
    return tuple(tuple(domain.to_sympy(c) for c in row) for row in matrix.nullspace().to_list())


def adjoint_basis(curve, points, degree):
    """Homogeneous adjoints, including all infinitely-near conditions."""
    if degree < 0:
        return ()
    x, y = curve.variables
    basis = tuple(
        x**i * y**j * W ** (degree - i - j)
        for i in range(degree, -1, -1)
        for j in range(degree - i, -1, -1)
    )
    for point in points:
        coordinates = ConstantCoordinates(point.domain, curve.domain)
        local = []
        for H in basis:
            pullback = _sub(H, dict(zip((x, y, W), point.projective_coordinates, strict=True)))
            local.append(
                sp.Poly(sp.cancel(pullback / point.adjoint_multiplier), U, V, domain=point.domain)
            )
        rows = []
        for i in range(point.multiplicity - 1):
            for j in range(point.multiplicity - 1 - i):
                entries = [coordinates.coordinates(h.coeff_monomial(U**i * V**j)) for h in local]
                rows.extend(tuple(c[k] for c in entries) for k in range(coordinates.degree))
        vectors = _kernel(rows, len(basis), curve.domain)
        basis = tuple(
            sp.expand(sum(c * h for c, h in zip(row, basis, strict=True))) for row in vectors
        )
        if not basis:
            break
    return basis


def normalize(curve, max_steps, max_dimension):
    from .curves import CurveError, CurveNormalization

    points = resolve(curve, max_steps, max_dimension)
    d = curve.degree
    arithmetic = (d - 1) * (d - 2) // 2
    delta = sum(p.residue_degree * p.multiplicity * (p.multiplicity - 1) // 2 for p in points)
    genus = arithmetic - delta
    canonical = adjoint_basis(curve, points, d - 3)
    # For a geometrically reducible plane curve this difference is r-1, where
    # r is the number of geometric components. This also detects enlarged constants.
    if genus != len(canonical):
        raise CurveError("curve is not geometrically integral over the declared constants")
    return CurveNormalization(
        curve, genus, arithmetic, delta, points, canonical, max_steps, max_dimension
    )
