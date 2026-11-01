"""Normalized plane-curve genus, ground-field rationality and birational maps."""

from dataclasses import dataclass, replace
from functools import cached_property

import sympy as sp

from .domains import field_element
from .errors import QuotientAlgebraError
from .quotient import QuotientAlgebra, _require_exact_domain
from .subfields import field_normal_form


class CurveError(QuotientAlgebraError):
    """Invalid curve or an unsupported exact curve-field operation."""


class CurveComputationLimitError(CurveError):
    """An explicit exact-computation budget was exceeded; no decision is made."""


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise CurveError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class PlaneCurve:
    """An affine plane equation irreducible over its coefficient field.

    Irreducibility over ``QQ`` or a number field does not imply geometric
    integrality over an algebraic closure. Construction verifies only the
    former; normalization and genus require the latter and raise ``CurveError``
    if it fails. This permits representing a base-field integral equation
    without prematurely running the more expensive geometric analysis.
    """

    polynomial: sp.Poly
    variables: tuple[sp.Symbol, sp.Symbol]
    domain: object

    @classmethod
    def from_polynomial(cls, polynomial, variables, *, domain=sp.QQ):
        """Construct a reduced, base-field irreducible plane curve.

        The constructor does not assert geometric integrality. A polynomial
        irreducible over ``QQ`` may factor over an algebraic extension; genus
        and normalization check this additional hypothesis separately.
        """
        variables = tuple(variables)
        if (
            len(variables) != 2
            or len(set(variables)) != 2
            or any(not isinstance(v, sp.Symbol) for v in variables)
        ):
            raise CurveError("a plane curve requires two distinct Symbols")
        domain = _require_exact_domain(domain)
        if getattr(domain, "is_FractionField", False):
            raise CurveError("curve constants must be rational or algebraic numbers")
        expression = (
            polynomial.as_expr() if isinstance(polynomial, sp.Poly) else sp.sympify(polynomial)
        )
        if expression.has(sp.Float) or expression.free_symbols - set(variables):
            raise CurveError("curve coefficients must be exact constants")
        try:
            poly = sp.Poly(expression, *variables, domain=domain)
        except (sp.PolynomialError, sp.polys.polyerrors.CoercionFailed) as exc:
            raise CurveError("a polynomial over the declared constants is required") from exc
        if poly.total_degree() < 1 or poly.is_zero:
            raise CurveError("a nonconstant curve equation is required")
        factors = poly.factor_list()[1]
        if len(factors) != 1 or factors[0][1] != 1:
            raise CurveError("the curve equation must be reduced and irreducible")
        return cls(poly.monic(), variables, domain)

    @property
    def degree(self):
        return int(self.polynomial.total_degree())

    @cached_property
    def homogeneous_polynomial(self):
        from ._curve_normalization import W

        return self.polynomial.homogenize(W).as_expr()

    @cached_property
    def _default_normalization(self):
        from ._curve_normalization import normalize

        return normalize(self, 256, 256)

    def normalization(self, *, max_steps=256, max_dimension=256):
        _positive(max_steps, "max_steps")
        _positive(max_dimension, "max_dimension")
        if (max_steps, max_dimension) == (256, 256):
            return self._default_normalization
        from ._curve_normalization import normalize

        return normalize(self, max_steps, max_dimension)

    @property
    def genus(self):
        return self.normalization().genus

    def rational_parametrization(
        self,
        parameter=None,
        *,
        point=None,
        extend_constants=False,
        max_steps=256,
        max_dimension=256,
    ):
        """Decide rationality over QQ, or use a supplied number-field point.

        A negative genus-zero decision over QQ includes a conic obstruction and
        an explicit quadratic splitting field. Other number fields require a
        point witness or an explicitly requested constants extension.
        """
        from ._curve_parametrization import rational_parametrization

        return rational_parametrization(
            self, parameter, point, extend_constants, max_steps, max_dimension
        )

    def function_field(self, *, domain=None):
        """A finite quotient over one coordinate's rational function field."""
        x, y = self.variables
        if self.polynomial.degree(y) == 0:
            x, y = y, x
        polynomial = sp.Poly(self.polynomial.as_expr(), y)
        leading = polynomial.LC()
        # Normalize in expressions first: this also avoids expensive repeated
        # coefficient-field normalization of a nonunit leading polynomial.
        equation = sum(
            sp.cancel(c / leading) * y ** monomial[0] for monomial, c in polynomial.terms()
        )
        algebra = QuotientAlgebra.from_polynomials(
            [equation], (y,), domain=(domain or self.domain).frac_field(x)
        )
        return replace(algebra, defining_polynomials=(self.polynomial.as_expr(),))


@dataclass(frozen=True)
class ResolutionPoint:
    """A closed singular point in centered local coordinates on a blowup chart."""

    local_equation: sp.Expr
    projective_coordinates: tuple[sp.Expr, ...]
    adjoint_multiplier: sp.Expr
    domain: object
    residue_degree: int
    multiplicity: int
    depth: int
    chart: str


@dataclass(frozen=True)
class CurveNormalization:
    """Genus of the smooth projective normalization, with exact replay data."""

    curve: PlaneCurve
    genus: int
    arithmetic_genus: int
    delta_sum: int
    singularities: tuple[ResolutionPoint, ...]
    canonical_adjoints: tuple[sp.Expr, ...]
    max_steps: int
    max_dimension: int

    def verify(self):
        from ._curve_normalization import normalize

        return self == normalize(self.curve, self.max_steps, self.max_dimension)


@dataclass(frozen=True)
class CurveParametrization:
    """A birational normalization map from P1, with a generic rational inverse."""

    curve: PlaneCurve
    domain: object
    parameter: sp.Symbol
    projective_coordinates: tuple[sp.Expr, ...]
    inverse: sp.Expr

    @property
    def affine_coordinates(self):
        X, Y, Z = self.projective_coordinates
        return sp.cancel(X / Z), sp.cancel(Y / Z)

    @property
    def forward_guard(self):
        return sp.Ne(self.projective_coordinates[2], 0)

    @property
    def inverse_guard(self):
        return sp.Ne(sp.fraction(sp.cancel(self.inverse))[1], 0)

    @property
    def exceptional_parameter_polynomial(self):
        x, y = self.curve.variables
        denominator = sp.fraction(sp.cancel(self.inverse))[1]
        pulled = sp.fraction(
            sp.cancel(
                denominator.subs(
                    dict(zip((x, y), self.affine_coordinates, strict=True)), simultaneous=True
                )
            )
        )[0]
        return (
            sp.Poly(pulled * self.projective_coordinates[2], self.parameter, domain=self.domain)
            .sqf_part()
            .monic()
        )

    @property
    def point_at_infinity(self):
        polys = [
            sp.Poly(p, self.parameter, domain=self.domain) for p in self.projective_coordinates
        ]
        degree = max(p.degree() for p in polys)
        return tuple(p.nth(degree) for p in polys)

    def homogeneous_coordinates(self, first, second):
        if not isinstance(first, sp.Symbol) or not isinstance(second, sp.Symbol) or first == second:
            raise CurveError("two distinct homogeneous parameter Symbols are required")
        degree = max(sp.degree(p, self.parameter) for p in self.projective_coordinates)
        return tuple(
            sp.expand(second**degree * p.subs(self.parameter, first / second))
            for p in self.projective_coordinates
        )

    def verify(self):
        from ._curve_normalization import W

        t = self.parameter
        if t in self.curve.variables or not isinstance(t, sp.Symbol):
            return False
        polys = [sp.Poly(p, t, domain=self.domain) for p in self.projective_coordinates]
        if len(polys) != 3 or polys[2].is_zero or all(p.degree() <= 0 for p in polys):
            return False
        gcd = polys[0].gcd(polys[1]).gcd(polys[2])
        if gcd.degree() != 0:
            return False
        x, y = self.curve.variables
        projective = dict(zip((x, y, W), self.projective_coordinates, strict=True))
        if sp.Poly(
            sp.expand(self.curve.homogeneous_polynomial.subs(projective, simultaneous=True)),
            t,
            domain=self.domain,
        ):
            return False
        substitutions = dict(zip((x, y), self.affine_coordinates, strict=True))
        if (
            field_element(
                self.inverse.subs(substitutions, simultaneous=True) - t, self.domain.frac_field(t)
            )
            != 0
        ):
            return False
        A = self.curve.function_field(domain=self.domain)
        return all(
            field_normal_form(A, c.subs(t, self.inverse) - v) == 0
            for c, v in zip(self.affine_coordinates, (x, y), strict=True)
        )


@dataclass(frozen=True)
class ConicModel:
    """A smooth conic and exact rational maps in both directions."""

    source: PlaneCurve
    conic: PlaneCurve
    forward: tuple[sp.Expr, sp.Expr]
    inverse: tuple[sp.Expr, sp.Expr]

    def verify(self):
        if (
            self.conic.domain != self.source.domain
            or self.conic.degree != 2
            or self.conic.genus != 0
        ):
            return False
        A = self.source.function_field()
        B = self.conic.function_field()
        x, y = self.source.variables
        u, v = self.conic.variables
        if (
            field_normal_form(
                A,
                self.conic.polynomial.as_expr().subs(
                    dict(zip((u, v), self.forward, strict=True)), simultaneous=True
                ),
            )
            != 0
        ):
            return False
        if (
            field_normal_form(
                B,
                self.source.polynomial.as_expr().subs(
                    dict(zip((x, y), self.inverse, strict=True)), simultaneous=True
                ),
            )
            != 0
        ):
            return False
        return all(
            field_normal_form(
                A, p.subs(dict(zip((u, v), self.forward, strict=True)), simultaneous=True) - var
            )
            == 0
            for p, var in zip(self.inverse, (x, y), strict=True)
        ) and all(
            field_normal_form(
                B, p.subs(dict(zip((x, y), self.inverse, strict=True)), simultaneous=True) - var
            )
            == 0
            for p, var in zip(self.forward, (u, v), strict=True)
        )


@dataclass(frozen=True)
class CurveRationalityResult:
    """A rational map, positive genus, or a certified ground-field point obstruction."""

    normalization: CurveNormalization
    status: str
    parametrization: CurveParametrization | None = None
    conic_model: ConicModel | None = None
    splitting_field: object | None = None
    splitting_point: tuple[sp.Expr, ...] | None = None

    @property
    def rational(self):
        return self.status == "rational"

    def verify(self):
        if not self.normalization.verify():
            return False
        if self.status == "positive_genus":
            return self.normalization.genus > 0 and self.parametrization is None
        if self.normalization.genus != 0:
            return False
        if self.conic_model is not None and not self.conic_model.verify():
            return False
        if self.status in ("rational", "rational_over_extension"):
            return (
                self.parametrization is not None
                and self.parametrization.curve == self.normalization.curve
                and self.parametrization.verify()
                and (
                    self.parametrization.domain == self.normalization.curve.domain
                    if self.status == "rational"
                    else self.parametrization.domain == self.splitting_field
                    and self.splitting_field != self.normalization.curve.domain
                )
            )
        if self.status == "no_ground_field_point" and self.conic_model is not None:
            from ._curve_parametrization import conic_point, validate_point

            return (
                self.parametrization is None
                and self.splitting_field != self.normalization.curve.domain
                and self.normalization.curve.domain == sp.QQ
                and conic_point(self.conic_model.conic) is None
                and validate_point(
                    self.conic_model.conic, self.splitting_point, self.splitting_field
                )
                is not None
            )
        return False


__all__ = [
    "PlaneCurve",
    "ResolutionPoint",
    "CurveNormalization",
    "CurveParametrization",
    "ConicModel",
    "CurveRationalityResult",
    "CurveError",
    "CurveComputationLimitError",
]
