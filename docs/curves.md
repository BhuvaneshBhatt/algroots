# Normalized curve genus and rationality

`algroots.curves` handles irreducible plane equations over `QQ` or exact algebraic-number constants. It computes the genus of the smooth projective normalization, including singularities at infinity and infinitely near points. Genus-zero parametrization includes an exact rationality decision over `QQ`. A supplied point or an explicit constants extension supports number fields. These APIs require SymPy 1.14 or later.

## Genus of the normalization

<!-- algroots: execute -->
```python
import sympy as sp
from algroots.curves import PlaneCurve

x, y = sp.symbols("x y")
curve = PlaneCurve.from_polynomial(y**2-x**5, (x,y))
normalization = curve.normalization()
assert normalization.arithmetic_genus == 6
assert normalization.delta_sum == 6
assert normalization.genus == 0 and normalization.verify()
assert any(p.depth > 0 for p in normalization.singularities)
elliptic = PlaneCurve.from_polynomial(y**2-x**3+x, (x,y))
assert elliptic.genus == 1
```

`PlaneCurve.from_polynomial(f, (x,y), domain=QQ)` rejects constant, inexact, nonpolynomial, repeated or reducible equations **over the declared coefficient field**. Construction does not certify geometric integrality; normalization and genus require geometric irreducibility after extension to an algebraic closure, and raise `CurveError` if it fails. For example, `x**2+y**2` is irreducible over `QQ` but splits over `QQ(i)`; its normalization cannot be treated as that of a single geometrically integral curve. The original `PlaneCurve` object still represents an integral equation over `QQ`.

`CurveNormalization` records arithmetic genus, the delta correction, geometric genus, canonical adjoints and `ResolutionPoint` records. Closed singular points are treated one conjugacy orbit at a time, with residue-degree weights. The algorithm resolves strict transforms by point blowups in two affine charts. The correction is the sum of `degree(P)*m(P)*(m(P)-1)/2` over all singular and infinitely near points. Homogeneous adjoints impose multiplicity `m-1` at every such point; their canonical-degree dimension independently checks geometric integrality and the computed genus. `verify()` replays these computations. This is resolution/genus data, not a general integral-closure ring API.

`normalization(max_steps=256, max_dimension=256)` has explicit limits on recorded singular points and the finite singular-locus quotient. Exceeding a limit raises `CurveComputationLimitError`; it is not a genus or rationality decision.

## Birational parametrization and exceptional charts

<!-- algroots: execute -->
```python
import sympy as sp
from algroots.curves import PlaneCurve

x, y, t = sp.symbols("x y t")
result = PlaneCurve.from_polynomial(y**2-x**3, (x,y)).rational_parametrization(t)
assert result.rational and result.verify()
p = result.parametrization
assert p.affine_coordinates == (t**2,t**3)
assert p.inverse == y/x
assert p.exceptional_parameter_polynomial == sp.Poly(t,t,domain=sp.QQ)
assert p.point_at_infinity == (0,1,0)
```

`rational_parametrization(parameter=None, point=None, extend_constants=False, max_steps=256, max_dimension=256)` returns `CurveRationalityResult` with status `rational`, `positive_genus`, `no_ground_field_point`, or `rational_over_extension`. Its `rational` property means rational over the original constants, so it is false for a parametrization over a larger field. For number-field extensions, that false value does not prove nonrationality over the original field; inspect the status and the returned map domain.

`CurveParametrization` records coprime projective polynomial coordinates in one affine parameter, the constants domain, and a generic rational inverse. `affine_coordinates`, `forward_guard` and `inverse_guard` expose affine charts. `exceptional_parameter_polynomial` records finite affine/inverse chart exclusions. `point_at_infinity` gives the projective image of parameter infinity; `homogeneous_coordinates(s,t)` supplies the full projective-line map. A chart exclusion does not delete the corresponding normalization point.

`verify()` checks the implicit equation, absence of a common projective factor, the parameter inverse identity and both curve-coordinate identities. Thus a multiple cover or an arbitrary rational map is not accepted as birational. The circle, for example, has an affine point at parameter infinity which a single affine parameter chart would omit.

## Genus zero and ground-field point obstructions

<!-- algroots: execute -->
```python
import sympy as sp
from algroots.curves import PlaneCurve

x, y, t = sp.symbols("x y t")
curve = PlaneCurve.from_polynomial(x**2+y**2-3, (x,y))
result = curve.rational_parametrization(t)
assert result.normalization.genus == 0
assert result.status == "no_ground_field_point" and result.verify()
assert result.parametrization is None
assert result.splitting_field == sp.QQ.algebraic_field(sp.sqrt(3))
extended = curve.rational_parametrization(t, extend_constants=True)
assert extended.status == "rational_over_extension"
assert not extended.rational and extended.parametrization.verify()
```

The `QQ` decision uses SymPy's exact ternary-quadratic descent, not a bounded point search. A negative result carries a verified `ConicModel` and an explicit quadratic splitting field and point. `verify()` replays the exact conic decision and validates the splitting point. A real point is insufficient for rationality over `QQ`; a nonreal splitting field is also possible.

For singular curves of higher degree, adjoints and point jets first give a rational parameter over a finite constants extension. The three regular vector fields `(1,t,t**2)*d/dt` descend by exact linear algebra; their double-zero locus gives an anticanonical conic over the original constants. Both directions of the curve/conic maps are checked. This handles plane models such as `x**2+(y-x**2)**2-3`, whose arithmetic genus and degree differ from its conic normalization and which still has no `QQ` parametrization.

Automatic point-obstruction decisions over general number fields are not implemented. Supply `point=(x0,y0)` or three homogeneous coordinates in the chosen constants, or request `extend_constants=True`. Without a witness or extension request, an unsupported number-field point decision raises `CurveError`; it is never reported as a negative decision. Supplied points on higher-degree plane models must be smooth. Extensions preserve the embedding of the original constants.

## Coefficient-generated fields

<!-- algroots: execute -->
```python
import sympy as sp
from algroots.quotient import QuotientAlgebra
from algroots.function_fields import coefficient_field

x, z, u, v, t = sp.symbols("x z u v t")
ambient = QuotientAlgebra.from_polynomials([x**2-z], (x,), domain=sp.QQ.frac_field(z))
field = coefficient_field([x**2,x**3], ambient=ambient, generator_symbols=(u,v))
assert field.genus == 0 and field.curve_model().verify()
p = field.rational_parametrization(t)
assert p.coefficient_images == (t**2,t**3)
assert p.generator == x and p.verify()
assert p.to_source(x+1) == t+1
```

`AlgebraicCoefficientField.curve_model()` returns `CoefficientCurveModel`, using a primitive element over its nonconstant coefficient. `coordinate_expressions` embed plane coordinates in the original coefficient symbols; `generator_images` map every original coefficient to the plane field. Both embeddings replay. `genus` and `rationality(...)` operate on this model.

`rational_parametrization(...)` returns `CoefficientFieldParametrization` only when rational over the existing constants. It exposes the rational generator in the original ambient field, its expression in the original coefficients, coefficient images, guards and constructive maps. Negative cases raise and refer to `rationality()` for the structured obstruction. Constant-generated number fields have no curve genus. A constants extension is inspected through `curve_model().curve`; it is not silently adjoined to the original ambient field.

## Scope and references

Algorithms are exact characteristic-zero computations. Blowup coefficients, number-field composita, saturated image elimination and adjoint linear algebra can grow substantially. The limits above are not wall-clock limits. The [decomposition frontend](functional-decomposition.md) discovers algebraic extensions and factors and reconstructs their components using this layer. For compositional chains, use `rational_decomposition_chain` or `algebraic_decomposition_chain` in the functional-decomposition module; curve normalization itself does not construct chains. Real image coverage and branch selection belong in downstream semialgebraic code.

The mathematics follows [resolution by infinitely near points](https://gradmath.org/wp-content/uploads/2022/09/GJM2022-Chalmoviansky.pdf), [adjoint linear systems for rational curves](https://macaulay2.com/doc/Macaulay2/share/doc/Macaulay2/Parametrization/html/index.html), and [anticanonical conic models](https://docs.magma-maths.org/ArithmeticGeometry/RationalCurvesAndConics/crvgen0-sec-introduction.html). The conic solver is documented in [SymPy's Diophantine API](https://docs.sympy.org/latest/modules/solvers/diophantine.html).
