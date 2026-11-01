# Generated subfields and function fields

These exact expert interfaces support rational and algebraic-number constants. Their two base-field conventions are explicit.

## Finite subfields over the existing base

`A.subfield(generators, over=None, generator_symbols=None)` returns a `FiniteSubfield` over `A.domain`. The ambient quotient must be a field, as certified by `A.is_field`. An explicit `over` must equal `A.domain`.

<!-- algroots: execute -->
```python
import sympy as sp
from algroots.quotient import QuotientAlgebra

x, z, u, T = sp.symbols("x z u T")
A = QuotientAlgebra.from_polynomials([x**4-z], (x,), domain=sp.QQ.frac_field(z))
E = A.subfield([x**2], generator_symbols=(u,))
assert E.dimension == 2 and E.verify()
member = E.express(1/(x**2+1))
assert member.belongs and member.verify()
assert E.to_ambient(member.expression) == A.inverse(x**2+1)
assert not E.contains(x)
assert E.contains(z)  # The original base field is included.
P = E.primitive_element(T)
assert P.verify()
assert P.to_source(x**2+1) == T+1
assert P.to_ambient(T+1) == x**2+1
```

`basis` and `basis_expressions` record ambient basis representatives and their expressions in the formal generators. `embedding_matrix` maps basis coordinates into ambient coordinates. `express(element)` returns `SubfieldMembership` with exact coordinates or a negative span decision, `expression`, `belongs`, and `verify()`. Rational expressions in ambient quotient variables are accepted by these field interfaces, including inverses of nonzero elements.

`primitive_element(parameter=None)` returns `PrimitiveSubfield` with a simple quotient presentation `field`, its `ambient_generator`, images of the original generators, and constructive `to_source` and `to_ambient` maps. Its deterministic characteristic-zero search uses a finite bound derived from the subfield degree. Verification replays independence, generator expressions, closure, and the primitive relation after direct substitution in the ambient field. Empty or constant generating sets produce the base field.

## Coefficients generated over constants

`algroots.function_fields.coefficient_field(coefficients, *, ambient, constants=QQ, generator_symbols=None, parameter=None)` constructs `constants(coefficients)`. It does **not** automatically adjoin ambient variables. `ambient` is a rational-function domain or a certified finite quotient field over a one-variable rational-function domain. Multivariate rational inputs must generate a field of transcendence degree at most one. `constants` must equal the ambient domain's rational/algebraic-number constant field. Here it names the chosen constants, not necessarily the ambient field's full algebraic closure of constants.

<!-- algroots: execute -->
```python
import sympy as sp
from algroots.function_fields import coefficient_field
from algroots.quotient import QuotientAlgebra

x, z, u = sp.symbols("x z u")
A = QuotientAlgebra.from_polynomials(
    [x**2-z*x+z**3], (x,), domain=sp.QQ.frac_field(z),
)
E = coefficient_field([x], ambient=A, generator_symbols=(u,))
assert E.presentation == sp.QQ.frac_field(u)
assert E.contains(x) and not E.contains(z)
assert E.express(1/(x+1)).verify() and E.verify()
assert A.subfield([x]).contains(z)  # Fixed-base subfields have different semantics.
```

An algebraic ambient field returns `AlgebraicCoefficientField`: saturated graph relations among `generator_symbols`, a finite `presentation` over a nonconstant coefficient, and exact membership. `base_index` identifies that coefficient; `degree` is the degree over this rebased field. If every coefficient is algebraic over the constants, the presentation is finite over the constants themselves. `express` returns `CoefficientMembership`, with an expression in the original coefficient coordinates or a negative decision. The membership test adjoins a query coordinate and compares exact finite degrees; it also detects a transcendental query over a constant-generated number field. `to_ambient` replays coefficient substitution, and `verify()` recomputes the saturated image and the finite presentation. Its `curve_model()`, `genus`, `rationality()` and `rational_parametrization()` methods connect to the [normalized curve layer](curves.md).

## Rational subfields and constructive Lüroth maps

For a one-variable rational ambient domain, `coefficient_field` returns `RationalFunctionSubfield`. It computes a generator of the rational subfield, its expression in the original coefficients, and the minimal polynomial of the ambient parameter over the subfield. The algorithm takes the monic gcd of `p_i(X)-f_i(z)*q_i(X)` over the ambient field and extracts a nonconstant coefficient. The constructive Lüroth theorem guarantees that this coefficient generates the subfield; degree and both embeddings are verified exactly. See the [constructive Lüroth theorem](https://www.math.fsu.edu/~hoeij/2017/FD_ISSAC_corrected_final.pdf).

<!-- algroots: execute -->
```python
import sympy as sp
from algroots.function_fields import coefficient_field

z, u, v, T = sp.symbols("z u v T")
E = coefficient_field([z**2,z**3], ambient=sp.QQ.frac_field(z),
                      generator_symbols=(u,v), parameter=T)
assert E.generator == z and E.generator_expression == v/u
assert E.extension_degree == 1 and E.verify()
P = E.rational_parametrization()
assert P.coefficient_images == (T**2,T**3)
assert P.inverse_guard == sp.Ne(u,0)
assert P.to_source(z+1) == T+1 and P.to_ambient(T+1) == z+1
F = coefficient_field([z**2], ambient=sp.QQ.frac_field(z), parameter=T)
assert F.extension_degree == 2 and not F.contains(z)
```

## Multivariate constructive Lüroth generators

`luroth_generator(functions, *, variables, constants=QQ, parameter=None, generator_symbols=None)` implements Algorithm 3.1 of [Gutiérrez, Rubio and Sevilla (2002), *On Multivariate Rational Function Decomposition*](https://grupos.unican.es/amac/articles/JSC02.pdf). It accepts rational functions in any nonempty tuple of independent variables, with rational or exact algebraic-number constants. The variables are inputs of the rational functions, rather than additional constants.

The returned `LurothGeneratorResult` has three possible outcomes:

| `status` | Meaning | `transcendence_degree` | `transcendence_degree_lower_bound` |
|---|---|---|---|
| `constant` | All inputs belong to the declared constants | `0` | `0` |
| `rational` | The inputs generate `K(generator)` | `1` | `1` |
| `higher_transcendence_degree` | No single Lüroth generator exists | `None` | `2` |

The last outcome proves a lower bound; it does not compute an exact dimension. Membership and coordinate maps are available for constant/rational outcomes; higher-degree results carry the generator decision only. All outcomes have `verify()`. Generator discovery computes the monic gcd of `p_i(Y)-f_i(X)*q_i(Y)` over `K(X)`, normalized in graded lexicographic order. A nonconstant coefficient of least rational total degree supplies the generator. Constant-only inputs have `gcd_polynomial=None`; a unit gcd certifies the last outcome. Independent comparison variables are recorded in `comparison_symbols`.

<!-- algroots: execute -->
```python
import sympy as sp
from algroots.function_fields import luroth_generator

x, y, t, u, v = sp.symbols("x y t u v")
h = x/y
E = luroth_generator([h**2, h**3], variables=(x, y), parameter=t,
                     generator_symbols=(u, v))
assert E.status == "rational" and E.verify()
assert E.generator == h and E.generator_expression == v/u
assert E.coefficient_images == (t**2, t**3)
assert E.contains(h) and not E.contains(x)
assert E.express(1/(h+1)).expression == 1/(t+1)
assert E.express(x).verify()  # A replayable negative membership decision.
P = E.rational_parametrization()
assert P.to_source(h+1) == t+1
assert sp.cancel(P.to_ambient(t+1) - (h+1)) == 0
assert P.inverse_guard == sp.Ne(u, 0)

proper = luroth_generator([h**2, h**4], variables=(x, y), parameter=t)
assert proper.contains(h**2) and not proper.contains(h)
negative = luroth_generator([x, y], variables=(x, y))
assert negative.status == "higher_transcendence_degree" and negative.verify()
assert negative.transcendence_degree is None
assert negative.transcendence_degree_lower_bound == 2
```

For rational results, `coefficient_images` express the original functions in `parameter`, while `generator_expression` expresses the generator in the formal original `generator_symbols`. `express(value)` and `contains(value)` decide membership in `K(generator)`, including algebraic nonmembers such as `h` in `K(h**2)`. `to_ambient` substitutes the generator for the formal parameter. `rational_parametrization()` provides the same bidirectional maps and generic denominator guards as the one-variable API. These are function-field identities; original pointwise poles erased by input simplification cannot be recovered.

Membership reduces numerators and denominators through the prime generic graph `p(X)-parameter*q(X)` over `K(parameter)`, and tests proportionality of their normal forms. Inverse coordinates in the original inputs are reconstructed by finite-subfield closure and linear algebra in the resulting one-variable image field, over one nonconstant input. This avoids image elimination among the original inputs. Verification recomputes the gcd decision and replays both maps. Discovery uses the paper's gcd method; coordinate reconstruction and verification add work beyond that gcd. There is no degree or wall-clock budget, and exact polynomial computations can be expensive.

`coefficient_field(..., ambient=K.frac_field(x,y))` dispatches to this interface for constant or rational fields. A field of higher transcendence degree raises `QuotientAlgebraError`; call `luroth_generator` to inspect its certified decision. The existing one-variable return type and its `extension_degree` are unchanged. `LurothGeneratorResult` has no `extension_degree` or `minimal_ambient_polynomial`: a multivariate ambient extension is generally not finite. Its rational fields have genus zero by the extended Lüroth theorem.

This interface does not perform multivariate functional-decomposition search (Algorithm 2.1), support positive characteristic, or extend finite algebraic ambient fields to several independent base variables.

`express` uses `parameter` as the coordinate of the recovered generator. `rational_parametrization()` returns `RationalSubfieldParametrization` with coefficient images, forward/inverse maps and denominator guards for their generic charts. These are function-field identities: guards may exclude singular points or removable chart poles. They are not complete real-domain or branch certificates. A constant-generated subfield has transcendence degree zero, no rational generator, and `extension_degree=None`.

## Boundaries and computational cost

Fixed-base subfields use exact dense span calculations; coefficient image and membership computations use saturated Gröbner elimination. They can be costly as degrees and the number of coefficients grow. This module supports one transcendental ambient parameter with explicit finite algebraic extensions. The [curve layer](curves.md) provides normalized genus and genus-zero parametrization, with exact point decisions over `QQ`. General number-field point decisions require witnesses or extensions. The [functional-decomposition frontend](functional-decomposition.md) discovers algebraic coefficient extensions and factors and reconstructs their components; supplied-candidate mode remains available. Denominator guards describe generic maps; exceptional fibers must be rebuilt and real branches checked separately.
