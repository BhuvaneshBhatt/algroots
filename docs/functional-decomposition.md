# Certified functional decomposition

`algroots.functional_decomposition` searches rational two-factor compositions, discovers algebraic coefficient extensions and factors, and reconstructs implicit algebraic decompositions. Explicit supplied factors remain supported. These are expert APIs; the numerical all-roots solvers and package-root exports retain their existing contracts. Coefficients must be exact, with constants `QQ` or a SymPy algebraic number field in characteristic zero.

## Rational functions: automatic search

<!-- algroots: execute -->
```python
import sympy as sp
from algroots.functional_decomposition import rational_decompose

x, t = sp.symbols("x t")
result = rational_decompose(x**6, x, parameter=t)
assert result.complete and result.status == "decomposable" and result.verify()
assert {d.degrees for d in result.decompositions} == {(2, 3), (3, 2)}
for d in result.decompositions:
    assert d.verify()
    assert sp.cancel(d.outer.subs(t, d.inner) - x**6) == 0
```

```text
rational_decompose(expression, variable, *, parameter=None, constants=QQ,
                   max_candidates=256, max_degree=32)
```

For a reduced `f=p/q`, search factors the near-separated polynomial `p(X)*q(x)-q(X)*p(x)`. It enumerates products containing `X-x` with proper divisor degrees, instead of testing only individual irreducible factors. A nonconstant coefficient of the monic candidate supplies an inner function; its near-separated identity and the recovered outer composition are checked. Every eligible right component over the declared constants occurs this way. The monic candidate is unique for its intermediate field, so the output has one representative per fractional linear (Möbius) change of intermediate coordinate. Inner functions need not be monic polynomials or have positive sign.

`RationalDecompositionResult` exposes `decompositions`, `examined`, `candidate_count`, `complete`, `status`, and `verify()`. A certificate exposes `outer`, `inner`, `variable`, `parameter`, `constants`, `function`, `degrees` (outer, inner), `original_expression`, `input_guard`, and `verify()`. `input_guard` retains visible original poles before cancellation. Expressions already simplified by the caller cannot reveal previously discarded holes.

The statuses are `decomposable` when a verified pair was found, `indecomposable` only after exhaustive failure, `incomplete` for an unfinished negative search, and `degree_below_two` for constants and degree-one functions. `complete` means all two-factor answers over the **declared constant field**, up to Möbius equivalence. It does not describe larger constant fields or recursive chains. Verification checks the irreducible factorization and the searched prefix. For a complete univariate search, a separate checker enumerates eligible factor subsets and tests near-separated polynomial identities without calling the search’s quotient-algebra reconstruction. A missing answer prevents a completeness or indecomposability certificate.

`max_candidates` bounds eligible factor products. Products are counted by dynamic programming and generated lazily, so a small budget does not first allocate the full powerset. The limit does not bound factorization or wall-clock time. `max_degree` rejects a larger rational degree before factorization with `DecompositionLimitError`. Unsupported exact operations propagate their errors; they never become negative proofs.

<!-- algroots: execute -->
```python
limited = rational_decompose(x**6, x, max_candidates=0)
assert limited.status == "incomplete" and not limited.complete and limited.verify()
negative = rational_decompose(x**4+x, x)
assert negative.status == "indecomposable" and negative.complete and negative.verify()
```

## Multivariate rational functions: GRS02 Algorithm 2.1

`multivariate_rational_decompose` searches for `f(X)=g(h(X))`, with a univariate rational outer function and a multivariate rational inner function. This is the uni-multivariate decomposition of [Gutiérrez, Rubio and Sevilla (2002), §2.1](https://grupos.unican.es/amac/articles/JSC02.pdf).

<!-- algroots: execute -->
```python
import sympy as sp
from algroots.functional_decomposition import multivariate_rational_decompose

x, y, t = sp.symbols("x y t")
f = (x**2+y**2)**2
result = multivariate_rational_decompose(f, (x,y), parameter=t)
assert result.complete and result.status == "decomposable" and result.verify()
assert len(result.decompositions) == 1
pair = result.decompositions[0]
assert pair.degrees == (2,2) and pair.verify()
assert sp.cancel(pair.outer.subs(t,pair.inner)-f) == 0
limited = multivariate_rational_decompose((x+y)**6,(x,y),max_candidates=0)
assert not limited.complete and limited.status == "incomplete" and limited.verify()
```

```text
multivariate_rational_decompose(expression, variables, *, parameter=None,
                               constants=QQ, max_candidates=256, max_degree=32)
```

`rational_decompose(f, (x,y), ...)` and list arguments dispatch to this route. Passing a single `Symbol` preserves the existing univariate API and result type. The explicit multivariate route also accepts a one-variable tuple and a generator of input variables. Constants remain rational or exact algebraic-number fields; there is no implicit extension of constants.

The returned `MultivariateRationalDecompositionResult` exposes `variables`, `comparison_variables`, `factors`, `examined`, `candidate_count`, `decompositions`, `complete`, `status`, and `verify()`. A `MultivariateRationalDecomposition` certificate has `variables`, `parameter`, `constants`, `outer`, `inner`, `original_expression`, `function`, `degrees`, `input_guard`, and `verify()`. Degrees are `(degree(g), total_degree(h))`, where rational total degree is the maximum of the reduced numerator's and denominator's total degrees.

Following the paper's nontriviality convention, both degrees must exceed one. Linear multivariate inner functions are excluded: `(x+y)**2` has no nontrivial decomposition under this convention. Degree multiplicativity rules out prime total degree, so these cases need no factorization. Constants and degree-one inputs have status `degree_below_two`; prime-degree inputs have `indecomposable`.

The implementation factors `p(Y)*q(X)-q(Y)*p(X)` in two independent variable blocks. A candidate product must have equal block degrees that properly divide `degree(f)`, and must vanish when `Y=X`. Its factors need not include a linear diagonal factor. A suffix dynamic program counts eligible products, including the diagonal condition, and a lazy traversal visits their configured prefix. For each candidate, a nonconstant monic coefficient proposes `h`. Its exact near-separated identity is checked, and generic graph reduction recovers the unique outer function. Products that are not symmetric near-separated polynomials are rejected.

Completeness means all nontrivial two-factor classes over the declared constants, modulo Möbius changes of intermediate coordinate. The canonical monic near-separated polynomial identifies each class once. Verification checks the irreducible factorization, candidate count, searched prefix, every recovered identity, and any exhaustive negative claim. A stopped negative prefix is `incomplete`, never `indecomposable`; a stopped positive prefix is `decomposable` with `complete=False`.

`max_degree` bounds rational total degree before factorization. `max_candidates` counts eligible factor products, including those rejected by near-separated recognition. It does not bound factorization, verification, or wall-clock time. Multivariate factorization can dominate the cost, and complete enumeration can be exponential. Unsupported exact operations raise instead of becoming negative proofs. This two-factor interface does not construct chains; use `rational_decomposition_chain` for one verified recursive chain. Coordinatewise multi-univariate decompositions and general vector-valued inner maps are outside the supported fragment.

## Implicit algebraic functions: automatic discovery

<!-- algroots: execute -->
```python
from algroots.functional_decomposition import algebraic_decompose, discover_algebraic_factors

x, z, t = sp.symbols("x z t")
F = x**4-z*x**2*(x+1)+z**3*(x+1)**2
search = algebraic_decompose(F, (x,z), parameter=t)
assert search.status == "decomposable" and search.verify()
assert search.decompositions[0].verify()
assert not search.complete  # The default stops at the first verified answer.
extensions = discover_algebraic_factors(F, (x,z), degrees=(2,))
assert extensions.complete and extensions.verify()
assert [o.extension_degree for o in extensions.orbits] == [2,4]
```

```text
algebraic_decompose(relation, variables, *, candidate=None, ambient=None,
                    parameter=None, constants=QQ, point=None,
                    max_steps=256, max_dimension=256, factor_degrees=None,
                    max_degree=6, max_factor_dimension=32,
                    max_separating_attempts=64, max_candidates=64,
                    exhaustive=False)
discover_algebraic_factors(relation, variables, *, degrees=None, constants=QQ,
                          max_degree=6, max_factor_dimension=32,
                          max_separating_attempts=64)
```

For each factor degree `m`, construct a generic monic factor with `m` unknown coefficients and set the remainder of monic `F` modulo this factor to zero. For separable output degree `n`, this finite algebra has `binomial(n,m)` geometric points: all degree-`m` root subsets. A separating linear form gives a cyclic presentation. Factoring its polynomial over `constants(input)` splits the algebra into fields, one for each Galois orbit of factors. Exact coordinate maps recover the factor and its cofactor in each extension. The extension is generated by the factor's coefficients over `constants(input)`.

This discovers extensions that may not lie in a field generated by one root. For example, `x**4+x+z` has a degree-six coefficient extension carrying a quadratic factor. No complete splitting field or numerical root selection is needed. The algebra is finite over `constants(input)`; reconstruction still forms the coefficient-generated field **over constants alone**.

`AlgebraicFactorDiscovery` exposes `requested_degrees`, `levels`, `orbits`, `issues`, `complete`, and `verify()`. Its default degrees are `1,...,n`. Completeness covers the **requested degrees**, with one representative per Galois orbit, instead of separate numerical choices for every conjugate. Each `AlgebraicFactorDegree` stores its universal quotient, defining polynomial, coordinate maps and irreducible component polynomials. Its verifier rebuilds the remainder equations, checks both cyclic maps, and checks that the component fields cover every point. A `DiscoveredAlgebraicFactor` exposes `factor`, `cofactor`, `ambient`, `coefficients`, `degree`, `extension_degree`, and `verify()`. Its factor/cofactor product is the input made monic in output; multiply by the original output leading coefficient to recover a nonmonic input equation.

With no supplied candidate, `algebraic_decompose` returns `AlgebraicDecompositionSearch`. It visits factor orbits in degrees `2,...,n`; degree one gives a trivial minimal outer component. Each orbit is passed to the same supplied-candidate reconstruction below. Small extension degrees are tried first within each factor degree. `exhaustive=False` stops at the first verified answer. `exhaustive=True` continues within the configured limits.

The result exposes `discovery`, `attempts`, `parameter`, `relation`, `examined`, `candidate_count`, `decompositions`, `status`, `complete`, and `verify()`. `AlgebraicCandidateAttempt` retains a reconstruction result or an unresolved diagnostic. `decompositions` contains independently verified resultant certificates, suitable for the semialg adapter. `status` is `decomposable` if an answer was found, `incomplete` when unhandled candidates remain, or `indecomposable` after every required orbit was certified rejected. `candidate_count` counts discovered orbits; skipped factor degrees are recorded separately in `discovery.issues` and prevent complete coverage.

This completeness concerns **nontrivial minimal coefficient-field decompositions over the declared constants**, following the supplied-factor contract. It does not enumerate larger constant fields, nonminimal insertions, or recursive chains. An output-degree-one relation has no such outer factor and returns `degree_below_two`; use `rational_decompose` for rational-function composition. A `factor_degrees` restriction does not establish global indecomposability unless it includes every degree `2,...,n`.

Limits have separate meanings. `max_degree` rejects a larger output degree before factor-algebra construction. `max_factor_dimension` skips degrees whose known binomial dimension exceeds the limit. `max_separating_attempts` limits the search for a separating form. `max_candidates` limits coefficient-field/curve reconstruction, after factor discovery. The curve limits are forwarded to each attempt. None is a wall-clock limit. Missing degrees, an early successful stop, an exhausted candidate limit, and unresolved curve/constant-field cases all prevent `complete=True`. Diagnostic messages are not negative proofs.

Automatic reconstruction retains the curve layer's support limits. An unsupported candidate is recorded and prevents indecomposability; internal arithmetic/certificate failures propagate. Point witnesses apply to supplied candidates, whose coefficient plane models can first be inspected. The automatic path never extends the declared constants. All extension identities are generic function-field identities. Specializing input can change degrees, multiplicities or chart validity; rebuild the fiber instead of treating a generic factorization as a uniform specialization proof.

## Implicit algebraic functions: supplied candidates

<!-- algroots: execute -->
```python
from algroots.quotient import QuotientAlgebra
from algroots.functional_decomposition import algebraic_decompose

x, z, a, t = sp.symbols("x z a t")
F = x**4-z*x**2*(x+1)+z**3*(x+1)**2
A = QuotientAlgebra.from_polynomials(
    [a**2-z*a+z**3], (a,), domain=sp.QQ.frac_field(z),
)
result = algebraic_decompose(
    F, (x,z), candidate=x**2-a*(x+1), ambient=A, parameter=t,
)
assert result.status == "decomposable" and result.verify()
assert not result.complete
assert not result.coefficient_field.contains(z)
d = result.decomposition
assert d.verify() and d.exponent == 1
assert sp.resultant(d.outer.as_expr(), d.inner.as_expr(), t) == d.unit*d.relation.as_expr()**d.exponent
```

```text
algebraic_decompose(relation, variables, *, candidate, ambient, parameter=None,
                    constants=QQ, point=None, max_steps=256, max_dimension=256)
# Supplied-candidate mode; automatic-search options retain their defaults.
```

`variables=(output,input)`. The input must be an irreducible polynomial depending on both variables. `ambient` is a certified finite `QuotientAlgebra` field over `constants(input)`; output and intermediate symbols must be fresh there. `candidate` is a polynomial in output with coefficients in this ambient field. It is reduced and made monic there, and must divide the input polynomial. Supplying both arguments selects this mode; supplying neither selects automatic discovery. Supplying just one is an input error.

The implementation constructs the field generated by the candidate's coefficients **over constants**, without adjoining input. The [coefficient-field](function-fields.md) and [curve](curves.md) layers provide a rational generator and exact maps. Replacing coefficients by rational functions of this generator gives `outer`; its minimal polynomial over `constants(input)` gives `inner`. Denominators are cleared to primitive polynomial relations and scalar-normalized. The reported `relation` is also scalar-normalized.

`AlgebraicDecomposition` records `relation`, `outer`, `inner`, `exponent`, `unit`, and the output/input/intermediate variables. `verify()` rechecks irreducibility, positive bidegrees, both degree products and the exact resultant identity `resultant(outer,inner,intermediate) = unit*relation**exponent`. A component of bidegree `(1,1)` is an invertible Möbius coordinate and is rejected as trivial. The exponent can exceed one when conjugate factors overlap; assuming exponent one would reject valid decompositions.

`AlgebraicDecompositionResult` retains `candidate`, `ambient`, `coefficient_field`, `rationality`, `parametrization`, `decomposition`, and `reason`. Its `verify()` also replays the candidate embedding, coefficient-field presentation, rational maps, and their connection to the returned components. `decompositions` is a zero- or one-element tuple. `status` is `decomposable` or `candidate_rejected`. Rejections include `positive_genus`, `no_ground_field_point`, `reducible_candidate`, and `trivial_component`. **`complete` is always false**: a rejected supplied factor is not a proof that the input is indecomposable.

Algebraic reconstruction retains the curve layer's boundaries: geometrically integral coefficient curves, automatic point decisions over `QQ`, and supplied points over general number fields. `point` refers to the coefficient plane model, which can be inspected through `result.coefficient_field.curve_model()` after a supported rejection or constructed directly with `coefficient_field`. Unsupported number-field decisions raise instead of claiming an obstruction. Curve resource limits are forwarded; they are not wall-clock limits.

## Recursive decomposition chains

The chain APIs choose the first verified split at each node, then recursively refine its outer and inner components. They return one deterministic chain; they do not enumerate every inequivalent chain.

<!-- algroots: execute -->
```python
import sympy as sp
from algroots.functional_decomposition import (
    rational_decomposition_chain, algebraic_decomposition_chain,
)

x, y, z = sp.symbols("x y z")
chain = rational_decomposition_chain(x**8, x)
assert chain.complete and chain.status == "complete" and chain.verify()
assert len(chain.components) == 3 and chain.compose() == x**8
multi = rational_decomposition_chain((x**2+y**2)**4, (x,y))
assert multi.complete and multi.verify() and len(multi.components) == 3
implicit = algebraic_decomposition_chain(x**8-z, (x,z))
assert implicit.complete and implicit.verify() and len(implicit.components) == 3
assert implicit.resultant() == implicit.unit*implicit.relation**implicit.exponent
limited = rational_decomposition_chain(x**8, x, max_depth=1)
assert not limited.complete and limited.verify() and limited.compose() == x**8
```

```text
rational_decomposition_chain(expression, variable, *, constants=QQ, initial=None,
    max_depth=8, max_nodes=63, max_candidates=256, max_degree=32)
algebraic_decomposition_chain(relation, variables, *, constants=QQ, initial=None,
    max_depth=8, max_nodes=63, max_candidates=64, max_degree=6,
    max_rational_degree=32, max_factor_dimension=32, max_separating_attempts=64,
    max_steps=256, max_dimension=256)
```

`initial` optionally accepts an already verified two-component certificate for this exact input and constant field. For example, pass `initial=algebraic_decompose(...).decompositions[0]` to retain a particular supplied-factor or automatically discovered decomposition and refine both sides. Limits on recursive searches still apply to its children.

`RationalDecompositionChain.components` is an outer-to-inner tuple of `RationalChainComponent(expression, variables)`. Only the last component may have several input variables. `compose()` computes their function-field identity; `input_guard` retains visible original poles. `AlgebraicDecompositionChain.components` is an ordered tuple of bivariate polynomials with adjacent coordinates matched. `resultant()` eliminates all internal coordinates; `exponent` and `unit` retain its exact resultant power. Linear-coordinate implicit relations also use rational decomposition in the appropriate graph orientation, covering polynomial maps and root towers.

Both results retain a `root` split tree and its `leaves`. Every `DecompositionChainNode` records its input, coordinate variables, constants, two-factor search results, selected split and child nodes, or a `stop_reason` and diagnostic `detail`. When used, a supplied initial certificate is retained in `root.supplied`. `verify()` replays this evidence and the final composition. An incomplete parent search can still yield a complete chain: after one valid split, only its selected descendants need to be resolved.

`complete` means every leaf has a complete negative two-factor search in the declared scope, or degree below two. Open leaves have reasons `max_depth`, `max_nodes`, `degree_limit`, `search_incomplete`, `unsupported`, or `repeated_component`. A valid algebraic resultant split can reproduce an ancestor after coordinate renaming; such a leaf remains unresolved. Algebraic completeness inherits the minimal coefficient-field scope of automatic discovery, augmented by rational graph decomposition. It is not arbitrary algebraic-function indecomposability or an assertion about larger constant fields.

`max_depth` counts recursive edges; zero retains an unsearched root. `max_nodes` bounds the stored binary tree, including open leaves. Candidate, factor-algebra, curve and degree limits apply separately at each visited node. `max_rational_degree` bounds the additional rational graph searches. These limits do not bound factorization time or certificate replay. No random specialization or heuristic negative decision is used.

## Real branches and remaining scope

These certificates describe rational-function identities or closed algebraic correspondences. They do not assert equality of arbitrary real branches or coverage of every affine chart. In `semialg.functional_decomposition`, `verify_real_decomposition` accepts an individual certificate with branch selectors and a declared input domain. It preserves affine rational poles and checks real graph equality, totality, and uniqueness separately.

`verify_real_decomposition_chain` compares an entire selected chain, binding all internal coordinates, retaining affine poles, and checking totality and uniqueness separately. A chain may be incomplete as a search while still giving a valid exact composition. Parametric rational/implicit synthesis and exhaustive enumeration of inequivalent chains remain separate work. No dependency from algroots to semialg is introduced.

The search follows the near-separated-polynomial characterization in [Ayad and Fleischmann, *On the decomposition of rational functions*](https://www.kent.ac.uk/ims/personal/pf10/calais/decomp.pdf). The multivariate search implements [Gutiérrez, Rubio and Sevilla, *On Multivariate Rational Function Decomposition*](https://grupos.unican.es/amac/articles/JSC02.pdf), Algorithm 2.1, with exact coefficient recognition and lazy bounded enumeration. The coefficient-field reconstruction and resultant powers follow [Kozen, Landau and Zippel, *Decomposition of Algebraic Functions*](https://www.cs.cornell.edu/~kozen/Papers/alg.pdf). The factor-coefficient algebra is the finite factorization algebra discussed by [Thorup, *On the invariants of the splitting algebra*](https://web.math.ku.dk/~thorup/art/galois.pdf).
