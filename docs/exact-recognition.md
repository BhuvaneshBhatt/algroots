# Exact Recognition

The public polynomial, algebraic, and monodromy solvers attempt exact recognition by default with a conservative `recognition_max_degree=8`. Numerical roots always remain available in `result.roots`; certified reconstructions are attached as `result.recognized_roots`. A failed recognition attempt does not invalidate a verified numerical solve and is reported in `result.recognition_error`. Pass `recognize=False` to disable automatic recognition.

Use `recognize_system_roots(...)` explicitly when you need custom `max_degree`, height, relation-score, or degree-cost settings.
`algroots` can reconstruct exact algebraic coordinates from roots that have already been found numerically. Recognition remains a logically separate post-processing stage even though it runs by default: it can strengthen the statement about an **individual root**, but it does not create evidence that a numerical root set is complete.

## Recognition pipeline

For each numerical root tuple, `recognize_system_roots` performs four steps:

1. pass each coordinate to `algrecognize` as an Arb/Acb enclosure at the solver's verification scale;
2. convert the recognized integer polynomial to exact SymPy roots;
3. select the exact root compatible with the validated numerical coordinate;
4. substitute the complete reconstructed tuple into every original equation and reduce those values exactly.

Thus scalar recognition and system certification are distinct. A scalar relation may be convincing while the assembled tuple still fails the original system.

## Core-solver results

<!-- algroots: execute -->
```python
import sympy as sp

from algroots import polysolve
from algroots.recognition import recognize_system_roots

x, y = sp.symbols("x y")
result = polysolve((x - y, y**2 - 2), (x, y), digits=70, recognize=False)
recognized = recognize_system_roots(
    result,
    max_degree=2,
    require_certified=True,
)

assert all(root.certified for root in recognized)
assert all(root.equation_values == (0, 0) for root in recognized)
```

For a conventional `PolynomialSystemRoots` or `AlgebraicSystemRoots`, the solver-level precision and verification settings apply to every returned root. The enclosure supplied to `algrecognize` is centered on the validated numerical coordinate with a radius derived from `verification_digits`; it is not a claim that all working digits are independently certified coordinate digits. Exact joint substitution remains the final system-level certification step.

## Monodromy results

`MonodromyOrbitResult` carries aligned `MonodromyRootInfo` records. Recognition uses each root's own `verification_digits` when creating the `algrecognize` input and uses its working `digits` when matching the recognized polynomial back to the correct exact root.

| Field | Meaning |
|---|---|
| `digits` | numerical working precision used for the retained root |
| `verification_digits` | residual-verification target used for that root |
| `relative_residual` | observed scale-normalized base-system residual |
| `verified` | whether numerical base-system verification succeeded |
| `source_loop` | first loop that discovered the root, or `None` for a seed |
| `path_index` | path that supplied the retained representative |

`digits` is **not** a claim that every working digit is correct. In particular, recognition uses the more conservative `verification_digits` as its trusted input precision.

## Certification and completeness are independent

A returned `RecognizedSystemRoot` has two relevant properties:

- `scalar_certified`: every coordinate recognizer reports a certified isolated algebraic relation;
- `jointly_certified`: exact substitution of the reconstructed tuple makes every original equation zero.

`certified` is true only when both are true.

For monodromy, this says nothing about whether all roots have been found. An orbit with `completeness_basis="none"` or `"statistical"` can contain individually exact-certified roots. Recognition never changes `stopping_reason` or `completeness_basis`.

## Choosing recognition bounds

| Option | Controls | Increase when |
|---|---|---|
| `max_degree` | largest algebraic degree considered | the expected number field has higher degree |
| `max_height` | coefficient-height search bound | the minimal polynomial has larger coefficients |
| `min_relation_bits` | minimum relation-quality requirement | weak or spurious relations must be rejected more aggressively |
| `degree_cost` | penalty against higher-degree explanations | lower-degree relations should be preferred when several fit |

These parameters control the recognition search; they are not substitutes for numerical verification or exact joint substitution.

## Failure modes

Recognition can fail because no relation satisfies the requested bounds, because the numerical coordinate does not uniquely identify one exact root of the recognized polynomial, or because the reconstructed tuple fails exact substitution. With `require_certified=True`, incomplete scalar or joint certification raises `ExactCertificationError` rather than returning a result that looks stronger than the evidence supports.
