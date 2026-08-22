# Total-Degree Homotopy Continuation

`polysolve(..., method="homotopy")` is an explicit numerical backend for **regular square zero-dimensional polynomial systems**. It is deliberately not part of `method="auto"` yet.

## Start system and path count

For a target system

$$
F=(f_1,\ldots,f_n), \qquad d_i=\deg(f_i),
$$

`algroots` uses the total-degree start system

$$
G_i(x)=x_i^{d_i}-1.
$$

Its roots are products of roots of unity, so all start points are known explicitly. The number of paths is the Bézout count

$$
D=\prod_{i=1}^n d_i.
$$

This is a path count, not necessarily the number of finite affine solutions of the target. Some paths can go to infinity, and several paths can coalesce at a singular target root.

## Gamma homotopy

For each start point, the backend tracks

$$
H_i(x,t)=(1-t)\,\gamma\,G_i(x)+t f_i(x), \qquad 0\le t\le1.
$$

The gamma trick is generic rather than universal: exceptional gamma values can produce singular or numerically poor paths. `algroots` therefore tries a deterministic sequence of exact complex rational gamma values controlled by `homotopy_seed` and `homotopy_gamma_attempts`.

<!-- algroots: execute -->
```python
import sympy as sp
from algroots import polysolve

x, y = sp.symbols("x y")
result = polysolve(
    (x**2 - 1, y**2 - 1),
    (x, y),
    method="homotopy",
    digits=35,
    recognize=False,
)

assert len(result.roots) == 4
assert result.homotopy_paths_total == 4
assert result.homotopy_paths_succeeded == 4
```

## Verification and regularity

Every successful path endpoint is re-refined against the target system and checked with the same scale-normalized residual criterion used by the other numerical backends. The target Jacobian is also evaluated in scaled coordinates. Results retain:

| Field | Meaning |
|---|---|
| `homotopy_paths_total` | total-degree Bézout path count |
| `homotopy_paths_succeeded` | paths reaching finite verified endpoints for the successful gamma |
| `homotopy_paths_failed` | failed paths for the successful gamma |
| `homotopy_paths_divergent` | paths heuristically classified as tending to infinity |
| `homotopy_gamma` | exact gamma used by the successful attempt |
| `homotopy_gamma_attempts` | number of gamma candidates tried |
| `homotopy_endpoint_smallest_singular_values` | scaled endpoint Jacobian smallest singular values |
| `homotopy_endpoint_condition_estimates` | scaled endpoint Jacobian condition estimates |

Very small Jacobian singular values are not rejected solely because they are small. The endpoint is re-refined at increased precision to distinguish a stable small nonzero singular value from one collapsing toward zero at a singular root.

## Parallel tracking

Independent start paths can be distributed across worker processes:

```python
result = polysolve(
    equations,
    variables,
    method="homotopy",
    homotopy_parallel=True,
    homotopy_max_workers=4,
)
```

Each worker compiles the symbolic homotopy once and then tracks multiple start points. Start points are generated lazily rather than materialized as one large Cartesian product.

## Algebraic equations

`algsolve(..., method="homotopy")` first algebraizes supported rational functions/rational powers/radicals, tracks the augmented polynomial system, projects endpoints back to the original variables, and verifies the original branch/domain semantics. In that case the path counts describe the **augmented polynomial cover**. `homotopy_projected_candidates` and `homotopy_projected_roots_rejected` describe what happens during projection/filtering.

## Current scope and failure semantics

The current backend is intentionally conservative. It is intended for regular square systems whose required paths terminate at distinct finite nonsingular roots. It raises `HomotopySolveError` rather than returning a potentially incomplete root set when:

- a required path cannot be tracked successfully;
- the affine path appears to diverge;
- distinct start paths coalesce at a singular endpoint;
- endpoint regularity cannot be resolved within the precision cap.

Not yet implemented:

- singular endpoint endgames;
- deflation;
- projective path tracking;
- multihomogeneous start systems;
- sparse/polyhedral homotopy;
- automatic selection of homotopy by `method="auto"`.

For the generic path-tracking engine and monodromy loops, see [Continuation and Monodromy](continuation-and-monodromy.md).
