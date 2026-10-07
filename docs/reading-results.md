# Reading results

`polysolve` returns `PolynomialSystemRoots`. `algsolve` returns
`AlgebraicSystemRoots`, which also describes the augmented polynomial system,
branch filtering and nonzero constraints. Use attributes rather than parsing
printed representations.

## Coordinates and counts

`roots` contains distinct numerical tuples in the order of `variables`. The root
list uses coordinate lexicographic ordering, with `ordering` distinguishing exact
ordering from numerical fallback. `quotient_dimension` and `total_multiplicity` count
algebraic multiplicity; `geometric_solution_count` counts distinct points.
`is_radical` and `has_multiple_roots` describe the exact finite polynomial ideal.
`multiplicities` and `multiplicity_evidence` report aligned individual counts;
unknown counts remain `None`. The default view is distinct points. With
`root_mode="with_multiplicity"`, iteration and `output_roots` repeat established
points while the underlying `roots` and metadata remain distinct.

<!-- algroots: execute -->
```python
import sympy as sp
from algroots import polysolve

x = sp.Symbol("x")
result = polysolve((x**2 * (x - 1),), (x,), recognize=False, certify="required")
assert len(result.roots) == result.geometric_solution_count == 2
assert result.total_multiplicity == result.quotient_dimension == 3
multiplicities = {
    a.certificate.point: a.certificate.multiplicity for a in result.root_certifications
}
assert multiplicities == {(sp.S.Zero,): 2, (sp.S.One,): 1}
```

For nontrivial algebraic branch/pole projections, polynomial-cover metadata does
not automatically describe the retained original roots. Unknown original-system
multiplicity/radical fields remain `None`; do not substitute the cover dimension.

## Completeness evidence

`result.completeness` contains `status`, `basis`, `expected_count`,
`distinct_roots`, `quotient_dimension` and `notes`.

| Status | Interpretation |
|---|---|
| `conditional` | Exact polynomial structure/count with numerical extraction assumptions |
| `numerical` | Numerical ordinary-homotopy accounting or polynomial-cover branch/pole filtering |
| `unknown` | No applicable enumeration evidence has been established |
| `certified` | Exact empty-ideal evidence, or complete finite-root accounting from exact endpoint proofs in bounded recovery |

Read `basis` and `notes` as well as `status`. There is no public `complete` Boolean.
Per-root recognition and box certificates are separate from global evidence and
do not automatically change that status. See [Completeness versus Verification](completeness-versus-verification.md).

## Recognition, proofs and diagnostics

| Attribute | Purpose |
|---|---|
| `diagnostics`, `max_relative_residual` | Numerical validation; a small residual alone is not a proof |
| `recognized_roots`, `recognition_error` | Default best-effort exact coordinate recognition and joint equation checks |
| `root_certifications` | Optional aligned exact endpoint attempts; inspect each status/error before accessing a certificate |
| `root_deflations` | Bounded-recovery deflation results; `None` for regular endpoints |
| `recovery_records` | Numerical path stages, precision, retry and proof outcomes |
| `cost_diagnostics` | Observed phase/cache/quotient costs and structural hints, never evidence of correctness |

Optional collections can be `None` when the operation was not requested. Nested
phase timings overlap: `solve.groebner` is part of `solve`; trace-pairing time can
include trace-vector work. Do not sum overlapping entries as elapsed total time.

See [Certification Workflow](certification-workflow.md), [Recovery Workflow](recovery-workflow.md)
and [examples](../examples/README.md) for executable uses.
