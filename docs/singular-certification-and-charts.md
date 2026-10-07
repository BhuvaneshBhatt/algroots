# Singular certification, deflation and chart switching

These APIs separate exact endpoint facts from numerical path evidence. An exact
root certificate can establish existence, isolation, singularity and individual
multiplicity without certifying which homotopy path led to that root or whether
all paths were found.

## Exact isolated-root certificates

`certify_isolated_root(equations, variables, point)` accepts exact algebraic
coefficients and exact algebraic coordinates, including `CRootOf`. It builds a
finite quotient from the original equations, checks every equation in an exact
algebraic field, computes exact Jacobian rank, and uses a separating multiplication
operator's characteristic polynomial to determine the local algebra dimension.
The eigenvalue's order in that polynomial is the individual root multiplicity.

The entire input ideal must be zero-dimensional. Approximate coordinates, floats,
unit ideals, positive-dimensional ideals and exhausted dimension budgets are
rejected. This certifier proves an exact isolated point; it does not attach an
interval claim to an unrelated floating candidate.

`IsolatedRootCertificate.verify()` independently rebuilds the proof from the
original equations and detects altered proof fields. Its `status` is `certified`;
`is_singular` compares the exact Jacobian rank with the number of variables.

<!-- algroots: execute -->
```python
import sympy as sp
from algroots.certification import certify_isolated_root
from algroots.deflation import deflate_isolated_root

x, y = sp.symbols("x y")
certificate = certify_isolated_root((x**2 - y, y**2), (x, y), (0, 0))
assert certificate.multiplicity == 4
assert certificate.jacobian_rank == 1
assert certificate.is_singular and certificate.verify()
deflation = deflate_isolated_root(certificate.equations, (x, y), (0, 0))
assert deflation.regular and deflation.verify()
```

## Rigorous isolation of an approximate candidate

`certify_root_box` proves that an **open** rational complex coordinate box contains
exactly one geometric root. `RationalComplexBox.bounds` consists of one tuple
`(real_lower, real_upper, imaginary_lower, imaginary_upper)` per coordinate.
Every endpoint must be an exact rational and both intervals must have positive
width. This API currently requires rational polynomial coefficients.

The proof counts roots of the squarefree separator polynomial in the linear
image of the input box using exact complex root isolation. It then refines the
parameter's rational isolating interval/rectangle and propagates it through the
exact RUR coordinate maps with rational rectangle arithmetic. Every coordinate
must be proved strictly inside the supplied box. A coarse separator image,
boundary root, empty box or insufficient refinement is rejected. Coarse-box
rejection is conservative: it need not mean the original coordinate box actually
contains multiple roots.

<!-- algroots: execute -->
```python
from algroots.certification import RationalComplexBox, certify_root_box

box = RationalComplexBox(((1, 2, -sp.Rational(1, 10), sp.Rational(1, 10)),))
certificate = certify_root_box(((x**2 - 2) ** 2,), (x,), box)
assert certificate.multiplicity == 2
assert certificate.is_singular and certificate.verify()
```

`cauchy_endgame(..., certify=True)` applies this proof to a converged numerical
endpoint. It constructs a rational box around the exact binary value of the
numerical estimate unless `certification_box` is supplied. The numerical estimate
must lie inside that box. `status="certified_endpoint"` means a unique exact target
root was proved in the recorded box; the numerical path and Puiseux cycle number
remain numerical. Certification failure raises `RootCertificationError` rather
than silently substituting a residual check. A nonconverged endgame returns no
certificate. The target parameter must be exact algebraic; rational-box
certification additionally requires rational specialized equations.

Pass `deflate=True` together with `certify=True` to attach targeted exact deflation.
The result's `certificate` and `deflation` records are distinct from its numerical
`endpoint` and sampled error estimate.

## Targeted determinantal deflation

`deflate_isolated_root` first certifies the original exact root. At each stage it
chooses an exact nonzero rank-r pivot minor of the Jacobian and adds the bordering
(r+1)-minors. These polynomial equations impose rank <=r in the pivot's local
chart. They vanish at the exact target and retain all original equations. No new
variables or variable-dependent divisions are introduced.

Full column Jacobian rank is checked exactly before `DeflationResult.regular`
becomes true. `stages` records pivot rows/columns, exact pivot value, added
equations and ranks. `max_stages` and `max_added_equations` produce an unresolved
result with an explicit stopping reason when exhausted. `verify()` replays both
the original certificate and deflation construction.

Deflation is **local and targeted**: it may remove other original roots. Solving
the deflated equations is not a substitute for enumerating the original system.
`DeflationResult.refine(approximate)` performs numerical overdetermined Newton
refinement and reports original and deflated residuals. Its convergence flag does
not certify association of the returned approximation with the exact target;
use an exact root box for that association.

The determinantal construction follows the rank-stratum principle used in
[Akoglu, Hauenstein and Szanto](https://arxiv.org/abs/1408.2721). This implementation
does not claim their full hybrid RUR reconstruction/alpha-theory algorithm.

## Automatic projective chart switching

`projective_homotopy` homogenizes the polynomial family. Its `system` uses the
specified fixed patch and remains suitable for direct `track_path` use.
`track_projective_path` adds adaptive coordinate charts while retaining the
original homogeneous equations. It normalizes into a usable coordinate chart,
tracks short segments, and switches to a dominant coordinate with hysteresis.
If a segment fails near a chart boundary, an accepted prefix can be resumed in a
new chart; otherwise the segment shrinks. Invalid seeds and controls are rejected.
Segment and switch budgets yield an explicit unresolved result.

<!-- algroots: execute -->
```python
from algroots.endgames import projective_homotopy
from algroots.projective_tracking import track_projective_path

t = sp.Symbol("t")
projective = projective_homotopy(((1 - t) * x - 1,), (x,), t, patch=(0, 1))
path = track_projective_path(projective, projective.lift((1,)))
assert path.success
assert path.chart_switches
assert path.classification == "numerically_near_infinity"
```

`ProjectivePathResult` records the final homogeneous point/patch, all segment
results and `ChartSwitch` events. Finite endpoints can be dehomogenized with
`affine_endpoint()`. A small homogenizing coordinate is classified as
`numerically_near_infinity`, not as certified infinity. Actual projective
singularities may still defeat numerical continuation. No interval tube,
certified path tracking, global projective root count or completeness proof is
claimed.

The high-level `polysolve(method="homotopy")` portfolio retains its finite regular
endpoint contract. With `homotopy_recovery=True`, bounded rational finite-system recovery orchestrates
these APIs and proves finite-root accounting. Numerical paths and infinity remain
uncertified. See [Recovery Workflow](recovery-workflow.md).
