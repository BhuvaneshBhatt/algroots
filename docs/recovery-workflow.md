# Bounded recovery workflow

Ordinary `method="homotopy"` requires finite regular endpoints for its required
Bézout paths. For finite rational square systems with singular or difficult
endpoints, request `homotopy_recovery=True`. Recovery computes an original-system
finite quotient, attempts an affine prefix and Cauchy endgame, and falls back to
projective chart tracking. Certified singular endpoints undergo bounded exact
deflation. It is sequential and is not selected by `method="auto"`.

<!-- algroots: execute -->
```python
import sympy as sp
from algroots import HomotopyRecoveryOptions, polysolve

x = sp.Symbol("x")
result = polysolve(
    (x**2,),
    (x,),
    method="homotopy",
    homotopy_recovery=True,
    recovery_options=HomotopyRecoveryOptions(max_paths=4, gamma_attempts=1),
    digits=30,
    recognize=False,
)
assert len(result.roots) == 1 and result.completeness.status == "certified"
assert result.root_certifications[0].certificate.multiplicity == 2
```

The proof is finite-root accounting from exact endpoint certificates and the
exact quotient geometric count. It does not certify numerical paths, roots at
infinity or global projective completeness. A deflation stage limit does not
invalidate an already proved endpoint.

## Recovery budgets

The table lists defaults of `HomotopyRecoveryOptions`. Solver-wide path, gamma,
quotient/proof and precision limits also apply; the tighter applicable bound wins.

| Option | Default | What it limits |
|---|---:|---|
| `max_paths` | 64 | Bézout start paths per gamma |
| `gamma_attempts` | 2 | Different gamma homotopies |
| `max_quotient_dimension` | 128 | Original finite quotient |
| `path_steps` | 600 | Tracker steps per segment |
| `max_segments` | 256 | Projective segments per path attempt |
| `max_chart_switches` | 32 | Projective chart changes |
| `endgame_samples` | 16 | Samples per endgame circle |
| `endgame_cycles` | 4 | Cycle search bound |
| `endgame_levels` | 3 | Radius levels |
| `max_deflation_stages` | 4 | Local deflation stages |
| `max_deflation_equations` | 128 | Added deflation equations |
| `max_retry_rounds` | 1 | Additional attempts per gamma |
| `retry_precision_growth` | 2 | Working-precision multiplier |

All options are integers; counts are positive except `max_retry_rounds`, which
permits zero. Samples must be at least eight, levels at least two and precision
growth at least two. Bool values are refused.

## Diagnose retries and failures

`recovery_records` report path/gamma index, stages, numerical outcome, chart
switches, cycle, phase seconds, retry round/reason, working precision, certification
status and retry disposition. Unproved candidates are retracked at higher precision
and correction accuracy within `max_precision_digits`; successful proofs and
exact extraction kernels are reused. Projective segment/chart budget failures
and numerical near-infinity classifications do not cause precision-only retries.
A new gamma reconsiders all starts because path identities can change.

If completeness is not proved, `HomotopySolveError` carries `recovery_records`,
`certification_attempts` and `certified_endpoints`. These prove only the recorded
partial endpoints. The solver never returns that partial set as a complete result.
Inspect the failure before increasing the relevant bound. Near-infinity is a
numerical classification, not a certified point at infinity.

See the [recovery example](../examples/12_bounded_homotopy_recovery.py),
[chart example](../examples/13_projective_chart_switching.py),
[Homotopy Continuation](homotopy-continuation.md) and [API](api.md).
