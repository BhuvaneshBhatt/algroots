# Runnable algroots examples

Install the package from the repository root with `python -m pip install -e ".[test,dev]"`.
Run any script directly, or use the catalog runner:

```bash
python examples/run_all.py --list
python examples/run_all.py
python examples/run_all.py 01_polynomial_system.py 15_limits_and_failures.py
```

Each script uses public APIs, supplies its own symbols and inputs, prints results,
and asserts meaningful mathematical or evidence properties. The runner stops on
failure. Continuation examples can take longer; every script is tested in a fresh
process with a timeout. No script downloads data or writes output files.
Backend timing is not asserted. Example 16 checks certified canonical ordering.

| Script | Demonstrates | Proof boundary |
|---|---|---|
| [01_polynomial_system.py](01_polynomial_system.py) | Solve a coupled polynomial system | Conditional finite-root extraction |
| [02_branches_and_poles.py](02_branches_and_poles.py) | Retain principal branches and exclude poles | Original expression and domain checks |
| [03_exact_recognition.py](03_exact_recognition.py) | Recognize and jointly certify numerical coordinates | Coordinate and joint equation certification |
| [04_quotient_and_cache.py](04_quotient_and_cache.py) | Inspect exact quotient structure and bounded caches | Exact finite algebra |
| [05_exact_rur.py](05_exact_rur.py) | Construct an RUR and extract exact points | Exact algebraic coordinates |
| [06_algebraic_coefficients.py](06_algebraic_coefficients.py) | Solve and certify an algebraic-field system | Exact point certificate; automatic boxes remain rational-only |
| [07_multiplicity_and_boxes.py](07_multiplicity_and_boxes.py) | Certify individual multiplicities with automatic boxes | Exact isolated endpoint proofs |
| [08_clustered_roots.py](08_clustered_roots.py) | Resolve closely spaced exact roots | Distinct exact endpoint proofs |
| [09_presolve_and_costs.py](09_presolve_and_costs.py) | Inspect cancellation-aware presolve and observed costs | Exact substitution; timings are observations |
| [10_border_bases.py](10_border_bases.py) | Compare exact border-basis constructions | Exact quotient relations and commutation |
| [11_singular_deflation.py](11_singular_deflation.py) | Deflate and replay an exact singular-root proof | Local exact deflation; finite ideal required |
| [12_bounded_homotopy_recovery.py](12_bounded_homotopy_recovery.py) | Recover and certify a singular finite endpoint | Exact finite-root accounting; no path proof |
| [13_projective_chart_switching.py](13_projective_chart_switching.py) | Track a point through an automatic chart change | Numerical projective tracking only |
| [14_seeded_monodromy.py](14_seeded_monodromy.py) | Discover a quadratic orbit from one supplied seed | Numerical orbit discovery; no automatic exact count |
| [15_limits_and_failures.py](15_limits_and_failures.py) | Handle unsupported problems and proof budgets | Explicit refusal; no weakened proof |

| [16_multiplicity_views_and_order.py](16_multiplicity_views_and_order.py) | Select repeated-root views and canonical order | Exact local multiplicities and ordering |

Start with 01 and 02. Read [result semantics](../docs/reading-results.md) before
using certificate or multiplicity fields. Examples 12–14 distinguish finite
endpoint proofs from numerical path/orbit evidence. Example 15 catches an
unsupported problem and demonstrates a proof-budget refusal.

For continuation controls, see the [recovery budgets](../docs/recovery-workflow.md#recovery-budgets).
For result-wide claims, see [completeness evidence](../docs/reading-results.md#completeness-evidence).

The documentation [gallery](../docs/examples-gallery.md) provides a task-oriented
index. The examples serve as executable tutorials rather than performance benchmarks.
