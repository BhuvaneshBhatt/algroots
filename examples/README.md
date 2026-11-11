# Runnable algroots examples

Install a native wheel, or build from the repository root with `ALGROOTS_BUILD_NATIVE_F4=1 ALGROOTS_BUILD_FF_SPARSE=1 ALGROOTS_BUILD_NATIVE_EXTENSION=1 python -m pip install -e ".[test,dev]"` and a configured FLINT dependency prefix. The three portable examples select SymPy explicitly. Run any script directly, or use the catalog runner:

```bash
python examples/run_all.py --list
python examples/run_all.py
python examples/run_all.py polynomial_system.py limits_and_failures.py
```

Each script uses public APIs, supplies its own symbols and inputs, prints results, and asserts meaningful mathematical or evidence properties. The runner stops on failure. Continuation examples can take longer; every script is tested in a fresh process with a timeout. No script downloads data or writes output files. Backend timing is not asserted. The multiplicity and ordering example checks certified canonical ordering.

| Script | Demonstrates | Proof boundary |
|---|---|---|
| [polynomial_system.py](polynomial_system.py) | Solve a coupled polynomial system | Conditional finite-root extraction |
| [branches_and_poles.py](branches_and_poles.py) | Retain principal branches and exclude poles | Original expression and domain checks |
| [exact_recognition.py](exact_recognition.py) | Recognize and jointly certify numerical coordinates | Coordinate and joint equation certification |
| [quotient_and_cache.py](quotient_and_cache.py) | Inspect exact quotient structure and bounded caches | Exact finite algebra |
| [exact_rur.py](exact_rur.py) | Construct an RUR and extract exact points | Exact algebraic coordinates |
| [algebraic_coefficients.py](algebraic_coefficients.py) | Solve and certify an algebraic-field system | Exact point certificate; automatic boxes remain rational-only |
| [multiplicity_and_boxes.py](multiplicity_and_boxes.py) | Certify individual multiplicities with automatic boxes | Exact isolated endpoint proofs |
| [clustered_roots.py](clustered_roots.py) | Resolve closely spaced exact roots | Distinct exact endpoint proofs |
| [presolve_and_costs.py](presolve_and_costs.py) | Inspect cancellation-aware presolve and observed costs | Exact substitution; timings are observations |
| [border_bases.py](border_bases.py) | Compare exact border-basis constructions | Exact quotient relations and commutation |
| [singular_deflation.py](singular_deflation.py) | Deflate and replay an exact singular-root proof | Local exact deflation; finite ideal required |
| [bounded_homotopy_recovery.py](bounded_homotopy_recovery.py) | Recover and certify a singular finite endpoint | Exact finite-root accounting; no path proof |
| [projective_chart_switching.py](projective_chart_switching.py) | Track a point through an automatic chart change | Numerical projective tracking only |
| [seeded_monodromy.py](seeded_monodromy.py) | Discover a quadratic orbit from one supplied seed | Numerical orbit discovery; no automatic exact count |
| [limits_and_failures.py](limits_and_failures.py) | Handle unsupported problems and proof budgets | Explicit refusal; no weakened proof |

| [multiplicity_views_and_order.py](multiplicity_views_and_order.py) | Select repeated-root views and canonical order | Exact local multiplicities and ordering |
| [generated_subfields.py](generated_subfields.py) | Generate fixed-base and constant-generated subfields | Exact membership and embeddings; rational maps hold on generic charts |
| [normalized_curve_rationality.py](normalized_curve_rationality.py) | Normalize singular curves and distinguish conic rationality from genus zero | Exact genus, maps and QQ point obstructions |
| [functional_decomposition.py](functional_decomposition.py) | Search rational compositions and reconstruct a supplied algebraic factor | Exact identities and explicit rational-search completeness |
| [automatic_algebraic_discovery.py](automatic_algebraic_discovery.py) | Discover coefficient extensions and algebraic factors automatically | Exact factor-orbit coverage and explicit search completeness |
| [multivariate_luroth.py](multivariate_luroth.py) | Construct multivariate Lüroth generators and test membership | Exact maps and constant/rational/higher-degree decisions |
| [multivariate_decomposition.py](multivariate_decomposition.py) | Search multivariate rational functional compositions | Exact classes, bounded prefixes and completeness replay |
| [decomposition_chains.py](decomposition_chains.py) | Build recursive rational and implicit decomposition chains | Exact compositions and resolved versus open leaves |
| [finite_algebras.py](finite_algebras.py) | Construct finite fields, reduced quotients and primary factors | Exact finite-algebra invariants |
| [groebner_operations.py](groebner_operations.py) | Reduce, eliminate, and change order | Exact basis and membership |
| [solver_settings.py](solver_settings.py) | Group settings and supervise a calculation | Verified result or deadline failure |

| [coefficient_fields.py](coefficient_fields.py) | Use coefficient fields and keep exceptional fibers | Verified bases and specialization guards |

Start with 01 and 02. Read [result semantics](https://github.com/BhuvaneshBhatt/algroots/blob/main/docs/reading-results.md) before using certificate or multiplicity fields. Examples 12–14 distinguish finite endpoint proofs from numerical path/orbit evidence. Example 15 catches an unsupported problem and demonstrates a proof-budget refusal.

For continuation controls, see the [recovery budgets](https://github.com/BhuvaneshBhatt/algroots/blob/main/docs/recovery-workflow.md#recovery-budgets). For result-wide claims, see [completeness evidence](https://github.com/BhuvaneshBhatt/algroots/blob/main/docs/reading-results.md#completeness-evidence).

The documentation [gallery](https://github.com/BhuvaneshBhatt/algroots/blob/main/docs/examples-gallery.md) provides a task-oriented index. The examples serve as executable tutorials instead of performance benchmarks.
