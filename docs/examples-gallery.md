# Examples gallery

The [examples catalog](../examples/README.md) contains 24 standalone public-API scripts. Each supplies its own imports/input, prints results and asserts its mathematical or evidence contract. Timings are not fixed outputs; example 16 checks certified canonical order.

| Goal | Runnable examples |
|---|---|
| Start solving | [Polynomials](../examples/01_polynomial_system.py), [branches/poles](../examples/02_branches_and_poles.py) |
| Obtain exact coordinates | [Recognition](../examples/03_exact_recognition.py), [exact RUR](../examples/05_exact_rur.py), [algebraic fields](../examples/06_algebraic_coefficients.py) |
| Inspect finite algebra | [Quotient/cache](../examples/04_quotient_and_cache.py), [border bases](../examples/10_border_bases.py) |
| Understand proofs and multiplicities | [Boxes/multiplicity](../examples/07_multiplicity_and_boxes.py), [clusters](../examples/08_clustered_roots.py), [deflation](../examples/11_singular_deflation.py), [repeated views/order](../examples/16_multiplicity_views_and_order.py) |
| Diagnose cost or refusal | [Presolve/costs](../examples/09_presolve_and_costs.py), [limits/failures](../examples/15_limits_and_failures.py) |
| Explore continuation | [Bounded recovery](../examples/12_bounded_homotopy_recovery.py), [automatic charts](../examples/13_projective_chart_switching.py), [seeded monodromy](../examples/14_seeded_monodromy.py) |

```bash
python examples/run_all.py --list
python examples/run_all.py 01_polynomial_system.py 07_multiplicity_and_boxes.py
```

Read [Reading Results](reading-results.md) for evidence semantics, and the [End-to-End Worked Example](end-to-end-example.md) for a longer algebraic tutorial.

For constructive generated subfields and rational function-field maps, run [example 17](../examples/17_generated_subfields.py) and see [Generated subfields and function fields](function-fields.md).

[Example 18](../examples/18_normalized_curve_rationality.py) covers singular normalization, positive genus and ground-field point obstructions.

[Example 19](../examples/19_functional_decomposition.py) searches rational compositions and reconstructs a supplied algebraic factor with exact certificates.

[Example 20](../examples/20_automatic_algebraic_discovery.py) discovers algebraic coefficient extensions and factors automatically, with exact orbit coverage and complete versus partial decomposition search results.

[Example 21](../examples/21_multivariate_luroth.py) constructs multivariate Lüroth generators, verifies both coordinate maps, and distinguishes proper subfields from fields of higher transcendence degree.

[Example 22](../examples/22_multivariate_decomposition.py) searches uni-multivariate rational decompositions, including GRS02 Example 2.1, and checks bounded and exhaustive search results.

[Example 23](../examples/23_decomposition_chains.py) constructs recursive rational and implicit chains, replays their composed certificates, and reports open leaves.

[Example 24](../examples/24_finite_algebras.py) demonstrates finite-field construction, nilradicals, reduced quotients and primary decomposition of nonmonogenic finite algebras.
