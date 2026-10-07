# Examples gallery

The [examples catalog](../examples/README.md) contains 16 standalone public-API
scripts. Each supplies its own imports/input, prints results and asserts its
mathematical or evidence contract. Timings are not fixed outputs; example 16 checks certified canonical order.

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

Read [Reading Results](reading-results.md) for evidence semantics, and the
[End-to-End Worked Example](end-to-end-example.md) for a longer algebraic tutorial.
