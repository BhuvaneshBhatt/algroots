# Gröbner benchmarks

`bench_cross_cas.py` compares the fixed native F4 kernel with SymPy, Singular and Macaulay2 on identical exact inputs. Candidate computation, interpreter startup and independent verification are recorded separately. Missing executables, timeouts and failed checks are retained in the output.

```sh
python benchmarks/bench_cross_cas.py --cases cyclic5 noon4 katsura5 --primes 101 1009 --orders grevlex --repetitions 5 --verify --output results.json
```

Native computations use `f4_tiled_finish`. Rational runs select candidate-only reconstruction or exact certification with `--qq-mode`; that choice controls verification, not the basis algorithm. `gb_case_catalog.py` records all polynomial definitions and source provenance. The certificate helpers check ideal membership, reducedness and S-pair identities independently of the native kernel.
