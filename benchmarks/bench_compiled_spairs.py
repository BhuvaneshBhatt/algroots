"""Native-versus-Python batched sparse S-polynomial support arithmetic."""

import json
import random
import statistics
import time
from pathlib import Path

from algroots.ff_sparse_matrix import batch_spolynomials, compiled_available


def main():
    if not compiled_available():
        raise RuntimeError("build the optional native extension before benchmarking")
    rng = random.Random(1117)
    prime = 1009
    polynomials = []
    for _ in range(50):
        powers = set()
        while len(powers) < 40:
            powers.add(tuple(rng.randrange(0, 5) for _ in range(5)))
        polynomials.append({power: rng.randrange(1, prime) for power in powers})
    descriptors = []
    for _ in range(200):
        a, b = rng.sample(range(len(polynomials)), 2)
        descriptors.append(
            (
                a,
                b,
                tuple(rng.randrange(0, 4) for _ in range(5)),
                tuple(rng.randrange(0, 4) for _ in range(5)),
                rng.randrange(1, prime),
                rng.randrange(1, prime),
            )
        )
    samples = {}
    for backend in ("python", "compiled"):
        timings = []
        for _rep in range(7):
            start = time.perf_counter()
            result = batch_spolynomials(polynomials, descriptors, prime, backend=backend)
            timings.append(time.perf_counter() - start)
            if backend == "python":
                reference = result
            else:
                assert result == reference
        samples[backend] = timings
    result = dict(
        prime=prime,
        polynomials=len(polynomials),
        terms_per_polynomial=40,
        pairs=len(descriptors),
        samples=samples,
        medians={key: statistics.median(value) for key, value in samples.items()},
    )
    Path("bench_compiled_spairs.json").write_text(json.dumps(result, indent=2))
    print(result["medians"])


if __name__ == "__main__":
    main()
