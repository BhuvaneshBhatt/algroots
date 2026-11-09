"""Reproducible exact polynomial systems for native F4 performance experiments.

No SymPy, FLINT, or native extension is imported by this module. All inputs
are integer-coefficient sparse polynomials mapping exponent tuples to integers.
For computations in GF(p), coefficients are reduced by the benchmark worker.

Sources:
* Alexander Maletzky, Isabelle AFP, Theory Benchmarks:
  https://isa-afp.org/browser_info/current/AFP/Groebner_Bases/Benchmarks.html
  (cyclic, Katsura, Eco, Noon definitions)
* Macaulay2 ExampleSystems, PoSSo Caprasse (equations cross-checked against
  user-provided regression input:, test supplied fixture b1sn37).
* Exact regression systems, with source identifiers below.
* Generic determinantal ideals and seeded random homogeneous systems: locally
  generated using the explicitly documented constructions below.

A suite name is a selection of experiments, *not* a mathematical invariant.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from itertools import combinations, permutations
from math import comb

from gb_published_additions import SYSTEMS, exact_lift_primes, published_case

Sparse = dict[tuple[int, ...], int]

AFP = "https://isa-afp.org/browser_info/current/AFP/Groebner_Bases/Benchmarks.html"
M2 = "https://macaulay2.com/doc/Macaulay2/share/doc/Macaulay2/ExampleSystems/html/_caprasse.html"


@dataclass(frozen=True)
class GBCase:
    name: str
    family: str
    nvars: int
    polynomials: tuple[Sparse, ...]
    source: str
    note: str = ""


def _poly(nvars: int, terms: list[tuple[int, tuple[int, ...]]]) -> Sparse:
    result: Sparse = {}
    for coeff, ids in terms:
        mono = [0] * nvars
        for idx in ids:
            if not 0 <= idx < nvars:
                raise ValueError("Variable index out of range")
            mono[idx] += 1
        key = tuple(mono)
        result[key] = result.get(key, 0) + coeff
    return {m: c for m, c in result.items() if c}


def cyclic(n: int) -> tuple[Sparse, ...]:
    if not 2 <= n <= 14:
        raise ValueError("2 <= cyclic n <= 14")
    result = []
    for degree in range(1, n):
        result.append(
            _poly(
                n,
                [(1, tuple((start + shift) % n for shift in range(degree))) for start in range(n)],
            )
        )
    result.append(_poly(n, [(1, tuple(range(n))), (-1, ())]))
    return tuple(result)


def katsura(n: int) -> tuple[Sparse, ...]:
    """Isabelle AFP katsura(n): n+1 variables and n+1 polynomials."""
    if not 1 <= n <= 12:
        raise ValueError("1 <= Katsura n <= 12")
    v = n + 1
    result = [_poly(v, [(1, (0,)), (-1, ())] + [(2, (i,)) for i in range(1, v)])]
    for i in range(n):
        terms = [(-1, (i,))]
        for j in range(-n, n + 1):
            if abs(i - j) <= n:
                terms.append((1, (abs(j), abs(i - j))))
        result.append(_poly(v, terms))
    return tuple(result)


def eco(n: int) -> tuple[Sparse, ...]:
    """Isabelle AFP eco(n): n equations and n variables, degree <= 3."""
    if not 2 <= n <= 14:
        raise ValueError("2 <= Eco n <= 14")
    m = n - 1
    result = [
        _poly(n, [(1, (j,)) for j in range(m)] + [(1, ())]),
        _poly(n, [(1, (m - 1, m)), (-m, ())]),
    ]
    for i in reversed(range(m - 1)):
        terms = [(1, (i, m))]
        terms.extend((1, (j, j + i + 1, m)) for j in range(m - i - 1))
        result.append(_poly(n, terms))
    return tuple(result)


def noon(n: int) -> tuple[Sparse, ...]:
    """Isabelle AFP noon(n), cyclic variable ordering p1,p0,p2,... ."""
    if not 2 <= n <= 14:
        raise ValueError("2 <= Noon n <= 14")
    result = []
    for i in (1, 0, *range(2, n)):
        terms = [(10, ()), (-11, (i,))]
        terms.extend((10, (j, j, i)) for j in range(n) if j != i)
        result.append(_poly(n, terms))
    return tuple(result)


def _det_sign(order: tuple[int, ...]) -> int:
    inv = sum(order[i] > order[j] for i in range(len(order)) for j in range(i + 1, len(order)))
    return -1 if inv & 1 else 1


def minors(rows: int, cols: int, size: int) -> tuple[Sparse, ...]:
    """All size-by-size minors of a *generic* rows x cols matrix."""
    if not (1 <= size <= min(rows, cols) and 2 <= rows <= 6 and 2 <= cols <= 6):
        raise ValueError("Unsupported generic matrix or minor size")
    n = rows * cols
    result = []
    for rr in combinations(range(rows), size):
        for cc in combinations(range(cols), size):
            terms = [
                (_det_sign(order), tuple(rr[i] * cols + cc[order[i]] for i in range(size)))
                for order in permutations(range(size))
            ]
            result.append(_poly(n, terms))
    return tuple(result)


def random_homogeneous(
    n: int, generators: int, degree: int, density: str, seed: int
) -> tuple[Sparse, ...]:
    """Deterministic homogeneous supports; dense uses all degree-d monomials.

    Sparse includes min(M,max(n+1,min(8,ceil(0.2*M)))) terms out of M possible
    with a seeded uniform sample, or M if fewer. Coefficients uniformly
    sampled from {-5,...,-1,1,...,5}. The seed is local; no global PRNG state.
    """
    if not (
        2 <= n <= 9
        and 1 <= generators <= 9
        and 1 <= degree <= 5
        and density in ("dense", "sparse")
        and 0 <= seed <= 2**32 - 1
    ):
        raise ValueError("Invalid random homogeneous parameters")
    from itertools import combinations_with_replacement

    universe = tuple(combinations_with_replacement(range(n), degree))
    total = comb(n + degree - 1, degree)
    assert len(universe) == total
    count = total if density == "dense" else min(total, max(n + 1, min(8, (total + 4) // 5)))
    rng = random.Random(seed)
    result = []
    for _ in range(generators):
        support = universe if density == "dense" else sorted(rng.sample(universe, count))
        result.append(
            _poly(n, [(rng.choice((-5, -4, -3, -2, -1, 1, 2, 3, 4, 5)), t) for t in support])
        )
    return tuple(result)


def _fixed_case(name: str) -> GBCase:
    """Exact polynomial regression systems."""
    if name == "trott_bivariate":
        return GBCase(
            name,
            "trott",
            2,
            (
                _poly(2, [(1, (0, 0)), (3, (0,)), (4, (1,))]),
                _poly(2, [(2, (1, 1)), (4, (0,)), (5, (1,)), (9, ())]),
            ),
            "regression input: supplied fixture yhep1c",
        )
    if name == "trott_spheres":
        return GBCase(
            name,
            "trott",
            3,
            tuple(
                _poly(3, [(1, (i, i)), (1, (j, j)), (-1, ())]) for i, j in ((0, 1), (0, 2), (1, 2))
            ),
            "regression input: supplied fixture 93og2e",
        )
    if name == "trott_elimination":
        return GBCase(
            name,
            "trott",
            3,
            (
                _poly(3, [(1, (0,)), (1, (1,)), (1, (2,))]),
                _poly(3, [(1, (0,)), (-2, (1,)), (1, (2, 2, 2))]),
                _poly(3, [(1, (0, 0)), (-2, (1, 1, 1)), (1, (2,))]),
            ),
            "regression input: supplied fixture 3g968r",
        )
    if name == "walk_ideal":
        return GBCase(
            name,
            "walk_ideal",
            3,
            (
                _poly(3, [(1, (0, 0)), (-3, (0, 1)), (5, (1,)), (1, (2, 2))]),
                _poly(3, [(1, (0, 0, 0)), (-1, (1, 1, 2)), (4, (0, 1, 2)), (-3, ())]),
            ),
            "regression input: supplied fixture b9t05d",
            "Polynomial input and grevlex basis only; not a test of Gröbner walk transitions.",
        )
    if name == "interreduce_unit":
        return GBCase(
            name,
            "regression",
            2,
            (_poly(2, [(1, (0, 0))]), _poly(2, [(1, ()), (-1, (0, 1))])),
            "regression input: supplied fixture lilv1i-bug-63493",
            "Ideal equals (1), for every field and monomial order.",
        )
    if name == "posso_butcher":
        # PoSSo Butcher equations from Macaulay2/M2 ExampleSystems/butcher.m2.
        # Variables are ordered (z,y,t,u,v,w,x), as in the source ring.
        # Each rational polynomial was multiplied by its own LCM denominator:
        # (2,3,6,4,8,12,24). Do not compare these lifts at p=2 or p=3.
        polys = (
            {
                (1, 0, 0, 1, 0, 0, 0): 2,
                (0, 1, 0, 0, 1, 0, 0): 2,
                (0, 0, 1, 0, 0, 1, 0): 2,
                (0, 0, 0, 0, 0, 2, 0): -2,
                (0, 0, 0, 0, 0, 1, 0): -1,
                (0, 0, 0, 0, 0, 0, 0): -1,
            },
            {
                (1, 0, 0, 2, 0, 0, 0): 3,
                (0, 1, 0, 0, 2, 0, 0): 3,
                (0, 0, 1, 0, 0, 2, 0): -3,
                (0, 0, 1, 0, 0, 0, 0): -1,
                (0, 0, 0, 0, 0, 3, 0): 3,
                (0, 0, 0, 0, 0, 2, 0): 3,
                (0, 0, 0, 0, 0, 1, 0): 4,
            },
            {
                (1, 0, 0, 0, 1, 0, 1): 6,
                (0, 0, 1, 0, 0, 2, 0): -6,
                (0, 0, 1, 0, 0, 1, 0): -3,
                (0, 0, 1, 0, 0, 0, 0): -1,
                (0, 0, 0, 0, 0, 3, 0): 6,
                (0, 0, 0, 0, 0, 2, 0): 6,
                (0, 0, 0, 0, 0, 1, 0): 4,
            },
            {
                (1, 0, 0, 3, 0, 0, 0): 4,
                (0, 1, 0, 0, 3, 0, 0): 4,
                (0, 0, 1, 0, 0, 3, 0): 4,
                (0, 0, 1, 0, 0, 1, 0): 4,
                (0, 0, 0, 0, 0, 4, 0): -4,
                (0, 0, 0, 0, 0, 3, 0): -6,
                (0, 0, 0, 0, 0, 2, 0): -10,
                (0, 0, 0, 0, 0, 1, 0): -1,
                (0, 0, 0, 0, 0, 0, 0): -1,
            },
            {
                (1, 0, 0, 1, 1, 0, 1): 8,
                (0, 0, 1, 0, 0, 3, 0): 8,
                (0, 0, 1, 0, 0, 2, 0): 4,
                (0, 0, 1, 0, 0, 1, 0): 4,
                (0, 0, 0, 0, 0, 4, 0): -8,
                (0, 0, 0, 0, 0, 3, 0): -12,
                (0, 0, 0, 0, 0, 2, 0): -14,
                (0, 0, 0, 0, 0, 1, 0): -3,
                (0, 0, 0, 0, 0, 0, 0): -1,
            },
            {
                (1, 0, 0, 0, 2, 0, 1): 12,
                (0, 0, 1, 0, 0, 3, 0): 12,
                (0, 0, 1, 0, 0, 2, 0): 12,
                (0, 0, 1, 0, 0, 1, 0): 8,
                (0, 0, 0, 0, 0, 4, 0): -12,
                (0, 0, 0, 0, 0, 3, 0): -18,
                (0, 0, 0, 0, 0, 2, 0): -14,
                (0, 0, 0, 0, 0, 1, 0): -1,
                (0, 0, 0, 0, 0, 0, 0): -1,
            },
            {
                (0, 0, 1, 0, 0, 3, 0): -24,
                (0, 0, 1, 0, 0, 2, 0): -24,
                (0, 0, 1, 0, 0, 1, 0): -8,
                (0, 0, 0, 0, 0, 4, 0): 24,
                (0, 0, 0, 0, 0, 3, 0): 36,
                (0, 0, 0, 0, 0, 2, 0): 26,
                (0, 0, 0, 0, 0, 1, 0): 7,
                (0, 0, 0, 0, 0, 0, 0): 1,
            },
        )
        return GBCase(
            name,
            "posso_butcher",
            7,
            polys,
            "https://github.com/Macaulay2/M2/blob/stable/M2/Macaulay2/packages/ExampleSystems/butcher.m2",
            "Original PoSSo Butcher system, integer denominator-cleared form",
        )
    if name == "posso_cassou":
        # Exact integer transcription from Macaulay2/M2 ExampleSystems/cassou.m2,
        # variables ordered (b,c,d,e). Exponents are listed directly to
        # avoid requiring SymPy/FLINT to import the case catalog.
        polynomials = (
            {
                (4, 3, 0, 0): 6,
                (4, 2, 1, 0): 21,
                (4, 1, 2, 0): 15,
                (4, 0, 3, 0): 9,
                (2, 2, 0, 1): -8,
                (2, 1, 1, 1): -28,
                (2, 1, 0, 0): -144,
                (2, 0, 2, 1): 36,
                (2, 0, 1, 0): -648,
                (0, 0, 0, 0): -120,
            },
            {
                (4, 4, 0, 0): 9,
                (4, 3, 1, 0): 30,
                (4, 2, 2, 0): 39,
                (4, 1, 3, 0): 18,
                (2, 3, 0, 1): -24,
                (2, 2, 1, 1): -16,
                (2, 2, 0, 0): -432,
                (2, 1, 2, 1): 16,
                (2, 1, 1, 0): -720,
                (2, 0, 3, 1): 24,
                (2, 0, 2, 0): -432,
                (0, 2, 0, 2): 16,
                (0, 1, 1, 2): -32,
                (0, 1, 0, 1): 576,
                (0, 1, 0, 0): -240,
                (0, 0, 2, 2): 16,
                (0, 0, 1, 1): -576,
                (0, 0, 0, 0): 5184,
            },
            {
                (2, 3, 0, 1): -15,
                (2, 2, 1, 1): 15,
                (2, 2, 0, 0): -81,
                (2, 1, 1, 0): 216,
                (2, 0, 2, 0): -162,
                (0, 2, 0, 2): 40,
                (0, 1, 1, 2): -80,
                (0, 1, 0, 1): 1008,
                (0, 0, 2, 2): 40,
                (0, 0, 1, 1): -1008,
                (0, 0, 0, 0): 5184,
            },
            {
                (2, 2, 0, 0): -4,
                (2, 1, 1, 0): 4,
                (2, 0, 2, 0): -3,
                (0, 1, 0, 1): 22,
                (0, 0, 1, 1): -22,
                (0, 0, 0, 0): 261,
            },
        )
        return GBCase(
            name,
            "posso_cassou",
            4,
            polynomials,
            "https://github.com/Macaulay2/M2/blob/stable/M2/Macaulay2/packages/ExampleSystems/cassou.m2",
            "Original PoSSo Cassou-Noguès integer system; variables b,c,d,e",
        )
    if name == "posso_trinks":
        # Original six-variable PoSSo Trinks system, using (x,y,z,t,u,v).
        # Source: Macaulay2 ExampleSystems/trinks, equations printed over QQ.
        def P(terms):
            return _poly(6, terms)

        return GBCase(
            name,
            "posso_trinks",
            6,
            (
                P([(45, (1,)), (35, (4,)), (-165, (5,)), (-36, ())]),
                P([(35, (1,)), (25, (2,)), (40, (3,)), (-27, (4,))]),
                P([(25, (1, 4)), (-165, (5, 5)), (15, (0,)), (-18, (2,)), (30, (3,))]),
                P([(15, (1, 2)), (20, (3, 4)), (-9, (0,))]),
                P([(-11, (5, 5, 5)), (1, (0, 1)), (2, (2, 3))]),
                P([(-11, (4, 5)), (3, (5, 5)), (99, (0,))]),
            ),
            "https://macaulay2.com/doc/Macaulay2/share/doc/Macaulay2/ExampleSystems/html/_trinks.html",
            "Exact PoSSo Trinks equations; variable order x,y,z,t,u,v",
        )
    if name == "wood_rational":
        # Gradient system printed as decimals in M2 ExampleSystems/wood.
        # Here 110.1,9.9,100.1 are interpreted as exact rationals, with
        # denominators cleared separately per polynomial (not floating point).
        def P(terms):
            return _poly(4, terms)

        return GBCase(
            name,
            "wood",
            4,
            (
                P([(200, (0, 0, 0)), (-200, (0, 1)), (1, (0,)), (-1, ())]),
                P([(-1000, (0, 0)), (1101, (1,)), (99, (3,)), (-200, ())]),
                P([(180, (2, 2, 2)), (-180, (2, 3)), (1, (2,)), (-1, ())]),
                P([(-900, (2, 2)), (99, (1,)), (1001, (3,)), (-200, ())]),
            ),
            "https://macaulay2.com/doc/Macaulay2/share/doc/Macaulay2/ExampleSystems/html/_wood.html",
            "Exact rational lift of printed floating-point Wood system, denominators cleared",
        )
    if name == "toric_walk":
        # Coordinate order is d, c, b, a.
        def P(terms):
            return _poly(4, terms)

        return GBCase(
            name,
            "regression",
            4,
            (
                P([(3, (3, 3)), (-8, (2,))]),
                P([(1, (3, 2)), (-6, (1,))]),
                P([(1, (2, 2)), (-36, (0,))]),
                P([(1, (3, 1)), (-16, (0,))]),
                P([(1, (2, 1)), (-6, (3, 0))]),
                P([(3, (1, 1)), (-8, (2, 0))]),
            ),
            "regression input: supplied fixture z2n214-bug-60293",
            "Original toric-polynomial input (d,c,b,a); tests GB, not walk transitions",
        )
    if name == "opposite_linears":
        return GBCase(
            name,
            "regression",
            2,
            (
                _poly(2, [(1, (0,)), (-1, ())]),
                _poly(2, [(1, (0,)), (1, ())]),
            ),
            "regression input: supplied fixture 62x61h-bug-73475",
            "Unit ideal for odd characteristic; nonunit ideal (x+1) in characteristic two",
        )
    if name in ("frobenius_p2", "frobenius_p3"):
        # (x+y)^p - x^p - y^p, plus two finite-field constraints.
        # The binomial coefficients vanish exactly modulo p.
        from math import comb as binomial

        exponent = 2 if name.endswith("p2") else 3

        def P(terms):
            return _poly(2, terms)

        first = P(
            [(binomial(exponent, i), (0,) * (exponent - i) + (1,) * i) for i in range(1, exponent)]
        )
        return GBCase(
            name,
            "characteristic",
            2,
            (
                first,
                P([(1, (0, 0)), (1, (1,)), (-1, ())]),
                P([(1, (1, 1)), (-1, (0,))]),
            ),
            "Synthetic Frobenius characteristic stress: (x+y)^p-x^p-y^p",
            "First generator becomes the zero polynomial modulo the named prime",
        )
    if name == "nine_variable_fallback":
        # More than 8 exponent coordinates: packed 8x8-bit key is unavailable.
        def P(terms):
            return _poly(9, terms)

        gens = [P([(1, (i,)), (-1, (i + 1,))]) for i in range(8)]
        gens.append(P([(1, (8, 8)), (-1, (8,))]))
        return GBCase(
            name,
            "representation_boundary",
            9,
            tuple(gens),
            "Synthetic batch-local packed-key fallback (nvars > 8)",
        )
    if name == "degree256_fallback":
        # An exponent 256 cannot fit into one 8-bit packed field.
        def P(terms):
            return _poly(2, terms)

        return GBCase(
            name,
            "representation_boundary",
            2,
            (
                P([(1, tuple([0] * 256)), (-1, (0,))]),
                P([(1, (1,)), (-1, ())]),
            ),
            "Synthetic batch-local packed-key fallback (exponent 256)",
        )
    if name == "binomial_curve":

        def P(terms):
            return _poly(3, terms)

        return GBCase(
            name,
            "positive_dimensional",
            3,
            (
                P([(1, (0, 1)), (-1, (2, 2))]),
                P([(1, (0, 2)), (-1, (1, 1))]),
                P([(1, (1, 2)), (-1, (0, 0))]),
            ),
            "Synthetic symmetric binomial ideal; nonzero-dimensional stress",
        )
    if name == "triangular_finite":

        def P(terms):
            return _poly(3, terms)

        return GBCase(
            name,
            "triangular",
            3,
            (
                P([(1, (0, 0)), (-1, (1,))]),
                P([(1, (1, 1)), (-1, (2,))]),
                P([(1, (2, 2)), (-1, ())]),
            ),
            "Synthetic triangular zero-dimensional system; mixed degrees",
        )
    if name == "redundant_generators":

        def P(terms):
            return _poly(3, terms)

        return GBCase(
            name,
            "redundant",
            3,
            (
                P([(1, (0, 0)), (-1, (1,))]),
                P([(1, (1, 1)), (-1, (2,))]),
                P([(2, (0, 0)), (-2, (1,))]),
                P([(-3, (0, 0)), (3, (1,))]),
                P([(1, (0, 0, 1)), (-1, (1, 1))]),
            ),
            "Synthetic redundant generators: repeated multiples and scalar associates",
        )
    if name == "nilpotent_primary":

        def P(terms):
            return _poly(2, terms)

        return GBCase(
            name,
            "nonradical",
            2,
            (
                P([(1, (0, 0))]),
                P([(1, (0, 1))]),
                P([(1, (1, 1, 1))]),
            ),
            "Synthetic nonradical primary monomial ideal (x^2,xy,y^3)",
            "Nonreduced quotient; standard monomial count 4 over every field",
        )
    if name == "coordinate_axes":

        def P(terms):
            return _poly(3, terms)

        return GBCase(
            name,
            "squarefree_monomial",
            3,
            (
                P([(1, (0, 1))]),
                P([(1, (0, 2))]),
                P([(1, (1, 2))]),
            ),
            "Synthetic squarefree ideal (xy,xz,yz)",
            "Radical positive-dimensional union of coordinate axes",
        )
    if name == "cusp_jacobian":

        def P(terms):
            return _poly(2, terms)

        return GBCase(
            name,
            "characteristic",
            2,
            (
                P([(1, (1, 1)), (-1, (0, 0, 0))]),
                P([(-3, (0, 0))]),
                P([(2, (1,))]),
            ),
            "Synthetic cusp Jacobian ideal (y^2-x^3, -3x^2, 2y); p=2,3 degenerate",
        )
    if name in ("caprasse", "caprasse_modified"):
        # Variables (x,y,z,t), using exact input polynomials from the provided
        # regression input: test 80 and test 81. The modified variant has
        # rational shifts in the first and second equations; multiply each
        # equation by its nonzero denominator before reducing modulo p.
        def P(terms):
            return _poly(4, terms)

        f1 = [(-2, (0,)), (2, (3, 0, 1)), (-1, (2,)), (1, (1, 1, 2))]
        f2 = [
            (2, ()),
            (4, (0, 0)),
            (-10, (3, 1)),
            (4, (3, 0, 0, 1)),
            (-10, (1, 1)),
            (2, (3, 1, 1, 1)),
            (4, (0, 2)),
            (-1, (0, 0, 0, 2)),
            (4, (0, 1, 1, 2)),
        ]
        f3 = [(-1, (0,)), (1, (3, 3, 0)), (-2, (2,)), (2, (3, 1, 2))]
        f4 = [
            (2, ()),
            (-10, (3, 3)),
            (-10, (3, 1)),
            (2, (3, 3, 3, 1)),
            (4, (0, 2)),
            (4, (3, 3, 0, 2)),
            (4, (2, 2)),
            (4, (3, 1, 2, 2)),
            (-1, (0, 2, 2, 2)),
        ]
        if name == "caprasse_modified":
            f1 = [(coeff * 100000, ids) for coeff, ids in f1]
            f1[2] = (-100001, (2,))
            f2 = [(coeff * 1000000, ids) for coeff, ids in f2]
            f2[0] = (2000001, ())
        return GBCase(
            name,
            "caprasse",
            4,
            (P(f1), P(f2), P(f3), P(f4)),
            (
                "regression input: supplied fixture dw4941"
                if name.endswith("modified")
                else "regression input: supplied fixture b1sn37; " + M2
            ),
            "Rationally perturbed Caprasse; denominators cleared exactly"
            if name.endswith("modified")
            else "PoSSo Caprasse",
        )
    raise ValueError(f"Unknown fixed case {name!r}")


def build_case(name: str) -> GBCase:
    if name in SYSTEMS:
        nvars, polynomials, source, scales = published_case(name)
        return GBCase(
            name,
            "m2_application",
            nvars,
            polynomials,
            source,
            "Exact decimal-rational lift; integer denominator clearing "
            f"scales={scales}; noninvertible primes={exact_lift_primes(name)}",
        )
    m = re.fullmatch(r"(cyclic|katsura|eco|noon)([0-9]+)", name)
    if m:
        family, size = m.group(1), int(m.group(2))
        polynomials = {"cyclic": cyclic, "katsura": katsura, "eco": eco, "noon": noon}[family](size)
        return GBCase(
            name,
            family,
            len(next(iter(polynomials[0]))),
            polynomials,
            AFP,
            "Katsura argument n means n+1 variables" if family == "katsura" else "",
        )
    m = re.fullmatch(r"minors([0-9]+)x([0-9]+)_r([0-9]+)", name)
    if m:
        rows, cols, size = map(int, m.groups())
        return GBCase(
            name,
            "determinantal",
            rows * cols,
            minors(rows, cols, size),
            "Generic determinantal ideal; all minors, built by signed permutation expansion",
            f"{rows}x{cols} matrix; all {size}x{size} minors",
        )
    m = re.fullmatch(r"random_(dense|sparse)_n([0-9]+)_g([0-9]+)_d([0-9]+)_s([0-9]+)", name)
    if m:
        density, n, g, d, seed = m.groups()
        n, g, d, seed = map(int, (n, g, d, seed))
        return GBCase(
            name,
            f"random_{density}",
            n,
            random_homogeneous(n, g, d, density, seed),
            "Seeded synthetic homogeneous system (algorithm and seed documented in gb_case_catalog.py)",
            f"n={n}, generators={g}, degree={d}, density={density}, seed={seed}",
        )
    return _fixed_case(name)


SUITES: dict[str, tuple[str, ...]] = {
    "smoke": (
        "trott_bivariate",
        "trott_spheres",
        "interreduce_unit",
        "walk_ideal",
        "eco3",
        "eco4",
        "noon3",
        "minors2x3_r2",
        "minors3x3_r3",
        "random_dense_n3_g3_d2_s17",
        "random_sparse_n4_g3_d2_s17",
        "cyclic4",
        "katsura3",
        "opposite_linears",
        "frobenius_p2",
        "frobenius_p3",
        "nine_variable_fallback",
        "triangular_finite",
        "redundant_generators",
        "nilpotent_primary",
        "coordinate_axes",
        "m2_lorentz",
    ),
    "core": (
        "trott_elimination",
        "caprasse",
        "eco5",
        "noon4",
        "minors2x4_r2",
        "minors3x4_r3",
        "random_dense_n4_g4_d2_s17",
        "random_sparse_n5_g4_d3_s17",
        "cyclic5",
        "katsura5",
        "wood_rational",
        "toric_walk",
        "degree256_fallback",
        "binomial_curve",
        "cusp_jacobian",
        "posso_trinks",
        "m2_reimer5",
        "m2_boon_rational",
    ),
    "stress": (
        "caprasse_modified",
        "eco6",
        "eco7",
        "noon5",
        "noon6",
        "minors3x4_r2",
        "minors3x5_r3",
        "random_dense_n5_g5_d3_s17",
        "random_sparse_n6_g5_d3_s17",
        "cyclic6",
        "cyclic7",
        "cyclic8",
        "katsura7",
        "katsura8",
        "posso_cassou",
        "posso_butcher",
        "m2_puma_decimal",
        "m2_chemical_decimal",
        "m2_rose_rational",
    ),
}


def names_for_suite(suite: str) -> tuple[str, ...]:
    if suite == "all":
        return tuple(name for tier in SUITES.values() for name in tier)
    return SUITES[suite]
