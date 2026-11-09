"""Bounded modular interpolation with FLINT linear algebra and sparse supports.

Sparse reconstruction restricts a shifted rational function to lines. Normalizing
all line denominators at their common regular anchor makes each coefficient a
homogeneous polynomial in the line direction. Prony recurrences recover those
supports from geometric directions; independent samples check the proposal.
These modular identities supply candidates, not exact characteristic-zero proofs.
"""

from math import comb
from random import Random

from flint import nmod_mat, nmod_poly

from .native_f4 import NativeF4LimitError


class UnluckyInterpolation(ArithmeticError):
    """A modular anchor, line, or exponent encoding cannot recover a stable image."""


def monomials(count, degree):
    result = [()]
    for _ in range(count):
        result = [p + (e,) for p in result for e in range(degree - sum(p) + 1)]
    return tuple(sorted(result, key=lambda m: (sum(m), m)))


def power(monomial, point, prime):
    value = 1
    for exponent, coordinate in zip(monomial, point, strict=True):
        value = value * pow(coordinate, exponent, prime) % prime
    return value


def evaluate(terms, point, prime):
    return sum(c * power(m, point, prime) for m, c in terms.items()) % prime


def rational_interpolate(points, values, prime, count, max_degree, max_terms):
    """Fit a rational function, reserving the last two samples for validation."""
    if not any(values):
        return {}, {(0,) * count: 1}
    for degree in range(max_degree + 1):
        width = 2 * comb(degree + count, count)
        if len(points) - 2 < width - 1:
            continue
        if width > max_terms:
            raise NativeF4LimitError("rational interpolation term budget exceeded")
        support = monomials(count, degree)
        rows = []
        for point, value in zip(points[:-2], values[:-2], strict=True):
            powers = [power(m, point, prime) for m in support]
            rows.append(powers + [(-value * v) % prime for v in powers])
        nullspace, nullity = nmod_mat(rows, prime).nullspace()
        if nullity != 1:
            continue
        vector = [int(nullspace[i, 0]) for i in range(width)]
        numerator = {m: c for m, c in zip(support, vector[: len(support)], strict=True) if c}
        denominator = {m: c for m, c in zip(support, vector[len(support) :], strict=True) if c}
        if not denominator:
            continue
        scale = pow(denominator[max(denominator)], -1, prime)
        numerator = {m: c * scale % prime for m, c in numerator.items()}
        denominator = {m: c * scale % prime for m, c in denominator.items()}
        if all(
            (den := evaluate(denominator, point, prime))
            and evaluate(numerator, point, prime) == value * den % prime
            for point, value in zip(points, values, strict=True)
        ):
            return numerator, denominator
    return None


def _prony(values, prime, support, max_terms):
    """Recover sparse polynomial coefficients from a checked Hankel recurrence."""
    if not any(values):
        return {}
    for rank in range(1, min(max_terms, (len(values) - 2) // 2) + 1):
        rows = [values[k : k + rank + 1] for k in range(len(values) - rank)]
        nullspace, nullity = nmod_mat(rows, prime).nullspace()
        if nullity != 1 or not int(nullspace[rank, 0]):
            continue
        recurrence = nmod_poly([int(nullspace[i, 0]) for i in range(rank + 1)], prime)
        factors = recurrence.factor()[1]
        if any(f.degree() != 1 or multiplicity != 1 for f, multiplicity in factors):
            continue
        roots = [(-int(f[0]) * pow(int(f[1]), -1, prime)) % prime for f, _ in factors]
        if len(roots) != rank:
            continue
        if any(root not in support or support[root] is None for root in roots):
            raise UnluckyInterpolation(
                "geometric exponent encodings collide or leave the degree bound"
            )
        matrix = nmod_mat([[pow(root, k, prime) for root in roots] for k in range(rank)], prime)
        coefficients = matrix.solve(nmod_mat([[v] for v in values[:rank]], prime))
        coefficients = [int(coefficients[k, 0]) for k in range(rank)]
        if all(
            sum(c * pow(root, k, prime) for c, root in zip(coefficients, roots, strict=True))
            % prime
            == value
            for k, value in enumerate(values)
        ):
            return {support[root]: c for root, c in zip(roots, coefficients, strict=True) if c}
    return None


def _translate(terms, shift, prime, max_terms):
    """Expand a sparse translation using modular binomial coefficients."""
    output = {}
    for monomial, coefficient in terms.items():
        expanded = {(): coefficient}
        for exponent, coordinate in zip(monomial, shift, strict=True):
            if not coordinate:
                expanded = {prefix + (exponent,): c for prefix, c in expanded.items()}
                continue
            if len(expanded) > max_terms // (exponent + 1):
                raise NativeF4LimitError("sparse translation term budget exceeded")
            expanded = {
                prefix + (e,): c * comb(exponent, e) * pow(coordinate, exponent - e, prime) % prime
                for prefix, c in expanded.items()
                for e in range(exponent + 1)
            }
        for m, c in expanded.items():
            output[m] = (output.get(m, 0) + c) % prime
        output = {m: c for m, c in output.items() if c}
        if len(output) > max_terms:
            raise NativeF4LimitError("sparse translation term budget exceeded")
    return output


def sparse_interpolate(sample, count, prime, limits, stats):
    """Recover many rational coordinates together from shared modular evaluations.

    ``sample(point)`` returns a sparse mapping from arbitrary coordinate keys to
    modular scalar values. Its caller enforces image signatures and point budgets.
    Missing coordinates are zero. Support and line-column bounds apply throughout
    reconstruction and translation; a separate enumeration bound caps decoding.
    """
    degree = limits.max_parameter_degree
    total = comb(count + degree, count)
    if total > limits.max_sparse_monomials:
        raise NativeF4LimitError("sparse exponent decoding budget exceeded")
    if prime <= 2 * degree + 3:
        raise UnluckyInterpolation("field is too small for distinct line samples")
    rng = Random(prime + 91)
    anchor = tuple(rng.randrange(prime) for _ in range(count))
    sample(anchor)
    # Random exponent encodings avoid the exponential degree of a Kronecker
    # substitution. A collision affecting a recovered support rejects the image.
    generators = tuple(rng.randrange(1, prime) for _ in range(count))
    dictionaries = [dict() for _ in range(degree + 1)]
    for monomial in monomials(count, degree):
        value = power(monomial, generators, prime)
        dictionary = dictionaries[sum(monomial)]
        dictionary[value] = monomial if value not in dictionary else None
    scalars = [0]
    while len(scalars) < 2 * degree + 3:
        value = rng.randrange(1, prime)
        if value not in scalars:
            scalars.append(value)
    lines = []
    keys = set()
    rank_bound = 1
    while rank_bound <= limits.max_interpolation_terms:
        needed = 2 * rank_bound + 2
        while len(lines) < needed:
            k = len(lines)
            direction = tuple(pow(g, k, prime) for g in generators)
            points, images = [], []
            for scalar in scalars:
                point = tuple(
                    (a + scalar * v) % prime for a, v in zip(anchor, direction, strict=True)
                )
                image = sample(point)
                points.append((scalar,))
                images.append(image)
                keys.update(image)
                if len(points) < 3:
                    continue
                recovered = {
                    key: rational_interpolate(
                        points,
                        [row.get(key, 0) for row in images],
                        prime,
                        1,
                        degree,
                        limits.max_interpolation_terms,
                    )
                    for key in keys
                }
                if any(value is None for value in recovered.values()):
                    continue
                line = {}
                for key, (numerator, denominator) in recovered.items():
                    constant = denominator.get((0,), 0)
                    if not constant:
                        raise UnluckyInterpolation("line denominator vanishes at its common anchor")
                    inverse = pow(constant, -1, prime)
                    line[key] = tuple(
                        {m[0]: c * inverse % prime for m, c in part.items()}
                        for part in (numerator, denominator)
                    )
                lines.append(line)
                break
            else:
                raise NativeF4LimitError("parameter reconstruction degree budget exceeded")
        result = {}
        for key in keys:
            parts = []
            for part in (0, 1):
                polynomial = {}
                for d in range(degree + 1):
                    values = [line.get(key, ({}, {0: 1}))[part].get(d, 0) for line in lines]
                    recovered = _prony(
                        values, prime, dictionaries[d], limits.max_interpolation_terms
                    )
                    if recovered is None:
                        break
                    polynomial.update(recovered)
                else:
                    parts.append(
                        _translate(
                            polynomial,
                            tuple(-a % prime for a in anchor),
                            prime,
                            limits.max_interpolation_terms,
                        )
                    )
                    continue
                break
            if len(parts) != 2 or not parts[1]:
                break
            if len(parts[0]) + len(parts[1]) > limits.max_interpolation_terms:
                raise NativeF4LimitError("sparse rational interpolation term budget exceeded")
            scale = pow(parts[1][max(parts[1])], -1, prime)
            result[key] = tuple({m: c * scale % prime for m, c in part.items()} for part in parts)
        else:
            for _ in range(3):
                point = tuple(rng.randrange(prime) for _ in range(count))
                actual = sample(point)
                if set(actual) - result.keys():
                    break
                if any(
                    not (den := evaluate(denominator, point, prime))
                    or evaluate(numerator, point, prime) != actual.get(key, 0) * den % prime
                    for key, (numerator, denominator) in result.items()
                ):
                    break
            else:
                stats["sparse_lines"] += len(lines)
                stats["sparse_terms"] += sum(len(p) for parts in result.values() for p in parts)
                return result
        rank_bound *= 2
    raise NativeF4LimitError("sparse interpolation support budget exceeded")
