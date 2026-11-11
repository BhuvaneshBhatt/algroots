"""Native modular images and FLINT interpolation over exact coefficient fields.

Parameters are evaluated only for candidate construction. Reconstructed bases
retain the original field; public callers independently verify ideal equality.
An image, a stable support, or fresh evaluation checks alone are not a proof.
"""

from dataclasses import dataclass, field
from math import comb
from random import Random
from time import perf_counter

import sympy as sp
from flint import nmod_poly
from sympy.polys.groebnertools import groebner as reference_groebner
from sympy.polys.rings import PolyRing

from .algebraic_function_fields import AlgebraicFunctionField
from .domains import field_element
from .finite_fields import FiniteFieldExtension
from .groebner_errors import GroebnerBackendUnavailable, GroebnerVerificationError
from .modular_arithmetic import rational_reconstruction
from .native_f4 import _MAX_NATIVE_PRIME, NativeF4LimitError, NativeF4Limits, native_prime_basis
from .parameter_interpolation import (
    UnluckyInterpolation,
    rational_interpolate,
    sparse_interpolate,
)
from .parameter_interpolation import (
    evaluate as _evaluate,
)
from .parameter_interpolation import (
    power as _power,
)
from .profiling import record_native_stats


def native_finite_basis(polynomials, ring, *, limits=None, certify=True):
    """Compute directly in a validated GF(p^d) using the FLINT extension kernel.

    Coefficients retain the domain's specified power basis and irreducible
    modulus. There is no reconstruction or change of characteristic. Public
    construction verifies the candidate independently in the original field.
    """
    if not isinstance(ring.domain, FiniteFieldExtension):
        raise TypeError("a validated finite extension coefficient field is required")
    if limits is None:
        limits = NativeF4Limits()
    if not isinstance(limits, NativeF4Limits):
        raise TypeError("limits must be NativeF4Limits")
    try:
        from . import _native_extension
    except ImportError as exc:
        raise GroebnerBackendUnavailable(
            "finite extension fields require the FLINT native extension; "
            "build with ALGROOTS_BUILD_NATIVE_EXTENSION=1 or use backend='sympy'"
        ) from exc
    domain = ring.domain
    prime = int(domain.characteristic())
    inputs = tuple(polynomials)
    sparse = [
        {
            tuple(m): tuple(int(c) % prime for c in reversed(value.rep.to_list()))
            for m, value in polynomial.items()
        }
        for polynomial in inputs
    ]
    modulus = tuple(int(c) % prime for c in reversed(domain.modulus.all_coeffs()))
    started = perf_counter()
    try:
        rows, profile = _native_extension.compute(
            sparse,
            prime,
            modulus,
            len(ring.symbols),
            str(ring.order),
            max_pairs=limits.max_pairs,
            max_terms=limits.max_terms,
            max_basis=limits.max_basis,
            batch_size=limits.batch_size,
            max_reduction_steps=limits.max_reduction_steps,
            max_field_degree=limits.max_field_degree,
            max_matrix_cells=limits.max_field_matrix_cells,
            collect_batch_metrics=limits.collect_batch_metrics,
        )
    except _native_extension.LimitError as exc:
        raise NativeF4LimitError(str(exc)) from exc
    profile["kind"] = "finite_extension"
    profile["native_call_seconds"] = perf_counter() - started
    record_native_stats(profile)
    basis = tuple(
        ring.from_dict(
            {
                tuple(m): domain.convert(
                    sum(c * domain.symbol**i for i, c in enumerate(coordinates))
                )
                for m, coordinates in row.items()
            }
        )
        for row in rows
    )
    if certify and basis != tuple(reference_groebner(list(inputs), ring)):
        raise GroebnerVerificationError("finite extension basis failed independent verification")
    return basis


class _UnluckyImage(ArithmeticError):
    pass


@dataclass
class NativeFieldResult:
    """A reconstructed candidate with image statistics and verification status."""

    basis: tuple
    primes: tuple[int, ...]
    stats: dict = field(default_factory=dict)
    verified: bool = False


def _mod_rational(value, prime):
    value = sp.Rational(value)
    denominator = int(value.q) % prime
    if not denominator:
        raise _UnluckyImage("coefficient denominator vanishes")
    return int(value.p) % prime * pow(denominator, -1, prime) % prime


class _FieldModel:
    """Exact input coefficients represented independently of coordinate monomials."""

    def __init__(self, polynomials, ring):
        self.ring = ring
        original_tower = ring.domain if isinstance(ring.domain, AlgebraicFunctionField) else None
        if original_tower:
            self.tower, self.alpha, self._tower_conversion = original_tower.rational_model()
        else:
            self.tower = None
        if self.tower:
            self.parameters = self.tower.parameters
            self.ground = self.tower.domain
            if self.parameters:
                self.ground = self.ground.domain
        elif ring.domain.is_FractionField:
            self.parameters = tuple(ring.domain.symbols)
            self.ground = ring.domain.domain
        else:
            self.parameters = ()
            self.ground = ring.domain
        if self.ground != sp.QQ and not self.ground.is_AlgebraicField:
            raise ValueError("native field images require rational or number-field constants")
        self.extension = bool(self.tower) or self.ground.is_AlgebraicField
        self.degree = (
            self.tower.rank
            if self.tower
            else self.ground.mod.degree()
            if self.ground.is_AlgebraicField
            else 1
        )
        if not self.tower:
            self.alpha = self.ground.ext.as_expr() if self.ground.is_AlgebraicField else sp.S.One
        self._contexts = {}
        self._modular_inputs = {}
        self._prime_rings = {}
        self.polynomials = tuple(
            {tuple(m): self.coefficient(c) for m, c in p.items()} for p in polynomials
        )

    def coefficient(self, value):
        """Separate parameter support from the constant power-basis coordinates.

        Tower coefficients share a scalar denominator so specialization cannot
        change their relative coordinates in a split modular component.
        """
        if self.tower:
            value = self._tower_conversion(value)
            coordinates = list(reversed(value.rep.to_list()))
            coordinates += [self.tower.domain.zero] * (self.degree - len(coordinates))
            # A common scalar denominator preserves the entire power-basis vector
            # through specialization, including in split modular components.
            denominator = self.tower.domain.one
            if self.parameters:
                denominator = self.tower.domain.field.ring.one
                for c in coordinates:
                    denominator = denominator.lcm(c.denom)
                numerators = [c.numer * (denominator.exquo(c.denom)) for c in coordinates]
                support = set().union(*(set(c) for c in numerators))
                numerator = {
                    m: tuple(c.get(m, self.ground.zero) for c in numerators) for m in support
                }
                denominator = {
                    m: (c,) + (self.ground.zero,) * (self.degree - 1)
                    for m, c in denominator.items()
                }
            else:
                numerator = {(): tuple(coordinates)}
                denominator = {(): (self.ground.one,) + (self.ground.zero,) * (self.degree - 1)}
            return numerator, denominator
        if self.parameters:
            return dict(value.numer.items()), dict(value.denom.items())
        return {(): value}, {(): self.ground.one}

    def sample(self, prime, point, limits, stats):
        """Compute one modular fiber and accumulate image time even on rejection."""
        started = perf_counter()
        try:
            if self.extension:
                return self._extension_sample(prime, point, limits, stats)
            return self._rational_sample(prime, point, limits, stats)
        finally:
            stats["modular_image_seconds"] += perf_counter() - started

    def _modular_input(self, prime):
        """Convert exact coefficients once per characteristic, before point evaluation."""
        if prime not in self._modular_inputs:

            def convert(terms):
                if self.ground == sp.QQ and not self.tower:
                    return {m: _mod_rational(c, prime) for m, c in terms.items()}
                return {
                    m: tuple(_mod_rational(c, prime) for c in self._coordinates(value))
                    for m, value in terms.items()
                }

            self._modular_inputs[prime] = tuple(
                {m: (convert(num), convert(den)) for m, (num, den) in p.items()}
                for p in self.polynomials
            )
        return self._modular_inputs[prime]

    def _prime_ring(self, prime):
        if prime not in self._prime_rings:
            self._prime_rings[prime] = PolyRing(self.ring.symbols, sp.GF(prime), self.ring.order)
        return self._prime_rings[prime]

    def _rational_sample(self, prime, point, limits, stats):
        sparse = []
        for polynomial in self._modular_input(prime):
            row = {}
            for monomial, (numerator, denominator) in polynomial.items():
                num = _evaluate(numerator, point, prime)
                den = _evaluate(denominator, point, prime)
                if not den:
                    raise _UnluckyImage("parameter denominator vanishes")
                value = num * pow(den, -1, prime) % prime
                if value:
                    row[monomial] = value
            sparse.append(row)
        finite = self._prime_ring(prime)
        inputs = tuple(finite.from_dict(p) for p in sparse)
        basis, profile = native_prime_basis(inputs, finite, prime, limits=limits)
        stats["native_calls"] += 1
        stats["processed_pairs"] += profile.get(
            "pairs_processed", profile.get("processed_pairs", 0)
        )
        if limits.collect_batch_metrics:
            stats["images"].append({"prime": prime, "point": point, "profile": profile})
        signature = tuple(tuple(p.LM) for p in basis)
        data = tuple({tuple(m): (int(c) % prime,) for m, c in p.items()} for p in basis)
        return signature, data

    def _coordinates(self, coefficient):
        if self.tower:
            return coefficient
        if self.ground == sp.QQ:
            return (coefficient,)
        values = tuple(reversed(coefficient.to_list()))
        return values + (sp.QQ.zero,) * (self.degree - len(values))

    def _context(self, prime, point=()):
        """Build coprime modular factors and CRT idempotents, rejecting ramification."""
        key = (prime, point) if self.tower else prime
        if key not in self._contexts:
            if self.tower:
                coefficients = tuple(reversed(self.tower.modulus.rep.to_list()))

                def evaluate(value):
                    if not self.parameters:
                        return _mod_rational(value, prime)
                    num = _evaluate(
                        {m: _mod_rational(c, prime) for m, c in value.numer.items()}, point, prime
                    )
                    den = _evaluate(
                        {m: _mod_rational(c, prime) for m, c in value.denom.items()}, point, prime
                    )
                    if not den:
                        raise _UnluckyImage("algebraic modulus has a pole")
                    return num * pow(den, -1, prime) % prime

                modulus = nmod_poly([evaluate(c) for c in coefficients], prime)
            else:
                coefficients = tuple(reversed(self.ground.mod.to_list()))
                modulus = nmod_poly([_mod_rational(c, prime) for c in coefficients], prime)
            if modulus.degree() != self.degree or modulus.gcd(modulus.derivative()).degree() != 0:
                raise _UnluckyImage("minimal polynomial loses degree or squarefreeness")
            factors = tuple(
                factor for factor, multiplicity in modulus.factor()[1] if multiplicity == 1
            )
            if sum(f.degree() for f in factors) != self.degree:
                raise _UnluckyImage("minimal polynomial has repeated modular factors")
            components = []
            for factor in factors:
                quotient = divmod(modulus, factor)[0]
                gcd, inverse, _ = quotient.xgcd(factor)
                if gcd != nmod_poly([1], prime):
                    raise _UnluckyImage("modular factors are not coprime")
                components.append((factor, (quotient * inverse) % modulus))
            # Parameter-dependent contexts are needed only for this evaluation.
            # Keeping all samples would retain every factorization and CRT table.
            if self.tower:
                self._contexts.clear()
            self._contexts[key] = modulus, tuple(components)
        return self._contexts[key]

    def _parameter_polynomial(self, terms, point, prime):
        result = [0] * self.degree
        for exponent, coefficient in terms.items():
            power = _power(exponent, point, prime)
            for index, value in enumerate(coefficient):
                result[index] = (result[index] + power * value) % prime
        return nmod_poly(result, prime)

    def _extension_sample(self, prime, point, limits, stats):
        """Evaluate power-basis coefficients once and glue all modular component bases."""
        modulus, components = self._context(prime, point)
        evaluated = tuple(
            {
                m: (
                    self._parameter_polynomial(num, point, prime),
                    self._parameter_polynomial(den, point, prime),
                )
                for m, (num, den) in polynomial.items()
            }
            for polynomial in self._modular_input(prime)
        )
        images = []
        signature = None
        for factor, weight in components:
            sparse = []
            for polynomial in evaluated:
                row = {}
                for monomial, (numerator, denominator) in polynomial.items():
                    num = numerator % factor
                    den = denominator % factor
                    gcd, inverse, _ = den.xgcd(factor)
                    if gcd != nmod_poly([1], prime):
                        raise _UnluckyImage(
                            "coefficient denominator is not invertible in a modular component"
                        )
                    coefficient = num * inverse % factor
                    if coefficient:
                        row[monomial] = tuple(int(coefficient[i]) for i in range(factor.degree()))
                sparse.append(row)
            if factor.degree() == 1:
                finite = self._prime_ring(prime)
                inputs = tuple(
                    finite.from_dict({m: c[0] for m, c in row.items()}) for row in sparse
                )
                basis, profile = native_prime_basis(inputs, finite, prime, limits=limits)
                leading = tuple(tuple(p.LM) for p in basis)
                rows = tuple({tuple(m): (int(c) % prime,) for m, c in p.items()} for p in basis)
            else:
                try:
                    from . import _native_extension
                except ImportError as error:
                    raise GroebnerBackendUnavailable(
                        "number-field images require the FLINT native extension; build with ALGROOTS_BUILD_NATIVE_EXTENSION=1"
                    ) from error
                try:
                    kernel_started = perf_counter()
                    basis, profile = _native_extension.compute(
                        sparse,
                        prime,
                        tuple(int(factor[i]) for i in range(factor.degree() + 1)),
                        len(self.ring.symbols),
                        str(self.ring.order),
                        max_pairs=limits.max_pairs,
                        max_terms=limits.max_terms,
                        max_basis=limits.max_basis,
                        batch_size=limits.batch_size,
                        max_reduction_steps=limits.max_reduction_steps,
                        max_field_degree=limits.max_field_degree,
                        max_matrix_cells=limits.max_field_matrix_cells,
                        collect_batch_metrics=limits.collect_batch_metrics,
                    )
                except _native_extension.LimitError as error:
                    raise NativeF4LimitError(str(error)) from error
                profile["native_call_seconds"] = perf_counter() - kernel_started
                rows = tuple(basis)
                leading = tuple(max(row, key=self.ring.order) for row in rows)
                stats["extension_calls"] += 1
            stats["native_calls"] += 1
            stats["processed_pairs"] += profile.get(
                "processed_pairs", profile.get("pairs_processed", 0)
            )
            if limits.collect_batch_metrics:
                stats["images"].append(
                    {
                        "prime": prime,
                        "point": point,
                        "factor": tuple(map(int, factor)),
                        "profile": profile,
                    }
                )
            if signature is not None and signature != leading:
                raise _UnluckyImage("modular algebraic components have different leading monomials")
            signature = leading
            images.append((weight, rows))
        output = []
        for index in range(len(signature)):
            support = set().union(*(set(rows[index]) for _, rows in images))
            row = {}
            for monomial in support:
                coefficient = nmod_poly([], prime)
                for weight, rows in images:
                    coefficient += nmod_poly(list(rows[index].get(monomial, ())), prime) * weight
                coefficient %= modulus
                row[monomial] = tuple(int(coefficient[i]) for i in range(self.degree))
            output.append(row)
        return signature, tuple(output)


def _interpolate(samples, monomial, coordinate, prime, count, max_degree, limits):
    """Recover one rational coordinate using a FLINT nullspace and held-out samples."""
    values = [row.get(monomial, (0,) * (coordinate + 1))[coordinate] for _, row in samples]
    return rational_interpolate(
        [point for point, _ in samples],
        values,
        prime,
        count,
        max_degree,
        limits.max_interpolation_terms,
    )


def _sparse_prime_image(model, prime, limits, stats):
    """Share native images across line interpolation and power-basis coordinates."""
    cache = {}
    signature = None

    def sample(point):
        nonlocal signature
        if point in cache:
            return cache[point]
        if len(cache) >= limits.max_evaluation_points:
            raise NativeF4LimitError("sparse parameter evaluation-point budget exceeded")
        stats["evaluation_points"] += 1
        try:
            leading, image = model.sample(prime, point, limits, stats)
        except _UnluckyImage as error:
            stats["rejected_points"] += 1
            raise UnluckyInterpolation(str(error)) from error
        if signature is None:
            signature = leading
        elif signature != leading:
            raise UnluckyInterpolation("sparse line has a different leading-monomial signature")
        cache[point] = {
            (i, m, j): c
            for i, row in enumerate(image)
            for m, coefficients in row.items()
            for j, c in enumerate(coefficients)
        }
        return cache[point]

    started = perf_counter()
    images_before = stats["modular_image_seconds"]
    try:
        result = sparse_interpolate(sample, len(model.parameters), prime, limits, stats)
    finally:
        stats["interpolation_seconds"] += max(
            0.0, perf_counter() - started - (stats["modular_image_seconds"] - images_before)
        )
    output = []
    zero = ({}, {(0,) * len(model.parameters): 1})
    for i in range(len(signature)):
        support = {m for index, m, _ in result if index == i}
        output.append(
            {m: tuple(result.get((i, m, j), zero) for j in range(model.degree)) for m in support}
        )
    return signature, tuple(output)


def _prime_image(model, prime, limits, stats):
    """Group regular evaluations by leading monomials before interpolating coefficients."""
    if not model.parameters:
        signature, image = model.sample(prime, (), limits, stats)
        return signature, tuple(
            {
                m: tuple(({(): c} if c else {}, {(): 1}) for c in coordinates)
                for m, coordinates in row.items()
            }
            for row in image
        )
    if stats["parameter_interpolation"] == "sparse":
        return _sparse_prime_image(model, prime, limits, stats)
    count = len(model.parameters)
    rng = Random(prime)
    groups = {}
    seen = set()
    for degree in range(limits.max_parameter_degree + 1):
        terms = comb(degree + count, count)
        if 2 * terms > limits.max_interpolation_terms:
            raise NativeF4LimitError("rational interpolation term budget exceeded")
        needed = 2 * terms + 1
        if needed > limits.max_evaluation_points:
            raise NativeF4LimitError(
                "parameter evaluation-point budget is too small for interpolation"
            )
        if needed > prime**count:
            raise _UnluckyImage("finite field is too small for distinct interpolation points")
        while max((len(group) for group in groups.values()), default=0) < needed:
            if len(seen) >= limits.max_evaluation_points:
                raise _UnluckyImage(
                    "no stable regular images within this prime's evaluation budget"
                )
            if len(seen) >= prime**count:
                raise _UnluckyImage("not enough regular evaluation points in this finite field")
            point = tuple(rng.randrange(prime) for _ in range(count))
            if point in seen:
                continue
            seen.add(point)
            stats["evaluation_points"] += 1
            try:
                signature, image = model.sample(prime, point, limits, stats)
            except _UnluckyImage:
                stats["rejected_points"] += 1
                continue
            groups.setdefault(signature, []).append((point, image))
        for signature, samples in sorted(
            groups.items(), key=lambda item: len(item[1]), reverse=True
        ):
            if len(samples) < needed:
                continue
            output = []
            for index in range(len(signature)):
                row_samples = [(point, image[index]) for point, image in samples]
                support = set().union(*(set(row) for _, row in row_samples))
                row = {}
                for monomial in sorted(support):
                    started = perf_counter()
                    coefficients = tuple(
                        _interpolate(
                            row_samples, monomial, coordinate, prime, count, degree, limits
                        )
                        for coordinate in range(model.degree)
                    )
                    stats["interpolation_seconds"] += perf_counter() - started
                    if any(coefficient is None for coefficient in coefficients):
                        break
                    row[monomial] = coefficients
                else:
                    output.append(row)
                    continue
                break
            else:
                return signature, tuple(output)
    raise NativeF4LimitError("parameter reconstruction degree budget exceeded")


def _flatten(signature, image):
    result = {}
    for leading, row in zip(signature, image, strict=True):
        for monomial, coefficients in row.items():
            for coordinate, parts in enumerate(coefficients):
                for part, terms in enumerate(parts):
                    for exponent, value in terms.items():
                        result[(leading, monomial, coordinate, part, exponent)] = value
    return result


def _reconstruct(signature, residues, modulus, model):
    """Lift rational parameter functions in the original algebraic power basis."""
    coefficients = {}
    for (leading, monomial, coordinate, part, exponent), residue in residues.items():
        value = rational_reconstruction(residue, modulus)
        if value is None:
            return None
        coefficients.setdefault((leading, monomial, coordinate, part), {})[exponent] = value
    result = []
    for leading in signature:
        support = {key[1] for key in coefficients if key[0] == leading}
        row = {}
        for monomial in support:
            value = sp.S.Zero
            for coordinate in range(model.degree):
                parts = []
                for part in (0, 1):
                    terms = coefficients.get((leading, monomial, coordinate, part), {})
                    parts.append(
                        sum(
                            (
                                sp.Rational(c)
                                * sp.prod(
                                    t**e for t, e in zip(model.parameters, exponent, strict=True)
                                )
                                for exponent, c in terms.items()
                            ),
                            sp.S.Zero,
                        )
                    )
                if not parts[1]:
                    return None
                value += parts[0] / parts[1] * model.alpha**coordinate
            row[monomial] = field_element(value, model.ring.domain)
        result.append(model.ring.from_dict(row))
    return tuple(result)


def _fresh_check(candidate, model, prime, limits, stats):
    """Compare ideals at a fresh prime; exact verification remains a separate step."""
    candidate_model = _FieldModel(candidate, model.ring)
    rng = Random(prime + 17)
    successful = 0
    attempts = 3 if not model.parameters else limits.max_evaluation_points
    for _ in range(attempts):
        point = tuple(rng.randrange(prime) for _ in model.parameters)
        stats["validation_points"] += 1
        if model.parameters:
            stats["evaluation_points"] += 1
        try:
            expected = model.sample(prime, point, limits, stats)
            actual = candidate_model.sample(prime, point, limits, stats)
        except _UnluckyImage:
            stats["rejected_validation_points"] += 1
            continue
        if actual != expected:
            return False
        successful += 1
        if successful == (3 if model.parameters else 1):
            return True
    return False


def native_field_basis(polynomials, ring, *, limits=None, certify=False):
    """Reconstruct a native candidate over number, rational-function, or algebraic-function fields.

    FLINT solves interpolation systems. The configured minimum number of modular images
    (two by default) and fresh-prime native calculations must agree. ``certify=True`` adds
    independent exact reduced-basis comparison; public ``groebner_basis``
    performs that comparison itself even when this expert function omits it.
    """
    started = perf_counter()
    limits = NativeF4Limits() if limits is None else limits
    if not isinstance(limits, NativeF4Limits):
        raise TypeError("limits must be NativeF4Limits")
    polynomials = tuple(polynomials)
    if isinstance(ring.domain, AlgebraicFunctionField):
        base = ring.domain.domain
        ground = base.domain if base.is_FractionField else base
        degree = ring.domain.rank * (ground.mod.degree() if ground.is_AlgebraicField else 1)
        if degree > limits.max_field_degree:
            raise NativeF4LimitError("algebraic tower degree budget exceeded")
    model = _FieldModel(polynomials, ring)
    if model.degree > limits.max_field_degree:
        raise NativeF4LimitError("constant-field degree budget exceeded")
    interpolation = limits.parameter_interpolation
    terms = comb(len(model.parameters) + limits.max_parameter_degree, len(model.parameters))
    if interpolation == "auto":
        interpolation = (
            "sparse"
            if len(model.parameters) > 1
            and (
                2 * terms > limits.max_interpolation_terms
                or 2 * terms + 1 > limits.max_evaluation_points
            )
            else "dense"
        )
    stats = {
        "kind": "field_images",
        "coefficient_domain": str(ring.domain),
        "parameter_interpolation": interpolation,
        "sparse_lines": 0,
        "sparse_terms": 0,
        "rejected_validation_points": 0,
        "native_calls": 0,
        "extension_calls": 0,
        "evaluation_points": 0,
        "rejected_points": 0,
        "validation_points": 0,
        "processed_pairs": 0,
        "images": [],
        "constant_field_degree": model.degree,
        "parameters": tuple(map(str, model.parameters)),
        "prepare_seconds": perf_counter() - started,
        "modular_image_seconds": 0.0,
        "interpolation_seconds": 0.0,
        "reconstruction_seconds": 0.0,
        "validation_seconds": 0.0,
        "exact_verification_seconds": 0.0,
        "rejected_primes": [],
    }
    signature = None
    residues = {}
    modulus = 1
    accepted = []
    prime = limits.first_prime
    if model.parameters and interpolation == "sparse":
        if terms > limits.max_sparse_monomials:
            raise NativeF4LimitError("sparse exponent decoding budget exceeded")
        target = min(16 * terms * limits.max_interpolation_terms, _MAX_NATIVE_PRIME // 2)
        prime = int(sp.nextprime(max(prime - 1, target)))
    stats["first_image_prime"] = prime
    for _ in range(limits.max_primes):
        if prime > _MAX_NATIVE_PRIME:
            raise NativeF4LimitError("no supported reconstruction primes remain")
        try:
            leading, image = _prime_image(model, prime, limits, stats)
        except (_UnluckyImage, UnluckyInterpolation) as error:
            stats["rejected_primes"].append({"prime": prime, "reason": str(error)})
            prime = int(sp.nextprime(prime))
            continue
        flat = _flatten(leading, image)
        key = leading, tuple(sorted(flat))
        if signature != key:
            signature, residues, modulus, accepted = key, flat, prime, [prime]
        else:
            inverse = pow(modulus % prime, -1, prime)
            residues = {
                key: value + modulus * ((flat[key] - value) * inverse % prime)
                for key, value in residues.items()
            }
            modulus *= prime
            accepted.append(prime)
        if modulus.bit_length() > limits.max_modulus_bits:
            raise NativeF4LimitError("coefficient reconstruction modulus budget exceeded")
        reconstruction_started = perf_counter()
        candidate = (
            _reconstruct(leading, residues, modulus, model)
            if len(accepted) >= limits.min_primes
            else None
        )
        stats["reconstruction_seconds"] += perf_counter() - reconstruction_started
        if candidate is not None:
            validation = int(sp.nextprime(prime))
            validation_started = perf_counter()
            valid = validation <= _MAX_NATIVE_PRIME and _fresh_check(
                candidate, model, validation, limits, stats
            )
            stats["validation_seconds"] += perf_counter() - validation_started
            if valid:
                if certify:
                    verification_started = perf_counter()
                    reference = tuple(reference_groebner(list(polynomials), ring))
                    if len(candidate) != len(reference) or {
                        frozenset(p.items()) for p in candidate
                    } != {frozenset(p.items()) for p in reference}:
                        raise GroebnerVerificationError(
                            "native field reconstruction failed exact verification"
                        )
                    stats["exact_verification_seconds"] += perf_counter() - verification_started
                stats.update(
                    seconds=perf_counter() - started,
                    validation_prime=validation,
                    modulus_bits=modulus.bit_length(),
                )
                record_native_stats(stats)
                return NativeFieldResult(candidate, tuple(accepted), stats, certify)
        prime = int(sp.nextprime(prime))
    raise NativeF4LimitError("no stable coefficient reconstruction within the prime budget")
