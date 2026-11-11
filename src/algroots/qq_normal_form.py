"""Exact multivariate QQ normal forms backed by compiled FLINT arithmetic.

The underlying fmpq_mpoly quotient/remainder operations execute in FLINT's C
library through python-flint. Python coordinates repeated single-divisor
passes because the bindings do not expose FLINT's ideal-division routine.

For a Groebner basis, a complete remainder is unique.  The input divisors need
not be a Groebner basis: the result is still a valid fully reduced remainder,
and every subtraction is exact.  This module never accepts a basis as valid
without the independent Buchberger/ideal-equality checks elsewhere.
"""

from functools import lru_cache

from .groebner_errors import GroebnerLimitError


class QQNormalFormLimitError(GroebnerLimitError):
    """The explicitly bounded exact normal-form computation was incomplete."""


_ORDERS = {"lex": "lex", "grlex": "deglex", "grevlex": "degrevlex"}


@lru_cache(maxsize=64)
def _context(names, order):
    from flint import fmpq_mpoly_ctx

    return fmpq_mpoly_ctx.get(names, _ORDERS[order])


class FLINTQQNormalForm:
    """Reusable compiled-arithmetic quotient/remainder evaluator over QQ.

    A fixed divisor list is converted just once.  The implementation does
    not compute or certify a Groebner basis; it performs only exact division.
    """

    def __init__(self, divisors, ring, *, max_passes=10000):
        from flint import fmpq

        order = str(ring.order)
        if order not in _ORDERS:
            raise ValueError("FLINT QQ normal forms support lex, grlex, grevlex")
        if not getattr(ring.domain, "is_QQ", False):
            raise ValueError("FLINT QQ normal forms require the rational field")
        if not isinstance(max_passes, int) or max_passes < 1:
            raise ValueError("max_passes must be a positive integer")
        self.ring = ring
        self.ctx = _context(tuple(str(s) for s in ring.symbols), order)
        self._fmpq = fmpq
        self.divisors = tuple(self.convert(poly) for poly in divisors)
        if any(not d for d in self.divisors):
            raise ZeroDivisionError("zero polynomial among normal-form divisors")
        self.max_passes = max_passes
        self.passes = 0

    def convert(self, polynomial):
        # str(c) is exact for both SymPy QQ representations (PythonMPQ and
        # FLINT-backed fmpq). Never convert through a floating-point number.
        """Convert one sparse QQ polynomial without passing coefficients through floating point."""
        return self.ctx.from_dict(
            {tuple(exponents): self._fmpq(str(coeff)) for exponents, coeff in polynomial.terms()}
        )

    def remainder(self, polynomial, *, converted=False):
        """Compute a fully reduced exact remainder within the single-divisor pass budget."""
        current = polynomial if converted else self.convert(polynomial)
        if not current or not self.divisors:
            return current
        for _ in range(self.max_passes):
            changed = False
            for divisor in self.divisors:
                quotient, result = divmod(current, divisor)
                if quotient:
                    changed = True
                    current = result
                    if not current:
                        self.passes += 1
                        return current
            self.passes += 1
            if not changed:
                return current
        raise QQNormalFormLimitError("FLINT QQ normal-form pass budget exhausted")

    def verify_rational_dag(self, dag, original, candidate, *, max_terms=100000):
        """Replay an untrusted rational proof circuit using exact FLINT QQ.

        The DAG structure is validated independently of its modular producer.
        Every node is evaluated to a rational polynomial, not to a vector of
        expanded generator-combination witness polynomials.
        """
        from fractions import Fraction

        from .compact_rational_proof import RationalProofDAG

        try:
            nonzero = tuple(f for f in original if f)
            if (
                not isinstance(dag, RationalProofDAG)
                or dag.generator_count != len(nonzero)
                or len(dag.scales) != len(nonzero)
                or len(dag.roots) != len(candidate)
                or len(dag.nodes) > max_terms
                or max_terms < 1
            ):
                return False
            originals = tuple(
                self.convert(poly) * self._fmpq(str(scale))
                for poly, scale in zip(nonzero, dag.scales, strict=True)
            )
            values = []
            nvars = len(self.ring.symbols)
            for index, edges in enumerate(dag.nodes):
                value = self.ctx.from_dict({})
                for child, terms in edges:
                    if (
                        type(child) is not int
                        or child >= index
                        or child < -len(originals)
                        or not hasattr(terms, "items")
                    ):
                        return False
                    coeffs = {}
                    for powers, coefficient in terms.items():
                        if (
                            not isinstance(powers, tuple)
                            or len(powers) != nvars
                            or any(type(e) is not int or e < 0 for e in powers)
                        ):
                            return False
                        number = Fraction(str(coefficient))
                        if number:
                            coeffs[powers] = self._fmpq(str(number))
                    if coeffs:
                        value += self.ctx.from_dict(coeffs) * (
                            originals[-child - 1] if child < 0 else values[child]
                        )
                        if len(value) > max_terms:
                            return False
                values.append(value)
            if any(type(root) is not int or root < 0 or root >= len(values) for root in dag.roots):
                return False
            return all(
                values[root] == self.convert(poly)
                for root, poly in zip(dag.roots, candidate, strict=True)
            )
        except (
            TypeError,
            ValueError,
            IndexError,
            KeyError,
            ArithmeticError,
            OverflowError,
            ZeroDivisionError,
        ):
            return False

    def verify_expanded_memberships(self, generators, basis, witnesses):
        """Replay generator-combination identities using compiled QQ arithmetic.

        Converts the original generators once and the witness coefficients
        only when used. Equality is exact; this is an independent identity
        checker, not a basis-generation algorithm.
        """
        if len(witnesses) != len(basis):
            return False
        originals = tuple(self.convert(poly) for poly in generators)
        for polynomial, row in zip(basis, witnesses, strict=True):
            if len(row) != len(originals):
                return False
            total = self.ctx.from_dict({})
            for coefficient, generator in zip(row, originals, strict=True):
                if coefficient and generator:
                    total += self.convert(coefficient) * generator
            if total != self.convert(polynomial):
                return False
        return True

    def verify_sparse_memberships(self, generators, basis, rows):
        """Verify exact sparse QQ witness rows without constructing SymPy polys."""
        if len(rows) != len(basis):
            return False
        originals = tuple(self.convert(poly) for poly in generators)
        for polynomial, row in zip(basis, rows, strict=True):
            if len(row) != len(originals):
                return False
            total = self.ctx.from_dict({})
            for coefficient, generator in zip(row, originals, strict=True):
                if not hasattr(coefficient, "items"):
                    return False
                # Validate sparse monomial coordinates before asking FLINT
                # to parse them. Malformed certificates must be rejected, not
                # accepted via coercion or interpreted as Laurent monomials.
                entries = {}
                for powers, value in coefficient.items():
                    if (
                        not isinstance(powers, tuple)
                        or len(powers) != len(self.ring.symbols)
                        or any(
                            not isinstance(e, int) or isinstance(e, bool) or e < 0 for e in powers
                        )
                    ):
                        return False
                    entries[powers] = self._fmpq(str(value))
                if entries and generator:
                    total += self.ctx.from_dict(entries) * generator
            if total != self.convert(polynomial):
                return False
        return True

    def s_pair_remainder(self, first, second):
        """Reduce an S-polynomial using compiled exact QQ arithmetic."""
        left = self.divisors[first]
        right = self.divisors[second]
        left_lm = left.monoms()[0]
        right_lm = right.monoms()[0]
        lcm = tuple(max(a, b) for a, b in zip(left_lm, right_lm, strict=True))
        shift_left = self.ctx.term(
            right.leading_coefficient(), tuple(a - b for a, b in zip(lcm, left_lm, strict=True))
        )
        shift_right = self.ctx.term(
            left.leading_coefficient(), tuple(a - b for a, b in zip(lcm, right_lm, strict=True))
        )
        # Cross-multiplication also makes this valid for nonmonic divisors.
        s_poly = shift_left * left - shift_right * right
        return self.remainder(s_poly, converted=True)


__all__ = ["FLINTQQNormalForm", "QQNormalFormLimitError"]
