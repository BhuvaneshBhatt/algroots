"""Multiplicity views and canonical ordering of distinct solution records."""

from dataclasses import dataclass, fields, replace
from functools import cmp_to_key, lru_cache
from itertools import repeat
from time import perf_counter

import sympy as sp

from ._validation import validate_integer
from .errors import (
    PolynomialSystemInputError,
    RootMultiplicityError,
    RootOrderingError,
    SystemSolveLimitError,
)


@dataclass(frozen=True)
class RootOrderingEvidence:
    """Whether the lexicographic coordinate order was established exactly."""

    status: str = "unresolved"
    basis: str = "none"
    notes: tuple[str, ...] = ()


class RootViews:
    """Keep aligned metadata on distinct points; expand only the public view."""

    @property
    def distinct_roots(self):
        return self.roots

    def iter_roots(self, *, with_multiplicity=None):
        expanded = (
            self.root_mode == "with_multiplicity"
            if with_multiplicity is None
            else with_multiplicity
        )
        if not isinstance(expanded, bool):
            raise ValueError("with_multiplicity must be Boolean")
        if not expanded:
            return iter(self.roots)
        if self.multiplicities is None or any(m is None for m in self.multiplicities):
            raise RootMultiplicityError(
                "repeated-root output requires established per-root multiplicities"
            )
        if sum(self.multiplicities) > self.max_returned_roots:
            raise SystemSolveLimitError("expanded root count exceeds max_returned_roots")
        return (
            root
            for root, multiplicity in zip(self.roots, self.multiplicities, strict=True)
            for root in repeat(root, multiplicity)
        )

    @property
    def output_roots(self):
        return tuple(self.iter_roots())

    @property
    def roots_with_multiplicity(self):
        return tuple(self.iter_roots(with_multiplicity=True))

    def __iter__(self):
        return self.iter_roots()

    def __len__(self):
        if self.root_mode == "distinct":
            return len(self.roots)
        self.iter_roots(with_multiplicity=True)  # Validate without allocating repeated tuples.
        return sum(self.multiplicities)


def validate_output_options(
    root_mode, multiplicity, root_order, max_returned_roots, ordering_max_refinements
):
    if root_mode not in ("distinct", "with_multiplicity"):
        raise PolynomialSystemInputError("root_mode must be distinct or with_multiplicity")
    if not (
        multiplicity is False
        or isinstance(multiplicity, str)
        and multiplicity in ("auto", "required")
    ):
        raise PolynomialSystemInputError("multiplicity must be False, auto or required")
    if root_order not in ("canonical", "required"):
        raise PolynomialSystemInputError("root_order must be canonical or required")
    validate_integer(max_returned_roots, "max_returned_roots")
    validate_integer(ordering_max_refinements, "ordering_max_refinements")


# Every field listed here has one entry for each distinct point. Global path
# histories and quotient monomials deliberately do not participate.
ALIGNED_FIELDS = (
    "roots",
    "diagnostics",
    "recognized_roots",
    "root_certifications",
    "root_deflations",
    "multiplicities",
    "multiplicity_evidence",
    "multiplicity_certifications",
    "homotopy_endpoint_smallest_singular_values",
    "homotopy_endpoint_condition_estimates",
)


def permute_root_records(result, permutation):
    count = len(result.roots)
    if sorted(permutation) != list(range(count)):
        raise ValueError("root permutation must contain each distinct index exactly once")
    available = {field.name for field in fields(result)}
    updates = {}
    for name in ALIGNED_FIELDS:
        if name not in available:
            continue
        values = getattr(result, name)
        if values is None or values == () and name in ("diagnostics", "multiplicity_evidence"):
            continue
        if len(values) != count:
            raise ValueError(f"{name} must align with distinct roots")
        updates[name] = (
            values if permutation == list(range(count)) else tuple(values[i] for i in permutation)
        )
    return replace(result, **updates)


@lru_cache(maxsize=256)
def _exact_sign(expression, max_refinements):
    expression = sp.simplify(expression)
    if expression == 0:
        return 0
    if expression.is_Rational:
        return 1 if expression > 0 else -1
    # Map an exact algebraic difference to its rational-polynomial root.
    # same_root uses a separation bound and bounded-error evaluation, not a
    # floating-point equality guess. Interval sign is then established exactly.
    if any(term.has(sp.CRootOf) for term in expression.atoms(sp.re, sp.im)):
        raise RootOrderingError("unresolved algebraic coordinate projection")
    degree_bound = 1
    for root in expression.atoms(sp.CRootOf):
        degree_bound *= root.poly.degree()
    if degree_bound > 64:
        raise RootOrderingError("exact ordering exceeds composite algebraic degree budget 64")
    parameter = sp.Dummy("ordering_t")
    polynomial = sp.Poly(sp.minpoly(expression, parameter), parameter)
    if polynomial.degree() == 1:
        root = -polynomial.nth(0) / polynomial.nth(1)
        return 1 if root > 0 else -1 if root < 0 else 0
    if polynomial.degree() > 64:
        raise RootOrderingError("exact ordering exceeds algebraic degree budget 64")
    for root in polynomial.all_roots(radicals=False):
        if root.is_real and polynomial.same_root(expression, root):
            if root.is_Rational:
                return 1 if root > 0 else -1 if root < 0 else 0
            interval = root._get_interval()
            for _ in range(max_refinements):
                if interval.a >= 0:
                    return 1
                if interval.b <= 0:
                    return -1
                interval = interval.refine()
            break
    raise RootOrderingError("exact coordinate order unresolved within refinement budget")


def _coordinate_key(point, digits):
    return tuple(
        component
        for value in point
        for component in (sp.re(sp.N(value, digits)), sp.im(sp.N(value, digits)))
    )


def _exact_points(result):
    points = []
    for i in range(len(result.roots)):
        point = None
        for attempts in (result.root_certifications, result.multiplicity_certifications):
            if attempts is not None and attempts[i].status == "certified":
                point = attempts[i].certificate.point
                break
        if point is None and result.recognized_roots is not None:
            point = (
                result.recognized_roots[i].exact_coordinates
                if result.recognized_roots[i].jointly_certified
                else None
            )
        if point is None:
            return None
        points.append(point)
    return points


def finish_root_output(
    result,
    *,
    root_mode="distinct",
    multiplicity="auto",
    root_order="canonical",
    max_returned_roots=10000,
    ordering_max_refinements=64,
    certification_max_dimension=128,
    certification_max_refinements=128,
    certification_max_box_attempts=16,
    polynomial=True,
):
    output_started = perf_counter()
    validate_output_options(
        root_mode, multiplicity, root_order, max_returned_roots, ordering_max_refinements
    )
    count = len(result.roots)
    attempts = (
        result.root_certifications
        if result.root_certifications is not None
        else result.multiplicity_certifications
    )
    needs_multiplicity = (
        polynomial
        and multiplicity is not False
        and (
            result.is_radical is False
            or multiplicity == "required"
            or root_mode == "with_multiplicity"
        )
    )
    if (
        polynomial
        and count
        and (needs_multiplicity or root_order == "required" and _exact_points(result) is None)
        and attempts is None
    ):
        from .certification import certify_numerical_roots

        attempts = certify_numerical_roots(
            result,
            max_quotient_dimension=certification_max_dimension,
            max_refinements=certification_max_refinements,
            max_box_attempts=certification_max_box_attempts,
        )
    multiplicities, evidence = [], []
    for i in range(count):
        if polynomial and attempts is not None and attempts[i].status == "certified":
            multiplicities.append(attempts[i].certificate.multiplicity)
            evidence.append("exact_local_quotient_certificate")
        elif polynomial and result.is_radical is True:
            multiplicities.append(1)
            evidence.append("exact_radical_quotient")
        else:
            multiplicities.append(None)
            evidence.append("unknown" if polynomial else "projected_cover_not_transferred")
    if (
        polynomial
        and all(m is not None for m in multiplicities)
        and result.total_multiplicity is not None
        and sum(multiplicities) != result.total_multiplicity
    ):
        raise RootMultiplicityError("local multiplicities disagree with the exact total")
    result = replace(
        result,
        multiplicities=tuple(multiplicities),
        multiplicity_evidence=tuple(evidence),
        multiplicity_certifications=attempts,
        root_mode=root_mode,
        max_returned_roots=max_returned_roots,
    )
    if multiplicity == "required" and any(m is None for m in multiplicities):
        raise RootMultiplicityError("per-root multiplicities could not be established")
    if root_mode == "with_multiplicity":
        result.iter_roots()  # Fail before returning an unusable expanded view.
    points = _exact_points(result) if count >= 2 else None
    if count < 2:
        ordering = RootOrderingEvidence("certified", "trivial_cardinality")
        permutation = list(range(count))
    elif points is not None:
        try:
            keys = [
                tuple(c for value in point for c in (sp.re(value), sp.im(value)))
                for point in points
            ]

            boxes = []
            for i in range(count):
                box = None
                for proofs in (result.root_certifications, result.multiplicity_certifications):
                    if proofs is not None and proofs[i].status == "certified":
                        box = proofs[i].certificate.box
                        if box is not None:
                            break
                boxes.append(box)

            def compare(i, j):
                for component, (a, b) in enumerate(zip(keys[i], keys[j], strict=True)):
                    if a == b:
                        continue
                    coordinate, part = divmod(component, 2)
                    if (
                        part == 0
                        and sp.expand(sp.conjugate(points[i][coordinate])) == points[j][coordinate]
                    ):
                        continue  # Conjugate coordinates have exactly equal real parts.
                    if boxes[i] is not None and boxes[j] is not None:
                        offset = 0 if part == 0 else 2
                        left = boxes[i].bounds[coordinate][offset : offset + 2]
                        right = boxes[j].bounds[coordinate][offset : offset + 2]
                        if left[1] <= right[0]:
                            return -1
                        if right[1] <= left[0]:
                            return 1
                    sign = _exact_sign(a - b, ordering_max_refinements)
                    if sign:
                        return sign
                return 0

            permutation = sorted(range(count), key=cmp_to_key(compare))
            ordering = RootOrderingEvidence("certified", "exact_algebraic_coordinate_comparison")
        except (
            RootOrderingError,
            NotImplementedError,
            ValueError,
            TypeError,
            sp.polys.polyerrors.BasePolynomialError,
        ) as exc:
            if root_order == "required":
                raise RootOrderingError(f"required canonical order unavailable: {exc}") from exc
            points = None
    if count >= 2 and points is None:
        if root_order == "required":
            raise RootOrderingError("required canonical order needs established exact points")
        keys = [_coordinate_key(point, result.precision_digits) for point in result.roots]
        permutation = sorted(range(count), key=lambda i: keys[i])
        ordering = RootOrderingEvidence(
            "numerical",
            "arbitrary_precision_coordinate_lexicographic",
            (
                "Precision-independent ordering is not established; near-ties may change across backends or precisions.",
            ),
        )
    result = replace(permute_root_records(result, permutation), ordering=ordering)
    if result.cost_diagnostics is not None:
        elapsed = perf_counter() - output_started
        phases = list(result.cost_diagnostics.phase_seconds)
        for index in range(len(phases) - 1, -1, -1):
            if phases[index][0] == "total":
                phases[index] = ("total", phases[index][1] + elapsed)
                break
        phases.append(("root_output", elapsed))
        result = replace(
            result, cost_diagnostics=replace(result.cost_diagnostics, phase_seconds=tuple(phases))
        )
    return result


__all__ = ["RootOrderingEvidence"]
