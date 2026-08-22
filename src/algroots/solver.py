"""Numerical roots of exact zero-dimensional polynomial systems."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import mpmath as mp
import sympy as sp

from ._validation import (
    normalize_equation,
    validate_recognition_options,
    validate_variables,
)
from .continuation import PathResult, PathTrackerOptions
from .errors import (
    ActionMatrixError,
    HomotopySolveError,
    NotZeroDimensionalError,
    NumericalRootError,
    PolynomialSystemError,
    PolynomialSystemInputError,
    ShapePositionError,
    SystemSolveLimitError,
    TriangularSolveError,
)
from .numerical import (
    acb_mid_to_sympy,
    compile_polynomials,
    compiled_jacobian,
    compiled_residual,
    flint_available,
    flint_dps,
    newton_refine,
    sympy_to_acb,
    sympy_to_mpc,
)
from .total_degree_homotopy import (
    TotalDegreeHomotopy,
    TotalDegreeHomotopyOptions,
    build_total_degree_homotopy,
    gamma_candidates,
    track_total_degree_paths,
)

if TYPE_CHECKING:
    from .recognition import RecognizedSystemRoot


SolveMethod = Literal["auto", "shape", "action", "triangular", "rur", "homotopy"]


@dataclass(frozen=True)
class RootDiagnostics:
    """Numerical validation data for one returned solution tuple."""

    initial_relative_residual: Any
    final_relative_residual: Any
    refinement_attempted: bool
    refinement_succeeded: bool
    refinement_improved: bool


@dataclass(frozen=True)
class _EndpointRegularity:
    smallest_singular_value: mp.mpf
    condition_estimate: mp.mpf
    digits: int
    resolved: bool


@dataclass(frozen=True)
class _CompiledPolynomialSystem:
    expressions: tuple[Any, ...]
    variables: tuple[Any, ...]
    polynomials: tuple[Any, ...]
    jacobian: tuple[tuple[Any, ...], ...]

    @classmethod
    def build(
        cls, expressions: tuple[Any, ...], variables: tuple[Any, ...]
    ) -> _CompiledPolynomialSystem:
        return cls(
            expressions,
            variables,
            compile_polynomials(expressions, variables),
            compiled_jacobian(expressions, variables),
        )

    def residual(self, values: tuple[Any, ...], digits: int) -> mp.mpf:
        return compiled_residual(self.polynomials, values, digits)

    def endpoint_regularity(
        self, root: tuple[Any, ...], *, digits: int, verification_digits: int
    ) -> _EndpointRegularity:
        with mp.workdps(digits + 10):
            values = _mp_root(root, digits + 5)
            variable_scales = [max(mp.mpf(1), abs(value)) for value in values]
            rows: list[list[mp.mpc]] = []
            for row_index, polynomial in enumerate(self.polynomials):
                equation_scale = mp.mpf(0)
                for coefficient, powers in polynomial.terms:
                    term = abs(sympy_to_mpc(coefficient, digits + 5))
                    for scale, power in zip(variable_scales, powers, strict=True):
                        if power:
                            term *= scale**power
                    equation_scale += term
                if equation_scale == 0:
                    return _EndpointRegularity(mp.mpf(0), mp.inf, digits, False)
                row: list[mp.mpc] = []
                for column, derivative in enumerate(self.jacobian[row_index]):
                    derivative_value = derivative.evaluate_mp(values, digits + 5)[0]
                    row.append(derivative_value * variable_scales[column] / equation_scale)
                rows.append(row)
            matrix = mp.matrix(rows)
            try:
                singular_values = [abs(value) for value in mp.svd(matrix, compute_uv=False)]
            except (ValueError, ZeroDivisionError):
                return _EndpointRegularity(mp.mpf(0), mp.inf, digits, False)
            if not singular_values:
                return _EndpointRegularity(mp.mpf(0), mp.inf, digits, False)
            smallest = min(singular_values)
            largest = max(singular_values)
            condition = mp.inf if smallest == 0 else largest / smallest
            resolution_threshold = mp.power(10, -max(8, digits - 8))
            return _EndpointRegularity(
                smallest, condition, digits, bool(smallest > resolution_threshold)
            )


@dataclass(frozen=True)
class PolynomialSystemRoots:
    """All distinct verified numerical roots found for a polynomial system.

    Coordinates are SymPy arbitrary-precision numbers in the same order as
    ``variables``. Every returned root has been checked against the original
    equations after a higher-precision refinement attempt. ``diagnostics`` is
    aligned with ``roots`` and records the residual before and after refinement.
    """

    roots: tuple[tuple[Any, ...], ...]
    variables: tuple[Any, ...]
    equations: tuple[Any, ...]
    groebner_basis: tuple[Any, ...]
    precision_digits: int
    working_digits: int
    verification_digits: int
    max_relative_residual: Any
    method: str
    diagnostics: tuple[RootDiagnostics, ...] = ()
    eliminant: Any | None = None
    parameter_variable: Any | None = None
    quotient_dimension: int | None = None
    standard_monomials: tuple[tuple[int, ...], ...] | None = None
    separator_coeffs: tuple[int, ...] | None = None
    recognized_roots: tuple[RecognizedSystemRoot, ...] | None = None
    rational_univariate_representation: Any | None = None
    homotopy_paths_total: int | None = None
    homotopy_paths_succeeded: int | None = None
    homotopy_paths_failed: int | None = None
    homotopy_paths_divergent: int | None = None
    homotopy_gamma: Any | None = None
    homotopy_gamma_attempts: int | None = None
    homotopy_endpoint_smallest_singular_values: tuple[Any, ...] | None = None
    homotopy_endpoint_condition_estimates: tuple[Any, ...] | None = None
    recognition_attempted: bool = False
    recognition_error: str | None = None

    def __post_init__(self) -> None:
        if self.recognized_roots is not None and len(self.recognized_roots) != len(self.roots):
            raise ValueError("recognized_roots must align one-to-one with roots")

    def __len__(self) -> int:
        return len(self.roots)

    def __iter__(self):
        return iter(self.roots)


@dataclass(frozen=True)
class _ShapeRepresentation:
    parameter: Any
    eliminant: Any
    coordinate_exprs: tuple[Any, ...]


def _normalize_equations(
    equations: Iterable[Any],
    variables: tuple[Any, ...],
) -> tuple[Any, ...]:
    normalized: list[Any] = []
    for equation in equations:
        expression = normalize_equation(equation, expand=True)
        try:
            polynomial = sp.Poly(expression, *variables, extension=True)
        except (sp.PolynomialError, TypeError, ValueError) as exc:
            raise PolynomialSystemInputError(
                f"equation is not polynomial in the supplied variables: {equation!r}"
            ) from exc

        for coefficient in polynomial.coeffs():
            if coefficient.is_algebraic is not True:
                raise PolynomialSystemInputError(
                    "polynomial coefficients must be exact algebraic numbers; "
                    f"unsupported coefficient: {coefficient!r}"
                )
        normalized.append(polynomial.as_expr())

    if not normalized:
        raise PolynomialSystemInputError("at least one equation is required")
    return tuple(normalized)


def _groebner_basis(equations: tuple[Any, ...], variables: tuple[Any, ...]):
    try:
        return sp.groebner(
            equations,
            *variables,
            order="lex",
            extension=True,
        )
    except (sp.PolynomialError, TypeError, ValueError) as exc:
        raise PolynomialSystemInputError(
            "could not construct an exact Gröbner basis for the system"
        ) from exc


def _is_unit_ideal(basis) -> bool:
    return len(basis.polys) == 1 and basis.polys[0].as_expr() == 1


def _relative_residual(
    expression: Any,
    variables: tuple[Any, ...],
    assignment: dict[Any, Any],
    digits: int,
):
    polynomial = sp.Poly(expression, *variables, extension=True)
    value = abs(sp.N(expression.subs(assignment), digits))

    term_scale = sp.Float(0, digits)
    for powers, coefficient in polynomial.terms():
        term = sp.N(coefficient, digits)
        for variable, power in zip(variables, powers, strict=True):
            if power:
                term *= assignment[variable] ** power
        term_scale += abs(term)

    scale = max(sp.Float(1, digits), sp.N(term_scale, digits))
    return sp.N(value / scale, digits)


def _assignment_satisfies(
    expressions: Iterable[Any],
    variables: tuple[Any, ...],
    assignment: dict[Any, Any],
    tolerance: Any,
    digits: int,
) -> bool:
    assigned = set(assignment)
    for expression in expressions:
        if expression.free_symbols <= assigned:
            residual = _relative_residual(
                expression,
                variables,
                assignment,
                digits,
            )
            if residual > tolerance:
                return False
    return True


def _nroots(
    expression: Any,
    variable: Any,
    digits: int,
    maxsteps: int,
    max_precision_digits: int | None = None,
) -> list[Any]:
    """Find all distinct roots of a univariate polynomial.

    python-flint/Arb is the normal backend. SymPy is retained only as a
    development fallback for environments where python-flint is unavailable.
    """
    try:
        polynomial = sp.Poly(expression, variable, extension=True).sqf_part()
    except (sp.PolynomialError, TypeError, ValueError) as exc:
        raise NumericalRootError(
            f"could not construct a univariate polynomial in {variable}"
        ) from exc
    if polynomial.degree() <= 0:
        raise NumericalRootError(f"polynomial in {variable} has no roots to compute")

    if not flint_available():  # pragma: no cover - development fallback
        try:
            roots = sp.nroots(polynomial, n=digits, maxsteps=maxsteps)
        except (ValueError, ArithmeticError, TypeError) as exc:
            raise NumericalRootError(
                f"numerical root computation failed for variable {variable}"
            ) from exc
        return [sp.N(root, digits) for root in roots]

    from flint import acb_poly, ctx

    max_digits = max_precision_digits or max(digits * 4, digits + 100)
    current = max(30, digits + 8)
    coeffs = list(reversed(polynomial.all_coeffs()))
    while current <= max_digits:
        with flint_dps(current):
            arb_poly = acb_poly([sympy_to_acb(coeff, current) for coeff in coeffs])
            try:
                roots = arb_poly.roots(maxprec=max(ctx.prec * 4, 256))
            except ValueError:
                roots = None
            if roots is not None and len(roots) == polynomial.degree():
                return [acb_mid_to_sympy(root, digits) for root in roots]
        current *= 2
    raise NumericalRootError(
        f"Arb could not isolate all roots for variable {variable} up to {max_digits} decimal digits"
    )


def _shape_representation(
    basis_exprs: tuple[Any, ...],
    variables: tuple[Any, ...],
) -> _ShapeRepresentation | None:
    parameter = variables[-1]

    eliminants: list[Any] = []
    for expression in basis_exprs:
        if expression.free_symbols <= {parameter} and parameter in expression.free_symbols:
            try:
                degree = sp.Poly(expression, parameter, extension=True).degree()
            except (sp.PolynomialError, TypeError, ValueError):
                continue
            if degree > 0:
                eliminants.append(expression)

    if not eliminants:
        return None
    eliminants.sort(
        key=lambda expr: (
            sp.Poly(expr, parameter, extension=True).degree(),
            int(sp.count_ops(expr)),
        )
    )
    eliminant = eliminants[0]

    coordinate_exprs: list[Any] = []
    for variable in variables[:-1]:
        relation = None
        for expression in basis_exprs:
            if variable not in expression.free_symbols:
                continue
            if not expression.free_symbols <= {variable, parameter}:
                continue
            try:
                polynomial = sp.Poly(expression, variable, domain="EX")
            except (sp.PolynomialError, TypeError, ValueError):
                continue
            if polynomial.degree() != 1:
                continue

            coeff = sp.expand(polynomial.coeff_monomial(variable))
            constant = sp.expand(polynomial.coeff_monomial(1))
            if coeff.free_symbols:
                continue
            if coeff == 0 or coeff.is_algebraic is not True:
                continue

            candidate = sp.cancel(-constant / coeff)
            if candidate.free_symbols <= {parameter}:
                relation = candidate
                break

        if relation is None:
            return None
        coordinate_exprs.append(relation)

    coordinate_exprs.append(parameter)
    return _ShapeRepresentation(
        parameter=parameter,
        eliminant=eliminant,
        coordinate_exprs=tuple(coordinate_exprs),
    )


def _solve_shape(
    representation: _ShapeRepresentation,
    normalized: tuple[Any, ...],
    basis_exprs: tuple[Any, ...],
    variables: tuple[Any, ...],
    *,
    digits: int,
    work_digits: int,
    verification_digits: int,
    maxsteps: int,
    max_solutions: int,
    max_precision_digits: int | None,
) -> PolynomialSystemRoots:
    parameter_roots = _nroots(
        representation.eliminant,
        representation.parameter,
        work_digits,
        maxsteps,
        max_precision_digits,
    )
    if len(parameter_roots) > max_solutions:
        raise SystemSolveLimitError(f"solution count exceeds max_solutions={max_solutions}")

    parameter = representation.parameter
    coordinate_polys = tuple(
        sp.Poly(expression, parameter, extension=True)
        for expression in representation.coordinate_exprs
    )

    def evaluate_coordinate(poly, root):
        with mp.workdps(work_digits + 8):
            value = mp.mpc(0)
            root_mp = sympy_to_mpc(root, work_digits + 8)
            for coefficient in poly.all_coeffs():
                value = value * root_mp + sympy_to_mpc(coefficient, work_digits + 8)
            real = sp.Float(mp.nstr(mp.re(value), work_digits + 5), work_digits)
            imag = sp.Float(mp.nstr(mp.im(value), work_digits + 5), work_digits)
            tiny = sp.Float(10, work_digits) ** (-max(12, work_digits - 5))
            return real if abs(imag) < tiny else real + sp.I * imag

    candidates = [
        tuple(evaluate_coordinate(poly, root) for poly in coordinate_polys)
        for root in parameter_roots
    ]
    roots, diagnostics, max_residual = _validate_roots(
        candidates,
        normalized,
        variables,
        digits=digits,
        work_digits=work_digits,
        verification_digits=verification_digits,
        maxsteps=maxsteps,
        expected_count=len(parameter_roots),
        error_type=NumericalRootError,
    )
    return PolynomialSystemRoots(
        roots=roots,
        variables=variables,
        equations=normalized,
        groebner_basis=basis_exprs,
        precision_digits=digits,
        working_digits=work_digits,
        verification_digits=verification_digits,
        max_relative_residual=max_residual,
        method="shape_position",
        diagnostics=diagnostics,
        eliminant=representation.eliminant,
        parameter_variable=representation.parameter,
    )


def _monomial_divides(left: tuple[int, ...], right: tuple[int, ...]) -> bool:
    return all(a <= b for a, b in zip(left, right, strict=True))


def _standard_monomials(
    basis,
    variables: tuple[Any, ...],
    max_dimension: int,
) -> tuple[tuple[int, ...], ...]:
    """Return the finite Gröbner staircase for a zero-dimensional ideal."""
    leading = tuple(
        tuple(int(e) for e in poly.LM(order=basis.order).exponents)
        for poly in basis.polys
        if poly.total_degree() >= 0
    )
    zero = (0,) * len(variables)
    queue = deque([zero])
    seen = {zero}
    standard: list[tuple[int, ...]] = []

    while queue:
        exponent = queue.popleft()
        if any(_monomial_divides(lm, exponent) for lm in leading):
            continue
        standard.append(exponent)
        if len(standard) > max_dimension:
            raise ActionMatrixError(
                f"quotient dimension exceeds max_action_dimension={max_dimension}"
            )
        for index in range(len(variables)):
            child = list(exponent)
            child[index] += 1
            child_tuple = tuple(child)
            if child_tuple not in seen:
                seen.add(child_tuple)
                queue.append(child_tuple)
    standard.sort(key=lambda exp: (sum(exp), exp))
    return tuple(standard)


def _monomial_expr(exponent: tuple[int, ...], variables: tuple[Any, ...]):
    expression = sp.Integer(1)
    for variable, power in zip(variables, exponent, strict=True):
        if power:
            expression *= variable**power
    return expression


def _remainder_coeffs(
    remainder: Any,
    variables: tuple[Any, ...],
    basis_index: dict[tuple[int, ...], int],
) -> list[Any]:
    polynomial = sp.Poly(sp.expand(remainder), *variables, extension=True)
    values = [sp.Integer(0)] * len(basis_index)
    for exponent, coefficient in polynomial.terms():
        exponent = tuple(int(e) for e in exponent)
        try:
            index = basis_index[exponent]
        except KeyError as exc:
            raise ActionMatrixError(
                "Gröbner remainder contains a monomial outside the quotient basis"
            ) from exc
        values[index] = coefficient
    return values


def _normal_form_coeffs(
    basis,
    expression: Any,
    variables: tuple[Any, ...],
    basis_index: dict[tuple[int, ...], int],
) -> tuple[Any, ...]:
    try:
        _, remainder = basis.reduce(sp.expand(expression))
    except (sp.PolynomialError, TypeError, ValueError) as exc:
        raise ActionMatrixError("exact Gröbner reduction failed") from exc
    return tuple(_remainder_coeffs(remainder, variables, basis_index))


def _separator_matrix_coeffs(
    basis,
    variables: tuple[Any, ...],
    monomials: tuple[tuple[int, ...], ...],
    coefficients: tuple[int, ...],
) -> tuple[tuple[Any, ...], ...]:
    """Return columns of M_L for L=sum(c_i*x_i), using D reductions."""
    basis_index = {exponent: index for index, exponent in enumerate(monomials)}
    linear_form = sum(
        coefficient * variable
        for coefficient, variable in zip(coefficients, variables, strict=True)
    )
    columns = []
    for exponent in monomials:
        product = linear_form * _monomial_expr(exponent, variables)
        columns.append(_normal_form_coeffs(basis, product, variables, basis_index))
    return tuple(columns)


def _coordinate_normal_forms(
    basis,
    variables: tuple[Any, ...],
    monomials: tuple[tuple[int, ...], ...],
) -> tuple[tuple[Any, ...], ...]:
    """Return NF(x_i) in the quotient basis, using one reduction per variable."""
    basis_index = {exponent: index for index, exponent in enumerate(monomials)}
    forms = []
    for index, variable in enumerate(variables):
        exponent = tuple(1 if i == index else 0 for i in range(len(variables)))
        if exponent in basis_index:
            coeffs = [sp.Integer(0)] * len(monomials)
            coeffs[basis_index[exponent]] = sp.Integer(1)
            forms.append(tuple(coeffs))
        else:
            forms.append(_normal_form_coeffs(basis, variable, variables, basis_index))
    return tuple(forms)


def _separator_candidates(count: int) -> tuple[tuple[int, ...], ...]:
    candidates = [tuple(index + 1 for index in range(count))]
    candidates.extend(tuple((index + 1) ** power for index in range(count)) for power in (2, 3, 4))
    candidates.extend(
        tuple(1 if index == pivot else 2 * index + 3 for index in range(count))
        for pivot in range(count)
    )
    unique: list[tuple[int, ...]] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return tuple(unique)


def _acb_separator_matrix(
    columns: tuple[tuple[Any, ...], ...],
    digits: int,
):
    """Construct M_L^T directly as an acb_mat."""
    from flint import acb_mat

    # ``columns[c][r]`` is row ``r`` of column ``c`` of the multiplication
    # matrix M_L.  Evaluation vectors are left eigenvectors of M_L, or
    # equivalently right eigenvectors of M_L.T.  Build the transpose here so
    # the right eigenvectors returned by python-flint have monomial-evaluation
    # coordinates, matching the exact fallback below.
    rows = [[sympy_to_acb(value, digits) for value in column] for column in columns]
    return acb_mat(rows)


def _coords_from_acb_vectors(
    eigenvectors: Any,
    coordinate_forms: tuple[tuple[Any, ...], ...],
    monomials: tuple[tuple[int, ...], ...],
    digits: int,
) -> list[tuple[Any, ...]]:
    constant_index = monomials.index((0,) * len(monomials[0]))
    dimension = len(monomials)
    roots: list[tuple[Any, ...]] = []
    for column in range(dimension):
        denominator = eigenvectors[constant_index, column]
        coordinates = []
        for form in coordinate_forms:
            value = sum(
                (
                    sympy_to_acb(coeff, digits) * eigenvectors[row, column]
                    for row, coeff in enumerate(form)
                    if coeff != 0
                ),
                sympy_to_acb(0, digits),
            )
            try:
                coordinates.append(acb_mid_to_sympy(value / denominator, digits))
            except (ValueError, ZeroDivisionError) as exc:
                raise ActionMatrixError(
                    "separator eigenvector could not be normalized at the constant basis element"
                ) from exc
        roots.append(tuple(coordinates))
    return roots


def _exact_action_fallback(
    columns: tuple[tuple[Any, ...], ...],
    coordinate_forms: tuple[tuple[Any, ...], ...],
    monomials: tuple[tuple[int, ...], ...],
    digits: int,
) -> list[tuple[Any, ...]]:
    """Small development fallback when python-flint is unavailable."""
    dimension = len(columns)
    matrix = sp.Matrix(dimension, dimension, lambda r, c: columns[c][r]).T
    eigenvectors = []
    for _, multiplicity, vectors in matrix.eigenvects():
        if multiplicity != 1 or len(vectors) != 1:
            raise ActionMatrixError("separator has non-simple eigenvalues")
        eigenvectors.append(vectors[0])
    if len(eigenvectors) != dimension:
        raise ActionMatrixError("separator did not yield a complete simple eigenbasis")
    constant_index = monomials.index((0,) * len(monomials[0]))
    roots = []
    for vector in eigenvectors:
        denominator = vector[constant_index]
        if denominator == 0:
            raise ActionMatrixError("separator eigenvector has zero constant coordinate")
        root = []
        for form in coordinate_forms:
            value = sum(coeff * vector[row] for row, coeff in enumerate(form)) / denominator
            root.append(sp.N(value, digits))
        roots.append(tuple(root))
    return roots


def _sympy_numeric(value: Any, digits: int):
    real_text = mp.nstr(mp.re(value), n=digits + 5)
    imag_text = mp.nstr(mp.im(value), n=digits + 5)
    real = sp.Float(real_text, digits)
    imag = sp.Float(imag_text, digits)
    if abs(sp.N(imag, digits)) < sp.Float(10, digits) ** (-max(12, digits - 5)):
        return real
    return real + sp.I * imag


def _mp_root(root: tuple[Any, ...], digits: int) -> tuple[mp.mpc, ...]:
    return tuple(sympy_to_mpc(value, digits) for value in root)


def _sp_root(root: tuple[Any, ...], digits: int) -> tuple[Any, ...]:
    values = []
    tiny = sp.Float(10, digits) ** (-max(12, digits - 5))
    for value in root:
        real = sp.Float(mp.nstr(mp.re(value), digits + 5), digits)
        imag = sp.Float(mp.nstr(mp.im(value), digits + 5), digits)
        values.append(real if abs(imag) < tiny else real + sp.I * imag)
    return tuple(values)


def _deduplicate_records(
    records: list[tuple[tuple[Any, ...], RootDiagnostics]],
    digits: int,
) -> list[tuple[tuple[Any, ...], RootDiagnostics]]:
    tolerance = sp.Float(10, digits) ** (-max(8, digits - 5))
    unique: list[tuple[tuple[Any, ...], RootDiagnostics]] = []
    for root, diagnostics in sorted(records, key=lambda item: _root_sort_key(item[0])):
        duplicate_index = next(
            (
                index
                for index, (prior, _) in enumerate(unique)
                if _roots_close(root, prior, tolerance)
            ),
            None,
        )
        if duplicate_index is None:
            unique.append((root, diagnostics))
            continue
        _, prior_diag = unique[duplicate_index]
        if diagnostics.final_relative_residual < prior_diag.final_relative_residual:
            unique[duplicate_index] = (root, diagnostics)
    return unique


def _validate_roots(
    candidates: Iterable[tuple[Any, ...]],
    normalized: tuple[Any, ...],
    variables: tuple[Any, ...],
    *,
    digits: int,
    work_digits: int,
    verification_digits: int,
    maxsteps: int,
    expected_count: int | None = None,
    error_type: type[PolynomialSystemError] = NumericalRootError,
    compiled_system: _CompiledPolynomialSystem | None = None,
) -> tuple[
    tuple[tuple[Any, ...], ...],
    tuple[RootDiagnostics, ...],
    Any,
]:
    """Refine candidates numerically and verify compiled original polynomials."""
    compiled_system = compiled_system or _CompiledPolynomialSystem.build(normalized, variables)
    polynomials = compiled_system.polynomials
    jacobian = compiled_system.jacobian
    tolerance_mp = mp.power(10, -verification_digits)
    records: list[tuple[tuple[Any, ...], RootDiagnostics]] = []

    for candidate in candidates:
        candidate_mp = _mp_root(tuple(candidate), work_digits + 8)
        initial_mp = compiled_residual(polynomials, candidate_mp, work_digits + 5)
        refine_digits = min(work_digits - 5, verification_digits + 8)
        refine_threshold = mp.power(10, -refine_digits)
        if mp.isfinite(initial_mp) and initial_mp <= refine_threshold:
            refined_mp = candidate_mp
            attempted = False
            succeeded = False
        else:
            attempted = len(normalized) >= len(variables)
            if attempted:
                refined_mp, succeeded = newton_refine(
                    polynomials, jacobian, candidate_mp, work_digits, maxsteps
                )
            else:
                refined_mp, succeeded = candidate_mp, False

        refined_residual = compiled_residual(polynomials, refined_mp, work_digits + 5)
        improved = bool(
            succeeded and mp.isfinite(refined_residual) and refined_residual < initial_mp
        )
        accepted_mp = (
            refined_mp
            if succeeded and mp.isfinite(refined_residual) and refined_residual <= initial_mp
            else candidate_mp
        )
        output_root = _sp_root(accepted_mp, digits)
        output_mp = _mp_root(output_root, work_digits + 5)
        final_mp = compiled_residual(polynomials, output_mp, work_digits + 5)
        if not mp.isfinite(final_mp) or final_mp > tolerance_mp:
            continue

        diagnostics = RootDiagnostics(
            initial_relative_residual=sp.Float(mp.nstr(initial_mp, work_digits), work_digits),
            final_relative_residual=sp.Float(mp.nstr(final_mp, work_digits), work_digits),
            refinement_attempted=attempted,
            refinement_succeeded=succeeded,
            refinement_improved=improved,
        )
        records.append((output_root, diagnostics))

    records = _deduplicate_records(records, digits)
    if expected_count is not None and len(records) != expected_count:
        raise error_type(
            "numerical validation did not retain the expected number of distinct "
            f"roots ({len(records)} != {expected_count})"
        )
    if not records and expected_count != 0:
        raise error_type("candidate roots did not verify against the original polynomial system")

    roots = tuple(root for root, _ in records)
    diagnostics = tuple(diag for _, diag in records)
    max_residual = max(
        (diag.final_relative_residual for diag in diagnostics),
        default=sp.Float(0, work_digits),
    )
    return roots, diagnostics, max_residual


def _solve_action_matrix(
    normalized: tuple[Any, ...],
    basis,
    basis_exprs: tuple[Any, ...],
    variables: tuple[Any, ...],
    *,
    digits: int,
    work_digits: int,
    verification_digits: int,
    maxsteps: int,
    max_solutions: int,
    max_action_dimension: int,
    max_precision_digits: int | None,
) -> PolynomialSystemRoots:
    monomials = _standard_monomials(basis, variables, max_action_dimension)
    dimension = len(monomials)
    if dimension == 0:
        raise ActionMatrixError("zero-dimensional quotient has empty monomial basis")
    if dimension > max_solutions:
        raise SystemSolveLimitError(f"quotient dimension exceeds max_solutions={max_solutions}")

    coordinate_forms = _coordinate_normal_forms(basis, variables, monomials)
    last_validation_error: ActionMatrixError | None = None

    for coefficients in _separator_candidates(len(variables)):
        # Only M_L is constructed. This costs D Gröbner reductions rather than
        # constructing n full multiplication matrices with n*D reductions.
        columns = _separator_matrix_coeffs(basis, variables, monomials, coefficients)

        if not flint_available():  # pragma: no cover - development fallback
            try:
                raw = _exact_action_fallback(columns, coordinate_forms, monomials, work_digits)
                roots, diagnostics, max_residual = _validate_roots(
                    raw,
                    normalized,
                    variables,
                    digits=digits,
                    work_digits=work_digits,
                    verification_digits=verification_digits,
                    maxsteps=maxsteps,
                    expected_count=dimension,
                    error_type=ActionMatrixError,
                )
            except ActionMatrixError as exc:
                last_validation_error = exc
                continue
            return PolynomialSystemRoots(
                roots=roots,
                variables=variables,
                equations=normalized,
                groebner_basis=basis_exprs,
                precision_digits=digits,
                working_digits=work_digits,
                verification_digits=verification_digits,
                max_relative_residual=max_residual,
                method="action_matrix",
                diagnostics=diagnostics,
                quotient_dimension=dimension,
                standard_monomials=monomials,
                separator_coeffs=coefficients,
            )

        max_digits = max_precision_digits or max(work_digits * 4, work_digits + 100)
        current_digits = max(30, work_digits + 5)
        while current_digits <= max_digits:
            with flint_dps(current_digits):
                try:
                    separator = _acb_separator_matrix(columns, current_digits)
                    # The default ACB eigensolver certifies that all eigenvalues
                    # are simple and isolated. Failure triggers a precision retry.
                    eigenvalues, eigenvectors = separator.eig(right=True)
                except (ValueError, ZeroDivisionError):
                    eigenvalues = eigenvectors = None
                if eigenvalues is None or eigenvectors is None or len(eigenvalues) != dimension:
                    current_digits *= 2
                    continue
                try:
                    raw = _coords_from_acb_vectors(
                        eigenvectors, coordinate_forms, monomials, current_digits
                    )
                    roots, diagnostics, max_residual = _validate_roots(
                        raw,
                        normalized,
                        variables,
                        digits=digits,
                        work_digits=max(work_digits, current_digits),
                        verification_digits=verification_digits,
                        maxsteps=maxsteps,
                        expected_count=dimension,
                        error_type=ActionMatrixError,
                    )
                except ActionMatrixError as exc:
                    # A rigorously separated eigensystem can still be too coarse
                    # for coordinate recovery/residual validation. Retry at higher
                    # precision before abandoning this separator.
                    last_validation_error = exc
                    current_digits *= 2
                    continue

                return PolynomialSystemRoots(
                    roots=roots,
                    variables=variables,
                    equations=normalized,
                    groebner_basis=basis_exprs,
                    precision_digits=digits,
                    working_digits=max(work_digits, current_digits),
                    verification_digits=verification_digits,
                    max_relative_residual=max_residual,
                    method="action_matrix",
                    diagnostics=diagnostics,
                    quotient_dimension=dimension,
                    standard_monomials=monomials,
                    separator_coeffs=coefficients,
                )

    if last_validation_error is not None:
        raise ActionMatrixError(
            "all tested separator/precision combinations failed exact root-count "
            "or original-system residual validation"
        ) from last_validation_error
    raise ActionMatrixError(
        "no tested linear form yielded a complete rigorously isolated simple "
        "eigensystem; the ideal may be non-radical or numerically ill-conditioned"
    )


def _candidate_equations(
    basis_exprs: tuple[Any, ...],
    variables: tuple[Any, ...],
    index: int,
) -> list[Any]:
    current = variables[index]
    allowed = set(variables[index:])
    candidates: list[tuple[int, int, int, Any]] = []

    for expression in basis_exprs:
        free = expression.free_symbols
        if current not in free or not free <= allowed:
            continue
        try:
            degree = sp.Poly(
                expression,
                *variables[index:],
                extension=True,
            ).degree(current)
        except (sp.PolynomialError, TypeError, ValueError):
            continue
        if degree > 0:
            candidates.append((degree, len(free), int(sp.count_ops(expression)), expression))

    candidates.sort(key=lambda item: item[:3])
    return [item[3] for item in candidates]


def _solve_univariate_branch(
    expression: Any,
    variable: Any,
    assignment: dict[Any, Any],
    digits: int,
    maxsteps: int,
) -> list[Any] | None:
    numeric_expression = sp.N(expression.subs(assignment), digits)
    try:
        polynomial = sp.Poly(numeric_expression, variable)
    except sp.PolynomialError:
        return None

    if polynomial.degree() <= 0:
        return None

    coefficients = polynomial.all_coeffs()
    if not coefficients:
        return None
    coefficient_scale = max(abs(coefficient) for coefficient in coefficients)
    if coefficient_scale == 0:
        return None

    try:
        roots = sp.nroots(polynomial.sqf_part(), n=digits, maxsteps=maxsteps)
    except (ValueError, ArithmeticError, TypeError) as exc:
        raise NumericalRootError(
            f"numerical root computation failed for variable {variable}"
        ) from exc
    return [sp.N(root, digits) for root in roots]


def _solve_triangular(
    normalized: tuple[Any, ...],
    basis_exprs: tuple[Any, ...],
    variables: tuple[Any, ...],
    *,
    digits: int,
    work_digits: int,
    verification_digits: int,
    maxsteps: int,
    max_solutions: int,
) -> PolynomialSystemRoots:
    branch_tolerance = sp.Float(10, work_digits) ** (-verification_digits)
    branches: list[dict[Any, Any]] = [{}]

    for index in range(len(variables) - 1, -1, -1):
        variable = variables[index]
        candidates = _candidate_equations(basis_exprs, variables, index)
        if not candidates:
            raise TriangularSolveError(
                f"lex Gröbner basis has no triangular equation for {variable}"
            )

        next_branches: list[dict[Any, Any]] = []
        for assignment in branches:
            roots: list[Any] | None = None
            last_error: NumericalRootError | None = None
            for candidate in candidates:
                try:
                    roots = _solve_univariate_branch(
                        candidate,
                        variable,
                        assignment,
                        work_digits,
                        maxsteps,
                    )
                except NumericalRootError as exc:
                    last_error = exc
                    continue
                if roots is not None:
                    break

            if roots is None:
                if last_error is not None:
                    raise last_error
                raise TriangularSolveError(
                    f"triangular equations became degenerate while solving {variable}"
                )

            for root in roots:
                candidate_assignment = dict(assignment)
                candidate_assignment[variable] = sp.N(root, work_digits)
                if _assignment_satisfies(
                    basis_exprs,
                    variables,
                    candidate_assignment,
                    branch_tolerance,
                    work_digits,
                ):
                    next_branches.append(candidate_assignment)

            if len(next_branches) > max_solutions:
                raise SystemSolveLimitError(
                    f"solution branch count exceeds max_solutions={max_solutions}"
                )

        branches = next_branches
        if not branches:
            raise NumericalRootError(
                f"no verified numerical branches remain after solving {variable}"
            )

    candidates = [tuple(assignment[variable] for variable in variables) for assignment in branches]
    roots, diagnostics, max_residual = _validate_roots(
        candidates,
        normalized,
        variables,
        digits=digits,
        work_digits=work_digits,
        verification_digits=verification_digits,
        maxsteps=maxsteps,
        expected_count=None,
        error_type=NumericalRootError,
    )
    if len(roots) > max_solutions:
        raise SystemSolveLimitError(f"solution count exceeds max_solutions={max_solutions}")

    return PolynomialSystemRoots(
        roots=roots,
        variables=variables,
        equations=normalized,
        groebner_basis=basis_exprs,
        precision_digits=digits,
        working_digits=work_digits,
        verification_digits=verification_digits,
        max_relative_residual=max_residual,
        method="triangular_groebner",
        diagnostics=diagnostics,
    )


def _root_sort_key(root: tuple[Any, ...]) -> tuple[float, ...]:
    key: list[float] = []
    for value in root:
        numeric = sp.N(value, 17)
        key.extend((float(sp.re(numeric)), float(sp.im(numeric))))
    return tuple(key)


def _roots_close(
    left: tuple[Any, ...],
    right: tuple[Any, ...],
    tolerance: Any,
    *,
    comparison_digits: int = 30,
) -> bool:
    return all(
        abs(sp.N(a - b, comparison_digits)) <= tolerance for a, b in zip(left, right, strict=True)
    )


def _deduplicate_roots(
    roots: list[tuple[Any, ...]],
    digits: int,
) -> list[tuple[Any, ...]]:
    tolerance = sp.Float(10, digits) ** (-max(8, digits - 5))
    comparison_digits = max(30, digits + 10)
    unique: list[tuple[Any, ...]] = []
    for root in sorted(roots, key=_root_sort_key):
        if not any(
            _roots_close(root, prior, tolerance, comparison_digits=comparison_digits)
            for prior in unique
        ):
            unique.append(root)
    return unique


def _validate_options(
    digits: int,
    verification_digits: int | None,
    guard_digits: int,
    maxsteps: int,
    max_solutions: int,
    max_action_dimension: int,
    max_precision_digits: int | None,
    method: SolveMethod,
    recognize: bool,
    recognition_max_degree: int,
) -> int:
    if not isinstance(digits, int) or digits < 15:
        raise ValueError("digits must be an integer >= 15")
    if verification_digits is None:
        verification_digits = max(10, digits // 2)
    if (
        not isinstance(verification_digits, int)
        or verification_digits < 1
        or verification_digits >= digits
    ):
        raise ValueError("verification_digits must be an integer in [1, digits)")
    if not isinstance(guard_digits, int) or guard_digits < 0:
        raise ValueError("guard_digits must be a nonnegative integer")
    if not isinstance(maxsteps, int) or maxsteps <= 0:
        raise ValueError("maxsteps must be a positive integer")
    if not isinstance(max_solutions, int) or max_solutions <= 0:
        raise ValueError("max_solutions must be a positive integer")
    if not isinstance(max_action_dimension, int) or max_action_dimension <= 0:
        raise ValueError("max_action_dimension must be a positive integer")
    if max_precision_digits is not None and (
        not isinstance(max_precision_digits, int) or max_precision_digits < digits
    ):
        raise ValueError("max_precision_digits must be None or an integer >= digits")
    if method not in {"auto", "shape", "action", "triangular", "rur", "homotopy"}:
        raise ValueError(
            "method must be 'auto', 'shape', 'action', 'triangular', 'rur', or 'homotopy'"
        )
    validate_recognition_options(recognize, recognition_max_degree)
    return verification_digits


@dataclass(frozen=True)
class _PreparedHomotopyProblem:
    base: TotalDegreeHomotopy
    tracker_options: PathTrackerOptions
    compiled_system: _CompiledPolynomialSystem


def _prepare_total_degree_problem(
    normalized: tuple[Any, ...],
    variables: tuple[Any, ...],
    *,
    work_digits: int,
    verification_digits: int,
    maxsteps: int,
    max_precision_digits: int | None,
    options: TotalDegreeHomotopyOptions,
) -> _PreparedHomotopyProblem:
    if len(normalized) != len(variables):
        raise HomotopySolveError("total-degree homotopy requires a square polynomial system")
    base = build_total_degree_homotopy(
        normalized,
        variables,
        seed=options.seed,
        max_paths=options.max_paths,
    )
    default_initial_digits = max(35, work_digits)
    initial_digits = (
        min(default_initial_digits, max_precision_digits)
        if max_precision_digits is not None
        else default_initial_digits
    )
    tracker = PathTrackerOptions(
        initial_digits=initial_digits,
        max_digits=max_precision_digits or max(work_digits * 4, work_digits + 100),
        residual_digits=verification_digits,
        max_steps=max(4000, maxsteps),
        max_newton_steps=max(8, min(maxsteps, 50)),
    )
    return _PreparedHomotopyProblem(
        base=base,
        tracker_options=tracker,
        compiled_system=_CompiledPolynomialSystem.build(normalized, variables),
    )


def _track_total_degree_problem(
    problem: _PreparedHomotopyProblem,
    gamma: Any,
    options: TotalDegreeHomotopyOptions,
) -> tuple[PathResult, ...]:
    return track_total_degree_paths(
        problem.base.with_gamma(gamma),
        options=problem.tracker_options,
        parallel=options.parallel,
        max_workers=options.max_workers,
    )


def _resolve_endpoint_regularity(
    candidate: tuple[Any, ...],
    compiled: _CompiledPolynomialSystem,
    *,
    start_digits: int,
    max_digits: int,
    verification_digits: int,
    maxsteps: int,
) -> tuple[tuple[Any, ...], _EndpointRegularity]:
    """Re-refine an endpoint until nonsingularity is numerically resolved.

    Very small singular values are evaluated at successively higher precision.
    A genuine small-but-nonzero singular value stabilizes, while an apparent
    singular value caused by an endpoint converging to a multiple root shrinks
    with the refinement precision.
    """
    digits = min(max_digits, max(30, start_digits))
    values = _mp_root(candidate, digits + 8)
    previous_smallest: mp.mpf | None = None
    suspicion_threshold = mp.power(10, -max(6, verification_digits // 2))
    while True:
        refined, _ = newton_refine(
            compiled.polynomials,
            compiled.jacobian,
            values,
            digits,
            maxsteps,
        )
        candidate_sp = _sp_root(refined, digits)
        regularity = compiled.endpoint_regularity(
            candidate_sp,
            digits=digits,
            verification_digits=verification_digits,
        )
        smallest = regularity.smallest_singular_value
        if regularity.resolved and smallest > suspicion_threshold:
            return candidate_sp, regularity
        if (
            previous_smallest is not None
            and smallest > 0
            and previous_smallest > 0
            and mp.isfinite(smallest)
            and mp.isfinite(previous_smallest)
        ):
            ratio = max(smallest, previous_smallest) / min(smallest, previous_smallest)
            if ratio <= mp.mpf(100):
                return candidate_sp, _EndpointRegularity(
                    smallest,
                    regularity.condition_estimate,
                    digits,
                    True,
                )
        if digits >= max_digits:
            return candidate_sp, _EndpointRegularity(
                smallest,
                regularity.condition_estimate,
                digits,
                False,
            )
        previous_smallest = smallest
        next_digits = min(max_digits, max(digits + 20, digits * 2))
        if next_digits <= digits:
            return candidate_sp, regularity
        values = refined
        digits = next_digits


def _validate_homotopy_endpoints(
    paths: tuple[PathResult, ...],
    normalized: tuple[Any, ...],
    variables: tuple[Any, ...],
    *,
    digits: int,
    work_digits: int,
    verification_digits: int,
    maxsteps: int,
    max_precision_digits: int | None,
    compiled_system: _CompiledPolynomialSystem,
) -> tuple[
    tuple[tuple[Any, ...], ...],
    tuple[RootDiagnostics, ...],
    Any,
    tuple[_EndpointRegularity, ...],
]:
    max_digits = max_precision_digits or max(work_digits * 4, work_digits + 100)
    candidates: list[tuple[Any, ...]] = []
    regularities: list[_EndpointRegularity] = []
    for path in paths:
        candidate, regularity = _resolve_endpoint_regularity(
            tuple(path.endpoint),
            compiled_system,
            start_digits=max(work_digits, path.final_digits),
            max_digits=max_digits,
            verification_digits=verification_digits,
            maxsteps=maxsteps,
        )
        if not regularity.resolved:
            raise HomotopySolveError(
                "total-degree homotopy reached a singular or numerically unresolved "
                "endpoint; singular endgames are not implemented yet"
            )
        candidates.append(candidate)
        regularities.append(regularity)
    roots, diagnostics, max_residual = _validate_roots(
        candidates,
        normalized,
        variables,
        digits=digits,
        work_digits=max(
            work_digits,
            max((r.digits for r in regularities), default=work_digits),
        ),
        verification_digits=verification_digits,
        maxsteps=maxsteps,
        expected_count=len(paths),
        error_type=HomotopySolveError,
        compiled_system=compiled_system,
    )
    tolerance = sp.Float(10, digits) ** (-max(8, digits - 5))
    comparison_digits = max(30, digits + 10)
    unused = set(range(len(candidates)))
    aligned: list[_EndpointRegularity] = []
    for root in roots:
        match = next(
            (
                index
                for index in unused
                if _roots_close(
                    root,
                    candidates[index],
                    tolerance,
                    comparison_digits=comparison_digits,
                )
            ),
            None,
        )
        if match is None:  # pragma: no cover - guarded by equal-count validation
            raise HomotopySolveError(
                "homotopy endpoint diagnostics could not be aligned with roots"
            )
        unused.remove(match)
        aligned.append(regularities[match])
    return roots, diagnostics, max_residual, tuple(aligned)


def _solve_total_degree_homotopy(
    normalized: tuple[Any, ...],
    variables: tuple[Any, ...],
    *,
    digits: int,
    work_digits: int,
    verification_digits: int,
    maxsteps: int,
    max_solutions: int,
    max_precision_digits: int | None,
    homotopy_options: TotalDegreeHomotopyOptions,
) -> PolynomialSystemRoots:
    # A nonzero constant equation makes the affine system inconsistent and
    # requires no path tracking. An identically zero equation is rejected by
    # the constructor because it falls outside the regular square-system scope.
    for equation in normalized:
        polynomial = sp.Poly(equation, *variables, extension=True)
        if not polynomial.is_zero and polynomial.total_degree() == 0:
            return PolynomialSystemRoots(
                roots=(),
                variables=variables,
                equations=normalized,
                groebner_basis=(),
                precision_digits=digits,
                working_digits=work_digits,
                verification_digits=verification_digits,
                max_relative_residual=sp.Integer(0),
                method="total_degree_homotopy",
                homotopy_paths_total=0,
                homotopy_paths_succeeded=0,
                homotopy_paths_failed=0,
                homotopy_paths_divergent=0,
                homotopy_gamma_attempts=0,
            )

    problem = _prepare_total_degree_problem(
        normalized,
        variables,
        work_digits=work_digits,
        verification_digits=verification_digits,
        maxsteps=maxsteps,
        max_precision_digits=max_precision_digits,
        options=homotopy_options,
    )
    path_count = problem.base.path_count
    if path_count > max_solutions:
        raise SystemSolveLimitError(
            f"total-degree homotopy path count exceeds max_solutions={max_solutions}"
        )

    failures: list[str] = []
    for attempt, gamma in enumerate(
        gamma_candidates(homotopy_options.seed, homotopy_options.gamma_attempts),
        start=1,
    ):
        paths = _track_total_degree_problem(problem, gamma, homotopy_options)
        failed = tuple(path for path in paths if not path.success)
        divergent = tuple(path for path in failed if path.classification == "likely_divergent")
        if failed:
            messages = sorted({path.message for path in failed if path.message})
            detail = messages[0] if messages else "path tracking failed"
            failures.append(
                f"gamma attempt {attempt}: {len(failed)}/{path_count} paths failed "
                f"({len(divergent)} likely divergent); {detail}"
            )
            continue
        try:
            roots, diagnostics, max_residual, regularities = _validate_homotopy_endpoints(
                paths,
                normalized,
                variables,
                digits=digits,
                work_digits=work_digits,
                verification_digits=verification_digits,
                maxsteps=maxsteps,
                max_precision_digits=max_precision_digits,
                compiled_system=problem.compiled_system,
            )
        except HomotopySolveError as exc:
            failures.append(f"gamma attempt {attempt}: {exc}")
            continue
        return PolynomialSystemRoots(
            roots=roots,
            variables=variables,
            equations=normalized,
            groebner_basis=(),
            precision_digits=digits,
            working_digits=max(
                [
                    work_digits,
                    *(path.final_digits for path in paths),
                    *(r.digits for r in regularities),
                ]
            ),
            verification_digits=verification_digits,
            max_relative_residual=max_residual,
            method="total_degree_homotopy",
            diagnostics=diagnostics,
            homotopy_paths_total=path_count,
            homotopy_paths_succeeded=len(paths),
            homotopy_paths_failed=0,
            homotopy_paths_divergent=0,
            homotopy_gamma=gamma,
            homotopy_gamma_attempts=attempt,
            homotopy_endpoint_smallest_singular_values=tuple(
                sp.Float(mp.nstr(r.smallest_singular_value, max(20, digits)), digits)
                for r in regularities
            ),
            homotopy_endpoint_condition_estimates=tuple(
                sp.Float(mp.nstr(r.condition_estimate, max(20, digits)), digits)
                if mp.isfinite(r.condition_estimate)
                else sp.oo
                for r in regularities
            ),
        )

    detail = "; ".join(failures[:3])
    raise HomotopySolveError(
        "total-degree homotopy failed for every deterministic gamma candidate"
        + (f": {detail}" if detail else "")
        + ". Singular endgames and projective tracking are not implemented yet."
    )


def _solve_rur(
    normalized: tuple[Any, ...],
    variables: tuple[Any, ...],
    *,
    digits: int,
    work_digits: int,
    verification_digits: int,
    maxsteps: int,
    max_solutions: int,
) -> PolynomialSystemRoots:
    from .rational_univariate import compute_rational_univariate_representation
    from .rational_univariate.solve import numerical_rur_roots

    representation = compute_rational_univariate_representation(normalized, variables)
    if representation.solution_count > max_solutions:
        raise SystemSolveLimitError(f"RUR solution count exceeds max_solutions={max_solutions}")
    candidates = list(
        numerical_rur_roots(representation, digits=work_digits, maxsteps=max(100, maxsteps))
    )
    roots, diagnostics, max_residual = _validate_roots(
        candidates,
        normalized,
        variables,
        digits=digits,
        work_digits=work_digits,
        verification_digits=verification_digits,
        maxsteps=maxsteps,
        expected_count=representation.solution_count,
        error_type=NumericalRootError,
    )
    return PolynomialSystemRoots(
        roots=roots,
        variables=variables,
        equations=normalized,
        groebner_basis=(),
        precision_digits=digits,
        working_digits=work_digits,
        verification_digits=verification_digits,
        max_relative_residual=max_residual,
        method="rational_univariate",
        diagnostics=diagnostics,
        quotient_dimension=representation.quotient_dimension,
        standard_monomials=representation.standard_exponents,
        rational_univariate_representation=representation,
    )


def polysolve(
    equations: Iterable[Any],
    variables: Sequence[Any],
    *,
    digits: int = 50,
    verification_digits: int | None = None,
    guard_digits: int = 10,
    maxsteps: int = 200,
    max_solutions: int = 10_000,
    max_action_dimension: int = 256,
    max_precision_digits: int | None = None,
    method: SolveMethod = "auto",
    recognize: bool = True,
    recognition_max_degree: int = 8,
    max_homotopy_paths: int = 10_000,
    homotopy_seed: int = 0,
    homotopy_gamma_attempts: int = 4,
    homotopy_parallel: bool = False,
    homotopy_max_workers: int | None = None,
) -> PolynomialSystemRoots:
    """Find all distinct numerical roots of an exact zero-dimensional system.

    ``method="auto"`` first uses shape position when available, then an
    action-matrix quotient-algebra solver, and finally triangular
    back-substitution when the action backend declines. ``method="shape"``
    requires shape position; ``method="action"`` requires a regular separated
    action-matrix solve. ``method="rur"`` uses an exact rational univariate
    representation over `QQ` or an exact algebraic number field.
    ``method="homotopy"`` explicitly selects total-degree homotopy continuation
    for regular square systems whose Bézout paths have finite nonsingular
    endpoints. It is not selected by ``method="auto"``.

    Exact algebraic recognition is attempted by default after the numerical
    solve. Successfully recognized roots are stored in ``recognized_roots``.
    Recognition failure does not invalidate an otherwise verified numerical
    solve; the failure is recorded in ``recognition_error``. Pass
    ``recognize=False`` to skip recognition entirely.

    Equations may be polynomial expressions interpreted as ``expr == 0`` or
    SymPy ``Eq`` objects. Coefficients must be exact algebraic numbers.
    """

    def finish(result: PolynomialSystemRoots) -> PolynomialSystemRoots:
        from .recognition import _auto_recognize

        return _auto_recognize(result, recognize=recognize, max_degree=recognition_max_degree)

    verification_digits = _validate_options(
        digits,
        verification_digits,
        guard_digits,
        maxsteps,
        max_solutions,
        max_action_dimension,
        max_precision_digits,
        method,
        recognize,
        recognition_max_degree,
    )
    homotopy_options = TotalDegreeHomotopyOptions(
        max_paths=max_homotopy_paths,
        seed=homotopy_seed,
        parallel=homotopy_parallel,
        max_workers=homotopy_max_workers,
        gamma_attempts=homotopy_gamma_attempts,
    )
    variables = validate_variables(variables)
    normalized = _normalize_equations(equations, variables)
    work_digits = digits + guard_digits
    if method == "homotopy":
        return finish(
            _solve_total_degree_homotopy(
                normalized,
                variables,
                digits=digits,
                work_digits=work_digits,
                verification_digits=verification_digits,
                maxsteps=maxsteps,
                max_solutions=max_solutions,
                max_precision_digits=max_precision_digits,
                homotopy_options=homotopy_options,
            )
        )
    if method == "rur":
        return finish(
            _solve_rur(
                normalized,
                variables,
                digits=digits,
                work_digits=work_digits,
                verification_digits=verification_digits,
                maxsteps=maxsteps,
                max_solutions=max_solutions,
            )
        )

    basis = _groebner_basis(normalized, variables)
    basis_exprs = tuple(poly.as_expr() for poly in basis.polys)

    if _is_unit_ideal(basis):
        result = PolynomialSystemRoots(
            roots=(),
            variables=variables,
            equations=normalized,
            groebner_basis=basis_exprs,
            precision_digits=digits,
            working_digits=digits + guard_digits,
            verification_digits=verification_digits,
            max_relative_residual=sp.Integer(0),
            method="shape_position" if method == "shape" else "triangular_groebner",
        )
        return finish(result)

    if not basis.is_zero_dimensional:
        raise NotZeroDimensionalError("the polynomial system is not zero-dimensional")

    representation = _shape_representation(basis_exprs, variables)
    if method in {"auto", "shape"} and representation is not None:
        result = _solve_shape(
            representation,
            normalized,
            basis_exprs,
            variables,
            digits=digits,
            work_digits=work_digits,
            verification_digits=verification_digits,
            maxsteps=maxsteps,
            max_solutions=max_solutions,
            max_precision_digits=max_precision_digits,
        )
        return finish(result)

    if method == "shape":
        raise ShapePositionError("the lex Gröbner basis is not in supported shape position")

    if method in {"auto", "action"}:
        try:
            result = _solve_action_matrix(
                normalized,
                basis,
                basis_exprs,
                variables,
                digits=digits,
                work_digits=work_digits,
                verification_digits=verification_digits,
                maxsteps=maxsteps,
                max_solutions=max_solutions,
                max_action_dimension=max_action_dimension,
                max_precision_digits=max_precision_digits,
            )
            return finish(result)
        except ActionMatrixError:
            if method == "action":
                raise

    result = _solve_triangular(
        normalized,
        basis_exprs,
        variables,
        digits=digits,
        work_digits=work_digits,
        verification_digits=verification_digits,
        maxsteps=maxsteps,
        max_solutions=max_solutions,
    )
    return finish(result)
