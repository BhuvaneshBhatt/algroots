"""Exact linear-algebra kernels used by rational-univariate construction."""

from __future__ import annotations

import sympy as sp


def krylov_minimal_polynomial(matrix: sp.Matrix, parameter: sp.Symbol) -> sp.Poly:
    """Return the minimal polynomial detected from the cyclic vector ``e_0``.

    RUR separating forms use ``e_0`` as the quotient class of one. The first
    exact linear dependence among its Krylov vectors gives the annihilating
    polynomial for that cyclic subspace.
    """
    dimension = matrix.rows
    if matrix.cols != dimension:
        raise ValueError("matrix must be square")
    if dimension == 0:
        return sp.Poly(1, parameter)
    vectors = [sp.eye(dimension).col(0)]
    for degree in range(1, dimension + 1):
        candidate = matrix * vectors[-1]
        basis = sp.Matrix.hstack(*vectors)
        augmented_rank = basis.row_join(candidate).rank()
        if augmented_rank == basis.rank():
            solution = basis.gauss_jordan_solve(-candidate)[0]
            expr = parameter**degree + sum(solution[i] * parameter**i for i in range(degree))
            return sp.Poly(sp.expand(expr), parameter)
        vectors.append(candidate)
    raise ValueError("Krylov sequence did not close within matrix dimension")


def packed_krylov_minimal_polynomial(matrix: sp.Matrix, parameter: sp.Symbol) -> sp.Poly:
    """Return the cyclic minimal polynomial using incremental exact elimination.

    Pivot rows and their Krylov-combination rows are retained across steps.
    Each new vector is reduced once; no rank computation or generic linear
    solve is performed inside the Krylov loop. A lower-degree closure means
    the class of one is not cyclic for the matrix and is reported explicitly.
    """
    dimension = matrix.rows
    if matrix.cols != dimension:
        raise ValueError("matrix must be square")
    if dimension == 0:
        return sp.Poly(1, parameter)

    pivot_rows: list[list[sp.Expr]] = []
    combination_rows: list[list[sp.Expr]] = []
    pivots: list[int] = []
    vector = [sp.Integer(i == 0) for i in range(dimension)]

    for degree in range(dimension + 1):
        row = list(vector)
        combination = [sp.Integer(0)] * (dimension + 1)
        combination[degree] = sp.Integer(1)

        for pivot, pivot_row, pivot_combination in zip(
            pivots, pivot_rows, combination_rows, strict=True
        ):
            factor = row[pivot]
            if factor == 0:
                continue
            row = [value - factor * base for value, base in zip(row, pivot_row, strict=True)]
            combination = [
                value - factor * base
                for value, base in zip(combination, pivot_combination, strict=True)
            ]

        pivot = next((index for index, value in enumerate(row) if value != 0), None)
        if pivot is None:
            if degree != dimension:
                raise ValueError(
                    "Krylov sequence closed before full dimension; the class of one is not cyclic"
                )
            leading = combination[degree]
            coefficients = [sp.cancel(value / leading) for value in combination[: degree + 1]]
            expr = sum(coefficients[index] * parameter**index for index in range(degree + 1))
            return sp.Poly(sp.expand(expr), parameter).monic()

        scale = row[pivot]
        row = [sp.cancel(value / scale) for value in row]
        combination = [sp.cancel(value / scale) for value in combination]
        pivots.append(pivot)
        pivot_rows.append(row)
        combination_rows.append(combination)

        if degree < dimension:
            vector = list(matrix * sp.Matrix(vector))

    raise ValueError("Krylov sequence did not close within matrix dimension")


def packed_krylov_coefficients(rows, domain):
    """Return a cyclic characteristic polynomial from exact domain rows.

    ``rows`` is a square row-major matrix whose entries are coercible into
    ``domain``. The coefficient tuple is returned in ascending degree order.
    Incremental elimination retains pivot and combination rows, while matrix-
    vector products stay in the coefficient domain.  A short Krylov closure is
    reported because it cannot replace the characteristic polynomial in RUR
    construction.
    """
    dimension = len(rows)
    if any(len(row) != dimension for row in rows):
        raise ValueError("matrix must be square")
    if dimension == 0:
        return (domain.one,)

    matrix = [[domain.convert(value) for value in row] for row in rows]
    zero, one = domain.zero, domain.one
    pivot_rows = []
    combination_rows = []
    pivots = []
    vector = [one] + [zero] * (dimension - 1)

    for degree in range(dimension + 1):
        row = list(vector)
        combination = [zero] * (dimension + 1)
        combination[degree] = one
        for pivot, pivot_row, pivot_combination in zip(
            pivots, pivot_rows, combination_rows, strict=True
        ):
            factor = row[pivot]
            if not factor:
                continue
            row = [value - factor * base for value, base in zip(row, pivot_row, strict=True)]
            combination = [
                value - factor * base
                for value, base in zip(combination, pivot_combination, strict=True)
            ]
        pivot = next((index for index, value in enumerate(row) if value), None)
        if pivot is None:
            if degree != dimension:
                raise ValueError(
                    "Krylov sequence closed before full dimension; the class of one is not cyclic"
                )
            leading = combination[degree]
            return tuple(value / leading for value in combination[: degree + 1])

        scale = row[pivot]
        row = [value / scale for value in row]
        combination = [value / scale for value in combination]
        pivots.append(pivot)
        pivot_rows.append(row)
        combination_rows.append(combination)

        if degree < dimension:
            vector = [
                sum(
                    (matrix[row_index][column] * vector[column] for column in range(dimension)),
                    zero,
                )
                for row_index in range(dimension)
            ]

    raise ValueError("Krylov sequence did not close within matrix dimension")


def matrix_density(matrix: sp.Matrix) -> sp.Rational:
    """Return the exact fraction of nonzero entries in a matrix."""
    size = matrix.rows * matrix.cols
    if not size:
        return sp.Rational(0)
    nonzero = (
        len(matrix.todok())
        if isinstance(matrix, sp.SparseMatrix)
        else sum(value != 0 for value in matrix)
    )
    return sp.Rational(nonzero, size)


def preferred_matrix_storage(matrix: sp.Matrix, *, sparse_cutoff=None) -> str:
    """Choose dense or sparse storage from measured matrix density."""
    cutoff = sp.Rational(1, 10) if sparse_cutoff is None else sparse_cutoff
    return "sparse" if matrix_density(matrix) < cutoff else "dense"


# Sparse production separators may use packed Krylov with full cyclic closure.
# Dense and noncyclic separators retain the characteristic-polynomial fallback.
__all__ = [
    "krylov_minimal_polynomial",
    "packed_krylov_minimal_polynomial",
    "packed_krylov_coefficients",
    "matrix_density",
    "preferred_matrix_storage",
]
