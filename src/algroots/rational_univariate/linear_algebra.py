"""Exact linear-algebra kernels used by rational-univariate construction."""

from __future__ import annotations

import sympy as sp
from sympy.polys.matrices import DomainMatrix


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


def packed_krylov_coefficients(rows, domain, *, allow_short_closure=False):
    """Return a cyclic characteristic polynomial from exact domain rows.

    ``rows`` is a square row-major matrix whose entries are coercible into
    ``domain``. The coefficient tuple is returned in ascending degree order.
    Incremental elimination retains pivot and combination rows, while matrix-
    vector products stay in the coefficient domain.  A short Krylov closure is
    reported because it cannot replace the characteristic polynomial in RUR
    construction. ``allow_short_closure=True`` instead returns the annihilator
    of the cyclic subspace, as needed for quotient-element minimal polynomials.
    """
    if isinstance(rows, DomainMatrix):
        if rows.shape[0] != rows.shape[1]:
            raise ValueError("matrix must be square")
        if rows.rep.fmt == "sparse":
            return sparse_krylov_coefficients(rows, allow_short_closure=allow_short_closure)
        rows = rows.to_list()
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
            if degree != dimension and not allow_short_closure:
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


def _subtract_scaled(row, other, scale):
    for column, value in other.items():
        result = row.get(column, 0) - scale * value
        if result:
            row[column] = result
        else:
            row.pop(column, None)


def sparse_krylov_coefficients(matrix, *, allow_short_closure=False):
    """Exact cyclic dependence using sparse actions and sparse pivot rows.

    Only nonzero matrix entries participate in each matrix-vector product.
    Pivot and combination rows fill as required by the problem; no dense copy
    of the action is made. Short closure is valid for a minimal polynomial,
    while characteristic-polynomial callers must require full closure.
    """
    dimension, columns = matrix.shape
    if columns != dimension or not matrix.domain.is_Field:
        raise ValueError("Krylov iteration requires a square matrix over a field")
    domain = matrix.domain
    if not dimension:
        return (domain.one,)
    rows = matrix.to_dod()
    vector = {0: domain.one}
    pivots = []
    for degree in range(dimension + 1):
        row, combination = dict(vector), {degree: domain.one}
        for pivot, previous, witness in pivots:
            scale = row.get(pivot)
            if scale:
                _subtract_scaled(row, previous, scale)
                _subtract_scaled(combination, witness, scale)
        if not row:
            if degree != dimension and not allow_short_closure:
                raise ValueError(
                    "Krylov sequence closed before full dimension; the class of one is not cyclic"
                )
            leading = combination[degree]
            return tuple(combination.get(i, domain.zero) / leading for i in range(degree + 1))
        pivot = min(row)
        scale = row[pivot]
        pivots.append(
            (
                pivot,
                {i: c / scale for i, c in row.items()},
                {i: c / scale for i, c in combination.items()},
            )
        )
        next_vector = {}
        for i, entries in rows.items():
            value = sum((c * vector[j] for j, c in entries.items() if j in vector), domain.zero)
            if value:
                next_vector[i] = value
        vector = next_vector
    raise ValueError("Krylov sequence did not close within matrix dimension")


def sparse_solve(matrix, column):
    """Solve one exact square system without constructing an inverse.

    Sparse elimination includes row pivoting and detects singularity even when
    the supplied right-hand side happens to belong to the image.
    """
    n, m = matrix.shape
    if n != m or column.shape != (n, 1) or matrix.domain != column.domain:
        raise ValueError("solve requires a square matrix and one matching domain column")
    from sympy.polys.matrices.exceptions import DMNonInvertibleMatrixError

    domain = matrix.domain
    entries, values = matrix.to_dod(), column.to_dod()
    rows = [dict(entries.get(i, {})) for i in range(n)]
    rhs = [values.get(i, {}).get(0, domain.zero) for i in range(n)]
    columns = [set() for _ in range(n)]
    for i, row in enumerate(rows):
        for j in row:
            columns[j].add(i)
    for pivot in range(n):
        candidates = columns[pivot]
        selected = min((i for i in candidates if i >= pivot), default=None)
        if selected is None:
            raise DMNonInvertibleMatrixError("singular matrix")
        if selected != pivot:
            for j in rows[pivot].keys() | rows[selected].keys():
                columns[j].discard(pivot)
                columns[j].discard(selected)
                if j in rows[pivot]:
                    columns[j].add(selected)
                if j in rows[selected]:
                    columns[j].add(pivot)
            rows[pivot], rows[selected] = rows[selected], rows[pivot]
            rhs[pivot], rhs[selected] = rhs[selected], rhs[pivot]
        scale = rows[pivot][pivot]
        rows[pivot] = {j: c / scale for j, c in rows[pivot].items()}
        rhs[pivot] /= scale
        for i in tuple(columns[pivot]):
            if i <= pivot:
                continue
            scale = rows[i][pivot]
            previous = set(rows[i])
            _subtract_scaled(rows[i], rows[pivot], scale)
            for j in previous - rows[i].keys():
                columns[j].discard(i)
            for j in rows[i].keys() - previous:
                columns[j].add(i)
            rhs[i] -= scale * rhs[pivot]
    solution = [domain.zero] * n
    for i in range(n - 1, -1, -1):
        solution[i] = rhs[i] - sum(
            (c * solution[j] for j, c in rows[i].items() if j > i), domain.zero
        )
    return DomainMatrix({i: {0: c} for i, c in enumerate(solution) if c}, (n, 1), domain)


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


def sparse_characteristic_coefficients(matrix):
    """Return exact characteristic coefficients using cyclic invariant quotients.

    Each Krylov chain closes modulo the invariant space of previous chains.
    Its monic dependence is the characteristic factor on that cyclic quotient.
    Their product keeps multiplicities even when no single cyclic vector spans
    the matrix. Column adjacency avoids scanning an entire sparse action for
    every unit-vector start, and no dense matrix is constructed.
    """
    n, m = matrix.shape
    if n != m or not matrix.domain.is_Field:
        raise ValueError("characteristic computation requires a square matrix over a field")
    domain = matrix.domain
    columns = {}
    for i, row in matrix.to_dod().items():
        for j, value in row.items():
            if value:
                columns.setdefault(j, {})[i] = value
    pivots = {}
    factors = []
    start = 0
    while len(pivots) < n:
        while start in pivots:
            start += 1
        vector = {start: domain.one}
        chain = len(factors)
        degree = 0
        while True:
            row, witness = dict(vector), {degree: domain.one}
            while True:
                candidates = (p for p in row if p in pivots)
                pivot = min(candidates, key=lambda p: pivots[p][0], default=None)
                if pivot is None:
                    break
                _, prior_chain, previous, combination = pivots[pivot]
                scale = row[pivot]
                _subtract_scaled(row, previous, scale)
                if prior_chain == chain:
                    _subtract_scaled(witness, combination, scale)
            if not row:
                leading = witness[degree]
                factors.append([witness.get(i, domain.zero) / leading for i in range(degree + 1)])
                break
            pivot = min(row)
            scale = row[pivot]
            pivots[pivot] = (
                len(pivots),
                chain,
                {i: c / scale for i, c in row.items()},
                {i: c / scale for i, c in witness.items()},
            )
            following = {}
            for j, value in vector.items():
                for i, coefficient in columns.get(j, {}).items():
                    following[i] = following.get(i, domain.zero) + coefficient * value
            vector = {i: c for i, c in following.items() if c}
            degree += 1
    if not factors:
        return [domain.one]
    # A balanced product prevents repeated multiplication by long partial
    # products when a matrix has many short invariant factors.
    ring = domain.poly_ring(sp.Dummy("charpoly"))
    polynomials = [
        ring.from_sympy(
            sum(domain.to_sympy(c) * ring.symbols[0] ** i for i, c in enumerate(factor))
        )
        for factor in factors
    ]
    while len(polynomials) > 1:
        polynomials = [
            polynomials[i] * polynomials[i + 1] if i + 1 < len(polynomials) else polynomials[i]
            for i in range(0, len(polynomials), 2)
        ]
    terms = polynomials[0].to_dict()
    return [terms.get((i,), domain.zero) for i in range(n, -1, -1)]


__all__ = [
    "krylov_minimal_polynomial",
    "packed_krylov_minimal_polynomial",
    "packed_krylov_coefficients",
    "sparse_krylov_coefficients",
    "sparse_characteristic_coefficients",
    "sparse_solve",
    "matrix_density",
    "preferred_matrix_storage",
]
