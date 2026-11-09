"""Active-column, native shifted-row, and lazy compact proof regressions."""

from random import Random

import pytest

from algroots.ff_sparse_matrix import (
    assemble_shifted_rows,
    compiled_available,
    reduce_rows,
)


@pytest.mark.parametrize("prime", [2, 3, 101, 1009, 65537])
@pytest.mark.parametrize("ncols", [12, 2000, 30000])
def test_active_columns_replay_exact(prime, ncols):
    if not compiled_available():
        pytest.skip("native extension not built")
    rng = Random(1234 + prime + ncols)
    columns = tuple(sorted(set([0, 1, 2, 7, ncols // 2, ncols - 1])))
    rows = []
    for index in range(15):
        colset = rng.sample(columns, rng.randrange(0, min(5, len(columns)) + 1))
        rows.append({col: rng.randrange(-3 * prime, 3 * prime) for col in colset})
        if index == 7:
            rows.append(dict(rows[-2]))
    expected = reduce_rows(rows, ncols, prime, backend="python")
    for mode in ("dense", "sparse", "adaptive"):
        actual = reduce_rows(rows, ncols, prime, backend="compiled", active_columns=mode)
        assert actual == expected
        for i, (row, operations, inverse) in enumerate(actual):
            replay = {col: value % prime for col, value in rows[i].items() if value % prime}
            for previous, factor in operations:
                assert previous < i
                for col, value in actual[previous][0].items():
                    result = (replay.get(col, 0) - factor * value) % prime
                    if result:
                        replay[col] = result
                    else:
                        replay.pop(col, None)
            assert {col: value * inverse % prime for col, value in replay.items()} == row


def test_native_shifted_row_assembler_and_validation():
    if not compiled_available():
        pytest.skip("native extension not built")
    basis = (
        (((2, 0), -3), ((0, 1), 5)),
        (((1, 1), 6), ((0, 0), 8)),
    )
    specifications = (((0, 2), 0), ((3, 1), 1), ((1, 0), 0))
    positions = {}
    for shift, index in specifications:
        for powers, _ in basis[index]:
            positions[tuple(a + b for a, b in zip(powers, shift, strict=False))] = 0
    positions = {power: i for i, power in enumerate(sorted(positions))}
    for prime in (3, 101, 1009):
        assert assemble_shifted_rows(
            basis, specifications, positions, prime, backend="compiled"
        ) == assemble_shifted_rows(basis, specifications, positions, prime, backend="python")
    with pytest.raises(KeyError):
        assemble_shifted_rows(basis, specifications, {}, 101, backend="compiled")
