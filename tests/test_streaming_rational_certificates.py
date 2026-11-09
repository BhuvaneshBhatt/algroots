"""Streaming rational membership certificates and end-to-end modular replay."""

import sympy as sp
from sympy.polys.rings import PolyRing

from algroots.rational_certificate import verify_sparse_memberships


def test_sparse_replay_rejects_malformed_shapes_and_coefficients():
    x, y = sp.symbols("x y")
    ring = PolyRing((x, y), sp.QQ, "lex")
    inputs = (ring.from_expr(x + y),)
    candidate = inputs
    assert not verify_sparse_memberships(inputs, candidate, (({(-1, 0): 1},),), ring)
    assert not verify_sparse_memberships(inputs, candidate, (({(0, 0): "nonsense"},),), ring)
    assert not verify_sparse_memberships(inputs, candidate, (({(0,): 1},),), ring)
    assert not verify_sparse_memberships(inputs, candidate, (({}, {}),), ring)
    assert not verify_sparse_memberships(inputs, candidate, (), ring)
