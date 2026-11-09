"""Differential and adversarial checks for optional proof-performance paths."""

import sympy as sp
from sympy.polys.rings import PolyRing

from algroots.basis_verification import verify_rational_basis


def test_flint_native_sparse_and_expanded_witnesses_reject_tampering():
    x, y = sp.symbols("x y")
    ring = PolyRing((x, y), sp.QQ, "lex")
    original = (ring.from_expr(x**2 - y), ring.from_expr(y**2 - 2))
    sparse = (({(0, 0): sp.Integer(1)}, {}), ({}, {(0, 0): sp.Integer(1)}))
    expanded = ((ring.one, ring.zero), (ring.zero, ring.one))
    for backend in ("sympy", "flint"):
        assert verify_rational_basis(
            original,
            original,
            ring,
            sparse_memberships=sparse,
            rational_normal_form_backend=backend,
        )
        assert verify_rational_basis(
            original,
            original,
            ring,
            membership_witnesses=expanded,
            rational_normal_form_backend=backend,
        )
        forged = (({(0, 0): sp.Integer(1), (1, 0): sp.Integer(1)}, {}), sparse[1])
        assert not verify_rational_basis(
            original,
            original,
            ring,
            sparse_memberships=forged,
            rational_normal_form_backend=backend,
        )
        forged_expanded = ((ring.one + ring.from_expr(x), ring.zero), expanded[1])
        assert not verify_rational_basis(
            original,
            original,
            ring,
            membership_witnesses=forged_expanded,
            rational_normal_form_backend=backend,
        )
