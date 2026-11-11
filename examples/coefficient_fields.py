"""Discover coefficient-field kernels and preserve exceptional parameter fibers."""

import sympy as sp

from algroots import groebner_backends, groebner_basis, parametric_groebner_basis


def main():
    x, t = sp.symbols("x t")
    native = groebner_backends()["native"]
    backend = "native" if native.available else "sympy"
    proof = parametric_groebner_basis([t * x], (x,), (t,), backend=backend)
    assert proof.verify()
    assert list(proof.specialize({t: 0})) == []
    assert list(proof.specialize({t: 1})) == [x]
    if native.available and "number fields" in native.domains:
        field = sp.QQ.algebraic_field(sp.sqrt(2)).frac_field(t)
        basis = groebner_basis([(t + sp.sqrt(2)) * x - 1], (x,), domain=field, backend="native")
        assert basis.reduce((t + sp.sqrt(2)) * x - 1)[1] == 0
    print("Generic basis:", proof.basis)
    print("Exceptional condition:", proof.exceptional_condition)


if __name__ == "__main__":
    main()
