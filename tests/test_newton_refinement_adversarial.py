import mpmath as mp
import sympy as sp

from algroots.numerical import (
    compile_polynomials,
    compiled_jacobian,
    compiled_residual,
    newton_refine,
)

x, y = sp.symbols("x y")


def test_newton_refines_well_conditioned_complex_root():
    equations = (x**2 + 1,)
    polynomials = compile_polynomials(equations, (x,))
    jacobian = compiled_jacobian(equations, (x,))
    start = (mp.mpc("0.1", "0.9"),)
    initial = compiled_residual(polynomials, start, 50)
    root, success = newton_refine(polynomials, jacobian, start, 50, 30)
    final = compiled_residual(polynomials, root, 50)
    assert success
    assert final < initial


def test_newton_handles_overdetermined_consistent_system():
    equations = (x + y - 1, x - y, 2 * x - 1)
    polynomials = compile_polynomials(equations, (x, y))
    jacobian = compiled_jacobian(equations, (x, y))
    root, success = newton_refine(polynomials, jacobian, (mp.mpf("0.45"), mp.mpf("0.55")), 50, 30)
    assert success
    assert compiled_residual(polynomials, root, 50) < mp.mpf("1e-30")


def test_newton_does_not_claim_easy_success_at_singular_multiple_root():
    equations = (x**2,)
    polynomials = compile_polynomials(equations, (x,))
    jacobian = compiled_jacobian(equations, (x,))
    start = (mp.mpf("1e-10"),)
    root, _ = newton_refine(polynomials, jacobian, start, 50, 8)
    assert compiled_residual(polynomials, root, 50) <= compiled_residual(polynomials, start, 50)
