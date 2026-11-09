"""Self-contained *exact* transcriptions of application polynomial systems.

The equations are transcribed from the individual Macaulay2/M2
ExampleSystems/*.m2 files at Macaulay2/M2 commit
a5ff8d7ebc500fa958f227ffdfa28a0eb922a51e (2026-10-22).
Decimal source literals are interpreted as rational *decimal values*, not
binary floats, and each equation is scaled by the least common multiple of
its coefficient denominators. Reduction modulo a prime dividing any scale
factor must be rejected to preserve the original characteristic-zero ideal.

No SymPy, FLINT or C++ extension is needed to build this catalog.
"""

from __future__ import annotations

import ast
from fractions import Fraction
from functools import cache
from math import lcm

PUBLISHED_SOURCE_ROOT = (
    "https://github.com/Macaulay2/M2/blob/a5ff8d7ebc500fa958f227ffdfa28a0eb922a51e/"
    "M2/Macaulay2/packages/ExampleSystems/"
)

# Each expression is a literal transcription of the source polynomial, with
# Polynomial exponent notation is normalized to Python for AST parsing.
# Variables follow the corresponding Macaulay2 ring declaration exactly.
SYSTEMS = {
    "m2_lorentz": (
        ("x1", "x2", "x3", "x4"),
        "lorentz.m2",
        (
            "x1*x2-x1*x3-x4+1",
            "x2*x3-x2*x4-x1+1",
            "-x1*x3+x3*x4-x2+1",
            "x1*x4-x2*x4-x3+1",
        ),
    ),
    "m2_reimer5": (
        ("x", "y", "z", "t", "u"),
        "reimer5.m2",
        (
            "-1+2*x**2-2*y**2+2*z**2-2*t**2+2*u**2",
            "-1+2*x**3-2*y**3+2*z**3-2*t**3+2*u**3",
            "-1+2*x**4-2*y**4+2*z**4-2*t**4+2*u**4",
            "-1+2*x**5-2*y**5+2*z**5-2*t**5+2*u**5",
            "-1+2*x**6-2*y**6+2*z**6-2*t**6+2*u**6",
        ),
    ),
    "m2_boon_rational": (
        ("s1", "s2", "g1", "g2", "C1", "C2"),
        "boon.m2",
        (
            "s1**2+g1**2-1",
            "s2**2+g2**2-1",
            "C1*g1**3+C2*g2**3-6/5",
            "C1*s1**3+C2*s2**3-6/5",
            "C1*g1**2*s1+C2*g2**2*s2-7/10",
            "C1*g1*s1**2+C2*g2*s2**2-7/10",
        ),
    ),
    "m2_puma_decimal": (
        ("x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8"),
        "puma.m2",
        (
            "x1**2+x2**2-1",
            "x3**2+x4**2-1",
            "x5**2+x6**2-1",
            "x7**2+x8**2-1",
            "0.004731*x1*x3-0.3578*x2*x3-0.1238*x1-0.001637*x2-0.9338*x4+x7-0.3571",
            "0.2238*x1*x3+0.7623*x2*x3+0.2638*x1-0.07745*x2-0.6734*x4-0.6022",
            "x6*x8+0.3578*x1+0.004731*x2",
            "-0.7623*x1+0.2238*x2+0.3461",
        ),
    ),
    "m2_chemical_decimal": (
        ("y1", "y2", "y3", "y4", "y5"),
        "chemequ.m2",
        (
            "y1*y2+y1-3*y5",
            "2*y1*y2+y1+1.9230e-6*y2**2+y2*y3**2+5.4518e-4*y2*y3+3.4074e-5*y2*y4+4.4975e-7*y2-10*y5",
            "2*y2*y3**2+5.4518e-4*y2*y3+3.8600e-1*y3**2+4.1062e-4*y3-8*y5",
            "3.4074e-5*y2*y4+2*y4**2-40*y5",
            "y1*y2+y1+9.6150e-7*y2**2+y2*y3**2+5.4518e-4*y2*y3+3.4074e-5*y2*y4+4.4975e-7*y2+1.930e-1*y3**2+4.1062e-4*y3+y4**2-1",
        ),
    ),
    "m2_rose_rational": (
        ("x", "y", "z"),
        "rose.m2",
        (
            "y**4-20/7*x**2",
            "x**2*z**4+7/10*x*z**4+7/48*z**4-50/27*x**2-35/27*x-49/216",
            "3/5*x**6*y**2*z+x**5*y**3+3/7*x**5*y**2*z+7/5*x**4*y**3-7/20*x**4*y*z**2-3/20*x**4*z**3+609/1000*x**3*y**3+63/200*x**3*y**2*z-77/125*x**3*y*z**2-21/50*x**3*z**3+49/1250*x**2*y**3+147/2000*x**2*y**2*z-23863/60000*x**2*y*z**2-91/400*x**2*z**3-27391/800000*x*y**3+4137/800000*x*y**2*z-1078/9375*x*y*z**2-5887/200000*x*z**3-1029/160000*y**3-24353/1920000*y*z**2-343/128000*z**3",
        ),
    ),
}


def _sum_poly(a, b):
    result = a.copy()
    for m, coeff in b.items():
        result[m] = result.get(m, Fraction(0)) + coeff
        if not result[m]:
            del result[m]
    return result


def _product(a, b):
    result = {}
    for ma, ca in a.items():
        for mb, cb in b.items():
            m = tuple(x + y for x, y in zip(ma, mb, strict=False))
            result[m] = result.get(m, Fraction(0)) + ca * cb
            if not result[m]:
                del result[m]
    return result


def exact_polynomial(text: str, variables: tuple[str, ...]):
    """Parse restricted algebraic expressions with literal-exact decimals.

    AST whitelist prevents executable code and supports +,-,*,/,nonnegative
    integer powers. Division must be by a nonzero scalar constant.
    """
    tree = ast.parse(text.replace("^", "**"), mode="eval")
    n = len(variables)
    one = (0,) * n
    symbols = {
        name: {tuple(int(i == j) for i in range(n)): Fraction(1)}
        for j, name in enumerate(variables)
    }

    def rec(node):
        if isinstance(node, ast.Name):
            if node.id not in symbols:
                raise ValueError(f"Unknown symbol {node.id}")
            return symbols[node.id]
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            token = ast.get_source_segment(text.replace("^", "**"), node)
            return {one: Fraction(token)}
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            a = rec(node.operand)
            return a if isinstance(node.op, ast.UAdd) else {m: -c for m, c in a.items()}
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, (ast.Add, ast.Sub)):
                a, b = rec(node.left), rec(node.right)
                return _sum_poly(
                    a, b if isinstance(node.op, ast.Add) else {m: -c for m, c in b.items()}
                )
            if isinstance(node.op, ast.Mult):
                return _product(rec(node.left), rec(node.right))
            if isinstance(node.op, ast.Div):
                a, b = rec(node.left), rec(node.right)
                if len(b) != 1 or one not in b or b[one] == 0:
                    raise ValueError("Only scalar division is allowed")
                return {m: c / b[one] for m, c in a.items()}
            if isinstance(node.op, ast.Pow):
                if not isinstance(node.right, ast.Constant) or type(node.right.value) is not int:
                    raise ValueError("Powers must be nonnegative integer literals")
                exponent = node.right.value
                if not 0 <= exponent <= 32:
                    raise ValueError("Exponent outside supported range")
                b = rec(node.left)
                out = {one: Fraction(1)}
                for _ in range(exponent):
                    out = _product(out, b)
                return out
        raise ValueError(f"Unsupported algebraic expression: {ast.dump(node)}")

    return rec(tree.body)


@cache
def published_case(name: str):
    if name not in SYSTEMS:
        raise ValueError(f"Unknown published system {name}")
    vars_, filename, equations = SYSTEMS[name]
    polynomials, scales = [], []
    for equation in equations:
        p = exact_polynomial(equation, vars_)
        scale = lcm(*(coeff.denominator for coeff in p.values()))
        result = {m: int(c * scale) for m, c in p.items()}
        if not result:
            raise ValueError(f"Published generator unexpectedly zero: {name}")
        polynomials.append(result)
        scales.append(scale)
    return (len(vars_), tuple(polynomials), PUBLISHED_SOURCE_ROOT + filename, tuple(scales))


def exact_lift_primes(name: str):
    """Small prime divisors of equation-specific denominator-clearing scales."""
    if name not in SYSTEMS:
        return ()
    scales = published_case(name)[3]
    bad = set()
    for scale in scales:
        divisor = 2
        while divisor * divisor <= scale:
            if scale % divisor == 0:
                bad.add(divisor)
                while scale % divisor == 0:
                    scale //= divisor
            divisor += 1
        if scale > 1:
            bad.add(scale)
    return tuple(sorted(bad))
