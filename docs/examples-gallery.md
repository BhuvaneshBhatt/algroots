# Examples Gallery

These examples emphasize systems that exercise different parts of `algroots`, not just isolated textbook polynomials.

## 1. Intersecting two conics

```python
import sympy as sp
from algroots import polysolve

x, y = sp.symbols("x y")

result = polysolve(
    [
        x**2 + y**2 - 1,
        x*y - sp.Rational(1, 4),
    ],
    (x, y),
    digits=60,
)
```

This is a finite coupled polynomial system. It is useful for comparing the shape and action backends.

## 2. Sphere, plane, and quadratic constraint

```python
x, y, z = sp.symbols("x y z")

result = polysolve(
    [
        x**2 + y**2 + z**2 - 1,
        x + y + z,
        x**2 - y**2 - sp.Rational(1, 5),
    ],
    (x, y, z),
    digits=60,
)
```

This illustrates a genuinely multivariate zero-dimensional intersection.

## 3. Exact algebraic coefficients

```python
x, y = sp.symbols("x y")

result = polysolve(
    [
        x - sp.sqrt(2)*y,
        y**2 - 3,
    ],
    (x, y),
    digits=70,
)
```

`sp.sqrt(2)` is an exact algebraic coefficient; it is not treated as a variable-dependent radical requiring an auxiliary variable.

## 4. Rational equation with a removable-looking pole

```python
x = sp.symbols("x")

expr = sp.Mul(
    x**2 - 1,
    sp.Pow(x - 1, -1, evaluate=False),
    evaluate=False,
)

result = algsolve([expr], (x,))
```

The polynomial numerator vanishes at $x=1$, but the original expression is undefined there. The pole is excluded.

## 5. Principal square-root branch

```python
x = sp.symbols("x")

result = algsolve(
    [sp.sqrt(x) - (x - 2)],
    (x,),
    digits=60,
)
```

Polynomialization produces an extra candidate. Original-expression validation removes it.

## 6. Coupled radical system

```python
x, y = sp.symbols("x y")

result = algsolve(
    [
        sp.sqrt(x + y) - x,
        y - 2,
    ],
    (x, y),
    digits=60,
)
```

The radical is algebraized with an auxiliary relation, then projected candidates are checked against the original principal branch.

## 7. Rational power

```python
x = sp.symbols("x")

result = algsolve(
    [x**sp.Rational(2, 3) - 4],
    (x,),
    digits=60,
)
```

This demonstrates why rational-power polynomialization and principal-branch filtering are separate steps.

## 8. Nearly colliding roots

```python
x = sp.symbols("x")
eps = sp.Rational(1, 10**30)

result = polysolve(
    [(x - eps)*(x + eps)],
    (x,),
    digits=60,
    max_precision_digits=240,
)
```

Compare `result.working_digits` with an easy system such as `x**2 - 1`. See [Precision and Conditioning](precision-and-conditioning.md).

## 9. Backend comparison

```python
system = [
    x**2 + y**2 - 1,
    x*y - sp.Rational(1, 4),
]

shape = polysolve(system, (x, y), method="shape")
action = polysolve(system, (x, y), method="action")
```

When both backends succeed, compare the roots as unordered numerical sets.

## 10. Exact recognition after numerical solving

```python
from algroots import recognize_system_roots

numerical = polysolve(
    [x - y, y**2 - 2],
    (x, y),
    digits=80,
)

recognized = recognize_system_roots(
    numerical,
    max_degree=2,
    require_certified=True,
)

for root in recognized:
    print(root.exact_coordinates)
    print(root.certified)
```

Recognition occurs **after** all-roots numerical solving. It does not replace completeness reasoning.

## 11. Unsupported positive-dimensional system

```python
x, y = sp.symbols("x y")

polysolve([x + y], (x, y))
```

The solution is a line, so an all-distinct-roots list is not the right representation. `algroots` rejects positive-dimensional systems.

## 12. Unsupported transcendental system

```python
algsolve([sp.sin(x) - x], (x,))
```

Generic transcendental equations are outside the algebraic quotient model.
