# Quick Start

## Polynomial systems

<!-- algroots: execute -->
```python
import sympy as sp
from algroots import polysolve

x, y = sp.symbols("x y")

result = polysolve(
    [x - y**2, y**3 - 2],
    (x, y),
    digits=60,
)

print(result.method)
print(result.roots)
```

`algroots` finds all three complex solutions. The default `method="auto"` tries shape position, then action matrices, then triangular Gröbner back-substitution.

## Radical equations

<!-- algroots: execute -->
```python
from algroots import algsolve

result = algsolve(
    [sp.sqrt(x) - (x - 2)],
    (x,),
    digits=60,
)

assert len(result.roots) == 1
```

Squaring would also produce `x = 1`, but that value violates the original principal-square-root equation. `algroots` filters it out.

## Rational equations

<!-- algroots: execute -->
```python
result = algsolve(
    [(x + 1)/(x - 2) - 3],
    (x,),
)
```

Denominator constraints are retained exactly, and denominator-zero components are removed from the polynomialized ideal by saturation before numerical solving.

## Exact recognition

<!-- algroots: execute -->
```python
from algroots import recognize_system_roots

recognized = recognize_system_roots(
    result,
    max_degree=8,
    require_certified=True,
)
```

The solvers attempt this recognition automatically by default using a degree-8 search budget. Call `recognize_system_roots` explicitly when you need custom recognition bounds, or pass `recognize=False` to the solver to disable automatic recognition. Recognition delegates scalar relation finding to `algrecognize`, reconstructs exact algebraic coordinates, and verifies the tuple jointly against the original equations.
