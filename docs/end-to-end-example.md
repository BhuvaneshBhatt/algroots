# End-to-End Worked Example

This example follows a system from its original algebraic form through algebraization, exact quotient structure, numerical extraction, branch filtering, and default best-effort exact recognition.

Consider:

$$
\sqrt{x+y}=x-1,
$$

$$
x^2+y^2=5.
$$

## 1. Original problem

<!-- algroots: execute -->
```python
import sympy as sp
from algroots import (
    algsolve,
    algebraize_system,
    recognize_system_roots,
)

x, y = sp.symbols("x y")

equations = [
    sp.sqrt(x + y) - (x - 1),
    x**2 + y**2 - 5,
]
```

The first equation is algebraic but not polynomial because of the square root.

## 2. Inspect algebraization

<!-- algroots: execute -->
```python
algebraized = algebraize_system(equations, (x, y))

print(algebraized.original_equations)
print(algebraized.polynomial_equations)
print(algebraized.auxiliary_variables)
print(algebraized.nonzero_constraints)
```

Conceptually, introduce an auxiliary variable $u$ representing:

$$
u=\sqrt{x+y}.
$$

The polynomial relation is:

$$
u^2=x+y,
$$

and the original equation also relates:

$$
u=x-1.
$$

Together with:

$$
x^2+y^2=5,
$$

this gives a zero-dimensional augmented polynomial system.

The exact generated symbol names are implementation details; use the fields on `AlgebraizedSystem` rather than depending on a particular auxiliary-variable name.

## 3. Why the original equation must be retained

The polynomial equation

$$
u^2=x+y
$$

does not by itself encode the principal square-root condition. It admits both signs of $u$.

Therefore the augmented polynomial system is a candidate-generating representation, not the final semantic definition of the problem.

`AlgebraizedSystem.original_equations` retains the expressions that must ultimately be satisfied.

## 4. Exact structural solve

Now call:

<!-- algroots: execute -->
```python
result = algsolve(
    equations,
    (x, y),
    digits=70,
    max_precision_digits=280,
    method="auto",
)
```

Internally the augmented polynomial equations are passed through exact polynomial preprocessing:

```text
augmented polynomial equations
        ↓
exact Gröbner basis
        ↓
leading monomials
        ↓
standard-monomial staircase
        ↓
finite quotient dimension D
```

For a zero-dimensional ideal:

$$
A=K[x,y,u]/I
$$

is finite-dimensional.

This exact stage establishes the finite algebraic structure before the expensive root extraction becomes numerical.

## 5. Backend selection

With `method="auto"`, `algroots` can use a favorable shape representation or the action-matrix backend and can fall back when a specialized representation is unsuitable.

In the action path, choose a separating linear form:

$$
L=c_xx+c_yy+c_uu.
$$

The solver constructs the multiplication matrix:

$$
M_L.
$$

It does **not** need separate dense matrices $M_x$, $M_y$, and $M_u$.

## 6. Arbitrary-precision numerical extraction

The exact entries needed for the separator action are converted to arbitrary-precision FLINT/Arb complex-ball arithmetic.

For a simple separating eigensystem:

$$
M_L^T v_j=\lambda_j v_j.
$$

The eigenvectors encode evaluation of quotient-basis elements at the corresponding roots.

Coordinate values are recovered from the exact normal forms:

$$
\operatorname{NF}(x),\qquad
\operatorname{NF}(y),\qquad
\operatorname{NF}(u).
$$

If the eigensystem cannot be isolated or the resulting roots fail verification, the action path can retry at higher precision up to `max_precision_digits`.

## 7. Numerical verification and refinement

Each projected candidate is evaluated in compiled numerical forms of the polynomial equations.

A scale-aware relative residual is used. Marginal candidates can be refined using a precompiled numerical Jacobian and Newton steps.

This stage answers:

> Does this numerical tuple accurately satisfy the polynomial equations?

It does not yet answer the branch question.

## 8. Projection and branch filtering

Auxiliary coordinates are removed and the candidate is projected back to:

$$
(x,y).
$$

Then `algroots` evaluates the **original** equation:

$$
\sqrt{x+y}-(x-1)
$$

using its principal branch.

Any polynomial candidate corresponding to the wrong square-root sign is rejected here.

This is the key distinction between solving the algebraized polynomial system and solving the user's original algebraic system.

## 9. Inspect the result

<!-- algroots: execute -->
```python
print(result.roots)
print(result.method)
print(result.quotient_dimension)
print(result.working_digits)
print(result.max_relative_residual)
print(result.auxiliary_variables)
```

`result.roots` contains the distinct projected numerical roots that survived original-equation/domain validation.

`result.quotient_dimension`, when available, describes the augmented quotient algebra and should not automatically be interpreted as the number of projected distinct roots when multiplicity or projection collisions are present.

## 10. Default best-effort exact recognition

If exact algebraic forms are useful:

<!-- algroots: execute -->
```python
recognized = recognize_system_roots(
    result,
    max_degree=8,
    require_certified=False,
)

for root in recognized:
    print("numerical:", root.numerical_root)
    print("exact:", root.exact_coordinates)
    print("scalar certified:", root.scalar_certified)
    print("jointly certified:", root.jointly_certified)
    print("certified:", root.certified)
```

The recognition layer:

1. recognizes each numerical coordinate as a candidate algebraic number;
2. reconstructs a compatible exact SymPy root;
3. substitutes the exact tuple into the **original equations**;
4. records whether exact joint certification succeeds.

With:

```python
require_certified=True
```

failure to obtain the stronger scalar-plus-joint certification raises `ExactCertificationError`.

## 11. The architecture in one diagram

```text
original algebraic equations
        ↓
exact algebraization
        ↓
auxiliary polynomial relations
+ retained domain information
        ↓
exact Gröbner basis
        ↓
exact quotient dimension / staircase
        ↓
shape position OR separator M_L
        ↓
FLINT/Arb arbitrary-precision roots/eigensystem
        ↓
coordinate recovery
        ↓
compiled residual verification
        ↓
numerical Newton refinement if useful
        ↓
project away auxiliary variables
        ↓
evaluate ORIGINAL branch/domain semantics
        ↓
verified numerical roots
        ↓
default best-effort algrecognize
        ↓
exact coordinate reconstruction
        ↓
exact joint certification
```

This division of labor is central to `algroots`: exact algebra determines the finite solution structure, arbitrary-precision numerical algebra extracts roots efficiently, and exact reconstruction is a default best-effort post-processing stage rather than being carried through every numerical operation.
