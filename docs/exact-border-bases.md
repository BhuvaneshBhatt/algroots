# Exact Border Bases

`algroots` exposes exact border-basis representations for rational-coefficient zero-dimensional polynomial ideals. This is a symbolic quotient-algebra tool; it is **not** the numerical AVI/SVD approximate-border algorithm used for noisy coefficients.

## Order ideals and borders

An order ideal $\mathcal O$ is a divisor-closed finite set of monomials. Its border is

$$
\partial\mathcal O
=\{x_i m : m\in\mathcal O\}\setminus\mathcal O.
$$

For each border monomial $b$, a border basis supplies an exact relation

$$
b-\sum_{m\in\mathcal O} c_m m=0
$$

in the quotient algebra. These relations determine exact normal forms and multiplication operators.

<!-- algroots: execute -->
```python
import sympy as sp
from algroots import compute_border_basis

x, y = sp.symbols("x y")
border = compute_border_basis((x**2 - 1, y - x), (x, y))

assert border.dimension == 2
assert border.has_commuting_multiplication_matrices()
```

## Multiplication matrices

For each coordinate $x_i$, `BorderBasisResult.multiplication_matrices[x_i]` represents multiplication by $x_i$ on the finite quotient algebra. Exact quotient multiplication must commute:

$$
M_{x_i}M_{x_j}=M_{x_j}M_{x_i}.
$$

`commutation_residuals()` returns the exact pairwise commutators and `has_commuting_multiplication_matrices()` checks that each is identically zero.

The result also supports:

- exact normal forms with `normal_form(expression)`;
- quotient coordinates with `coordinates(expression)`;
- multiplication by an arbitrary quotient element with `multiplication_matrix(expression)`;
- structural diagnostics including quotient-basis rank and border rank.

## Construction algorithms

`compute_border_basis(..., algorithm="groebner")` derives the quotient representation from a supporting Gröbner basis. `compute_border_basis_linear()` / the linear-Macaulay algorithm obtains border relations through exact coefficient linear algebra while retaining a Gröbner basis for zero-dimensionality and certification.

The current coefficient-domain contract is narrower than RUR: exact border-basis construction is presently rational-coefficient (`QQ`) infrastructure. Native algebraic-number-field support may be added separately.

## Relation to the action backend

Border bases and the action/endomorphism-matrix backend represent the same finite quotient algebra in different bases. Either representation yields multiplication operators. This makes exact border bases useful both as a public algebraic object and as an independent structural oracle for multiplication-matrix code.

## Noisy systems

The current implementation should not be used as an approximate border basis for noisy/inexact polynomial coefficients. A future AVI/SVD-style backend would need numerical rank decisions, tolerance selection, near-commutation diagnostics, and approximate joint diagonalization/root refinement.
