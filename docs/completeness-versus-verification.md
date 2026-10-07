# Completeness versus Verification

A numerical root solver has two separate responsibilities:

1. return points that really satisfy the equations;
2. avoid silently missing other roots.

These are **soundness/verification** and **completeness** respectively.

## Why tiny residuals are not enough

Consider:

$$
x^4-5x^2+4=0.
$$

Its roots are:

$$
-2,\ -1,\ 1,\ 2.
$$

Suppose a hypothetical numerical routine returns only:

$$
-2,\ -1,\ 1.
$$

Every returned value has exactly zero residual. A verification loop such as

```python
for root in roots:
    assert abs(f(root)) < tolerance
```

passes perfectly.

But the answer is incomplete because $x=2$ was never returned.

No amount of residual checking on the three returned roots can reveal the missing fourth root.

## Structural information changes the problem

For a zero-dimensional polynomial ideal,

$$
A=K[x_1,\ldots,x_n]/I
$$

has finite dimension $D$.

For a radical ideal with simple points, $D$ equals the number of distinct complex solutions. That gives the numerical stage a structural target count.

The action-matrix pipeline can therefore reason approximately as follows:

```text
exact quotient dimension D
        ↓
separator matrix of size D × D
        ↓
D isolated simple quotient characters/eigenvectors
        ↓
D recovered coordinate tuples
        ↓
D tuples pass original-system verification
```

If one stage produces fewer objects than expected, tiny residuals on the survivors do not make the solve complete.

## Multiplicity caveat

The quotient dimension counts algebraic multiplicity.

For:

$$
x^2=0,
$$

$D=2$ but there is only one distinct geometric root. Therefore the simple rule

```text
number of distinct roots == quotient dimension
```

applies only when the relevant ideal/root structure is radical/simple.

This is why multiplicity and simple-eigensystem checks matter.

## Algebraic systems add another verification layer

For:

$$
\sqrt{x}=x-2,
$$

the algebraized polynomial system has an extraneous root $x=1$.

So there are two different completeness/soundness questions:

1. did the polynomial backend recover all candidates represented by the augmented polynomial system?
2. which projected candidates satisfy the original branch/domain semantics?

`algroots` uses the polynomial system to generate candidates but validates the final projected roots against the original equations.

## Exact recognition does not repair incompleteness

`recognize_system_roots` operates on roots already found numerically.

If three numerical roots are recognized and exactly certified, that proves those three tuples satisfy the equations exactly. It does **not** prove a fourth root was not missed during numerical extraction.

Exact recognition strengthens root identity and equation satisfaction; it is not a replacement for all-roots structure.

## Practical interpretation

Use:

- `max_relative_residual` and `diagnostics` to assess numerical verification;
- `completeness` and its basis/notes to inspect global evidence, together with `quotient_dimension` and `geometric_solution_count`;
- `RecognizedSystemRoot.certified` for optional exact certification of a returned tuple.

Do not treat any one of those as a synonym for all the others.

See [Guarantees and Result Semantics](guarantees-and-result-semantics.md).


Requested `root_certifications` prove individual endpoints and multiplicities;
they do not automatically promote global evidence. Bounded homotopy recovery
certifies finite-root accounting when distinct exact endpoint proofs equal the
exact geometric count. Numerical paths and infinity remain outside that proof.
See [Reading Results](reading-results.md).
