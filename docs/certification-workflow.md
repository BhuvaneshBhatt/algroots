# Certification workflow

Choose the evidence you need before selecting a proof API. Numerical residuals,
exact recognition, isolated-point proofs and global completeness answer different
questions. All certification failures are explicit; increasing a budget does not
weaken a proof requirement.

## Automatic numerical-root boxes

Use `polysolve(..., certify="auto")` for best-effort aligned attempts, or
`certify="required"` when failure should raise `RootCertificationError`.
Automatic boxes require rational coefficients and a globally finite ideal.
Adaptive radii are proposals; exact separator counts and coordinate enclosures
prove the box contains exactly the reported point.

<!-- algroots: execute -->
```python
import sympy as sp
from algroots import polysolve

x = sp.Symbol("x")
result = polysolve((x**2 - 2,), (x,), recognize=False, certify="auto")
for attempt in result.root_certifications:
    if attempt.certificate is None:
        print(attempt.status, attempt.error)
    else:
        assert attempt.status == "certified" and attempt.certificate.verify()
        print(attempt.certificate.point, attempt.box_attempts)
```

The controls are `certification_max_dimension`, `certification_max_refinements`
and `certification_max_box_attempts`. An insufficient dimension budget returns
`unavailable` under `auto`; an exhausted box attempt returns `failed`. Neither
is a certificate. See [Limits and failures](../examples/15_limits_and_failures.py).

## Exact points and user-supplied boxes

`certify_isolated_root(equations, variables, point)` accepts exact algebraic
coefficients and exact algebraic coordinates. It proves membership, isolation,
Jacobian rank and individual multiplicity. Floating-point coordinates are refused.
`certify_root_box` accepts rational coefficients and a `RationalComplexBox` whose
open rectangles have `(real_lo, real_hi, imag_lo, imag_hi)` rational endpoints.

<!-- algroots: execute -->
```python
import sympy as sp
from algroots.certification import RationalComplexBox, certify_root_box

x = sp.Symbol("x")
box = RationalComplexBox(((1, 2, -sp.Rational(1, 10), sp.Rational(1, 10)),))
certificate = certify_root_box((x**2 - 2,), (x,), box)
assert certificate.point[0].is_real is True
assert certificate.multiplicity == 1 and certificate.verify()
```

`verify()` replays from the original equations and detects altered certificate
fields. Replay has its own dimension/refinement limits. Persisting a printed
certificate is not a documented serialization format.

## Recognition and completeness

Recognition reconstructs coordinates and checks the tuple jointly against the
original equations. It does not discover missing roots or supply a path proof.
Ordinary per-root certificates do not automatically promote global completeness.
Bounded recovery promotes finite-root accounting only when distinct exact
certificates equal the exact finite quotient's geometric count.

See [Singular certification and deflation](singular-certification-and-charts.md),
[Reading Results](reading-results.md) and [Exact Recognition](exact-recognition.md).
