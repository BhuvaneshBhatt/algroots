# Gröbner bases

Use `algroots.groebner_basis` to reduce polynomial equations to a canonical basis, test ideal membership, eliminate variables, or build a finite quotient algebra. It returns a SymPy `GroebnerBasis`, so familiar operations such as `.reduce`, `.fglm`, `.polys`, and `.is_zero_dimensional` remain available.

## Start with a mathematical result

With a native wheel installed, the default call is:

```python
import sympy as sp
from algroots import groebner_basis

x, y = sp.symbols("x y")
basis = groebner_basis([x**2 - y, y**2 - 1], [x, y])
```

For a portable installation, choose SymPy explicitly:


<!-- algroots: execute -->
```python
import sympy as sp
from algroots import groebner_basis

x, y = sp.symbols("x y")
basis = groebner_basis([x**2 - y, y**2 - 1], [x, y], backend="sympy")
assert basis.reduce(x**4 - 1)[1] == 0
assert basis.reduce(x)[1] == x
assert basis.is_zero_dimensional
```

Variables are supplied as an ordered sequence. The default monomial order is `grevlex`, which is useful for constructing a basis before solving or changing order. Choose `order="lex"` for elimination. Rational input uses QQ. An explicit `domain=sp.GF(101)` computes in characteristic 101; it changes the mathematical problem and is never used as a substitute for a rational answer.

## Choose a backend

| Backend | QQ | Finite fields | Number, parameter and algebraic function fields | Verification at this API |
| --- | --- | --- | --- | --- |
| `sympy` | Yes | Yes | Yes | Exact field algorithm |
| `native` (default) | Modular lifting | Direct compiled prime F4; FLINT for GF(p^d) | Rational functions and certified algebraic towers; FLINT extension for algebraic components | Exact reference by default; optional identity certificate |

`native` is the default and runs `f4_tiled_finish`. Install a native wheel to use it. A missing native kernel raises `GroebnerBackendUnavailable` with installation guidance. Use `backend="sympy"` explicitly for the portable implementation; selecting native never changes to a different algorithm after an error.

```python
import sympy as sp
from algroots import NativeF4Limits, groebner_basis

x, y = sp.symbols("x y")
basis = groebner_basis(
    [x**2 - y, y**2 - 1], [x, y], domain=sp.GF(101),
    backend="native", limits=NativeF4Limits(max_pairs=20000),
)
```

## Solvers and finite algebras

Native F4 is the default throughout algroots basis construction: `polysolve`, `algsolve`, quotient algebras, RURs, supporting bases for border bases, finite-primary components, parameter families and certificate constructors. Solver and quotient interfaces use `groebner_backend="native"` and `groebner_limits`; direct basis and certificate interfaces use `backend="native"` and `limits`. Select `"sympy"` explicitly for portable construction. Presolve may finish without constructing a basis. Homotopy needs no Gröbner basis: either valid backend name is accepted, and Gröbner limits are rejected.

<!-- algroots: execute -->
```python
import sympy as sp
from algroots import QuotientAlgebra, polysolve

x, y = sp.symbols("x y")
system = [x**2 - y, y**2 - 1]
algebra = QuotientAlgebra.from_polynomials(system, [x, y], groebner_backend="sympy")
assert algebra.dimension == 4
assert algebra.normal_form(x**4 - 1) == 0
roots = polysolve(system, [x, y], groebner_backend="sympy", recognize=False)
assert len(roots.roots) == 4
```

The examples above select the portable fallback explicitly so they also run without native binaries. Omitting that selection uses native F4. Install the native wheel for the default solver and field-construction workflows; installing the pure Python wheel alone does not change defaults. Independent exact verification continues to use SymPy regardless of the construction backend.

## Verification and time measurements

By default, an accelerated result is compared coefficient by coefficient with an independently computed exact reduced monic basis in the requested field and order. Uniqueness of a reduced monic basis proves equality of ideals, including both inclusions. Checking only that input generators reduce to zero and that S-pairs vanish would accept an incorrect larger ideal such as the unit ideal.

Reference verification is included in public API and solver time when selected. `verification="certificate"` instead proves both ideal inclusions and Buchberger identities. `certificate=proof.prepare()` reuses a detached verified snapshot; its constructor checks the proof once, and repeated uses avoid generic replay. See [Reusable certificates](sparse-quotients-and-certificates.md). The standalone expert kernels can compute unverified candidates for research, but those results do not bypass the verified construction path. Kernel-only timing is therefore different from end-to-end verified time. The benchmark `benchmarks/bench_cross_cas.py` records kernel computation, worker wall time and independent verification separately on identical exact inputs. Failed checks and unavailable executables are retained in its output.

## Budgets and errors

The native public backend verifies QQ results with SymPy and does not require alggeom. The expert `native_modular_basis(..., certify=True)` interface uses the optional alggeom witness verifier.

`NativeF4Limits` controls native operation and reconstruction budgets. SymPy does not accept native limits. Integer budgets reject Boolean and nonintegral values. Passing a limit object to the wrong backend is an error. `GroebnerLimitError` is the shared catch point for budget exhaustion; `GroebnerVerificationError` signals a rejected result. Optional native kernels raise `GroebnerBackendUnavailable` when absent. No incomplete basis is returned and explicit selections do not fall back to another backend.

Limits cover kernel operations and reconstruction, not a global wall-clock deadline. The independent exact reference is not bounded by native limits. Use `execution=ExecutionLimits(timeout=...)` to supervise the complete basis calculation, including its reference verification. See [Supervised execution](supervised-execution.md) for process-transfer requirements and platform-specific memory support.

## Limits and next opportunities

The native engine supports prime fields with modulus at most 1000000007, up to 64 coordinates, and lex, grlex, or grevlex orders. Number and parameter fields use modular reconstruction, with FLINT finite-extension arithmetic for algebraic components. Finite separable characteristic-zero algebraic function fields and towers are supported through `AlgebraicFunctionField`. Finite extension fields GF(p^k) use the FLINT extension kernel with the supplied irreducible modulus. Positive-characteristic parameter fields require explicit SymPy construction. Block orders, strong bases over ZZ, comprehensive bases and a Gröbner walk are not implemented. Solving requires a zero-dimensional characteristic-zero ideal; Gröbner construction also supports positive-dimensional ideals and prime fields. Verification can dominate time. Identity-certificate verification can replace reference computation; producing witnesses from native row provenance remains a possible performance improvement.

The alggeom ideal and coordinate-ring factories default to native construction, including their cached bases. Use `groebner_backend="sympy"` when creating an ideal to select portable construction. `Ideal.groebner_basis_from()` also defaults to native, and prime fields retain their characteristic. The semialg construction adapter defaults to native and translates its extension and modulus options to exact coefficient fields. Semialg RUR, parameter-equality preprocessing and primary-decomposition entry points also default to native. Finite fields cannot replace ordered real coefficient fields in inequality solving.

## Check engine availability

<!-- algroots: execute -->
```python
from algroots import NativeF4Limits, groebner_backends

engines = groebner_backends()
assert engines["sympy"].available
assert engines["native"].limits_type is NativeF4Limits
assert "prime fields" in engines["native"].domains
if not engines["native"].available:
    print(engines["native"].installation)
```

`groebner_backends()` lists exactly `native` and `sympy`. Discovery reports availability without running a basis calculation or changing the default. An installed engine can still reject an unsupported domain or exceed its operation budget.

| Your task | Starting choice | Cost or requirement |
| --- | --- | --- |
| Ordinary exact basis, solver, quotient or parameter field | `native` (default) | Compiled F4; algebraic components also require the FLINT extension |
| Portable computation or an unsupported native field | explicit `sympy` | No compiled algroots kernel required |
| Try compiled F4 over QQ or a prime field | `native` (default) | Install the native wheel; public time includes the selected exact verification |
| Need an overall calculation deadline | Any supported engine with `execution=ExecutionLimits(...)` | Spawned worker; see [Supervised execution](supervised-execution.md) |

## Test ideal membership

A polynomial belongs to the input ideal precisely when its remainder is zero. `.reduce` returns division quotients and a remainder; inspect the remainder for membership.

<!-- algroots: execute -->
```python
import sympy as sp
from algroots import groebner_basis

x, y = sp.symbols("x y")
basis = groebner_basis([x**2 - y, y**2 - 1], [x, y], backend="sympy")
assert basis.reduce(x**4 - 1)[1] == 0
assert basis.reduce(x**2)[1] == y
assert basis.reduce(x)[1] == x
```

## Eliminate a variable

Lexicographic order eliminates earlier coordinates. With `[x, y]`, basis polynomials containing only `y` generate the elimination ideal. This works for positive-dimensional ideals too; numerical root solvers have a narrower finite-solution contract.

<!-- algroots: execute -->
```python
import sympy as sp
from algroots import groebner_basis

x, y = sp.symbols("x y")
basis = groebner_basis([x + y - 3, x*y - 2], [x, y], order="lex", backend="sympy")
assert list(basis) == [x + y - 3, y**2 - 3*y + 2]
eliminated = tuple(p.as_expr() for p in basis.polys if not p.as_expr().has(x))
assert eliminated == (y**2 - 3*y + 2,)
```

## Change order for a finite ideal

Compute in grevlex first, then use FGLM to obtain a lexicographic basis when the ideal is zero-dimensional. FGLM does not apply to a positive-dimensional ideal.

<!-- algroots: execute -->
```python
import sympy as sp
from algroots import groebner_basis

x, y = sp.symbols("x y")
graded = groebner_basis([x + y - 3, x*y - 2], [x, y], backend="sympy")
assert graded.is_zero_dimensional
lexicographic = graded.fglm("lex")
assert list(lexicographic) == [x + y - 3, y**2 - 3*y + 2]
assert lexicographic == groebner_basis([x + y - 3, x*y - 2], [x, y], order="lex", backend="sympy")
```

For native number fields, mixed fields, exact specialization guards, installation requirements, and exceptional fibers, see [Coefficient fields and specialization](coefficient-fields.md).

See [Sparse quotients and reusable certificates](sparse-quotients-and-certificates.md) for sparse exact arithmetic, proof replay, and RUR specialization guards, and [Algebraic function fields](algebraic-function-fields.md) for parameter-dependent coefficient towers.
