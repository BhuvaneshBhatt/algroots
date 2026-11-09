# Supervised execution

Use a worker deadline when a difficult exact calculation must return or stop within a bounded calculation window. Kernel operation limits alone do not cover independent reference verification. A supervised call returns the same mathematical result type as an ordinary call; it raises `ExecutionTimeout` when the deadline expires and never returns a partial basis or root set.

Run this as a Python script:

```python
import sympy as sp
from algroots import ExecutionLimits, ExecutionTimeout, groebner_basis

def main():
    x, y = sp.symbols("x y")
    try:
        basis = groebner_basis(
            [x**2 - y, y**2 - 1], [x, y], backend="sympy",
            execution=ExecutionLimits(timeout=20),
        )
    except ExecutionTimeout:
        print("The calculation exceeded its deadline.")
        return
    assert str(basis.order) == "grevlex"
    assert basis.reduce(x**4 - 1)[1] == 0

if __name__ == "__main__":
    main()
```

For a complete solver call, use `SolverOptions(execution=ExecutionLimits(timeout=30))`. For another public operation, use `run_supervised(QuotientAlgebra.from_polynomials, equations, variables, execution=ExecutionLimits(timeout=30))`. The same wrapper can run RUR construction or an adapter operation. Supply module-level callables and pickleable inputs. Lambdas, local functions, open files, and generators are unsuitable transfer inputs. Gröbner basis transfer preserves the coefficient domain and monomial order in both directions; it does not recompute the basis or bypass verification.

| Platform | Worker wall deadline | Memory cap | Timeout termination |
| --- | --- | --- | --- |
| Linux | Yes | `memory_mib` sets `RLIMIT_AS` | Worker process group, including CAS children |
| macOS | Yes | Unsupported; requesting it raises | Worker process group, including CAS children |
| Windows | Yes | Unsupported; requesting it raises | Direct worker; descendant processes are outside this supervisor |

The deadline clock includes worker startup, calculation, independent verification, and result transfer. The caller performs input serialization and starts the OS process synchronously; those operations cannot be preempted by this supervisor. Shutdown also requires OS scheduling. This is a calculation deadline, not a hard real-time guarantee for arbitrary input serialization or a stalled operating system. Regular calls do not create workers or pay serialization overhead.

On Linux, `ExecutionLimits(timeout=30, memory_mib=512)` limits virtual address space, including the loaded interpreter and libraries. It is not an RSS quota. A small cap can prevent imports or result serialization; the failure may be `MemoryError` or `ExecutionError` without a transferable result. Existing stricter process limits are retained. Other platforms reject a memory cap before launching the worker.

The worker uses multiprocessing's spawn context. Use the `if __name__ == "__main__"` guard; launch a script instead of running a spawning call directly in an interactive notebook. Transferable calculation exceptions retain their types. `ExecutionError` reports a worker or transport failure, and its subclass `ExecutionTimeout` is also a `TimeoutError`. On POSIX a timed-out worker group is killed; side effects already performed by a user-supplied callable are not rolled back.

The executable [supervision example](../examples/26_solver_settings.py) and the [Gröbner guide](groebner-bases.md) show these controls alongside ordinary exact results.
