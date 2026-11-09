"""Run expensive exact calculations in a worker with a wall-clock deadline."""

import copyreg
import io
import math
import multiprocessing as mp
import os
import pickle
import signal
import sys
import threading
from contextlib import suppress
from dataclasses import dataclass
from time import monotonic

from ._validation import validate_integer


class ExecutionError(RuntimeError):
    """An isolated calculation failed to produce a complete transferable result."""


class ExecutionTimeout(ExecutionError, TimeoutError):
    """The worker exceeded its overall wall-clock deadline."""


@dataclass(frozen=True)
class ExecutionLimits:
    """A worker deadline in seconds and an optional Linux address-space cap in MiB.

    The deadline includes worker startup, calculation, verification, and result
    transfer. Process creation and serialization of input arguments happen in
    the calling process and cannot be preempted by this supervisor. Memory caps
    require Linux; unsupported platforms raise before launching a worker.
    """

    timeout: float
    memory_mib: int | None = None

    def __post_init__(self):
        if (
            isinstance(self.timeout, bool)
            or not isinstance(self.timeout, (int, float))
            or not math.isfinite(self.timeout)
            or self.timeout <= 0
        ):
            raise ValueError("timeout must be a positive finite number of seconds")
        if self.memory_mib is not None:
            validate_integer(self.memory_mib, "memory_mib")


def _restore_basis(polynomials, variables, domain, order):
    import sympy as sp
    from sympy.polys.polyoptions import build_options

    options = build_options(variables, {"domain": domain, "order": order, "polys": False})
    return sp.GroebnerBasis._new(list(polynomials), options)


def _reduce_basis(basis):
    return _restore_basis, (tuple(basis.polys), basis.gens, basis.domain, str(basis.order))


def _restore_poly(terms, variables, domain):
    import sympy as sp

    return sp.Poly.from_dict(terms, *variables, domain=domain)


def _reduce_poly(poly):
    return _restore_poly, (poly.as_dict(), poly.gens, poly.domain)


def _restore_prime_field(modulus, symmetric):
    import sympy as sp

    return sp.GF(modulus, symmetric=symmetric)


def _reduce_prime_field(domain):
    return _restore_prime_field, (domain.mod, domain.sym)


def _restore_fraction_field(ground, symbols, order):
    return ground.frac_field(*symbols, order=order)


def _reduce_fraction_field(domain):
    return _restore_fraction_field, (domain.domain, domain.symbols, domain.field.order)


def _restore_algebraic_field(ground, extensions, alias):
    return ground.algebraic_field(*extensions, alias=alias)


def _reduce_algebraic_field(domain):
    return _restore_algebraic_field, (domain.dom, domain.orig_ext, domain.ext.alias)


def _restore_quotient(variables, basis, domain, exponents, cache_size, equations):
    from .quotient import QuotientAlgebra

    return QuotientAlgebra(variables, basis, domain, exponents, cache_size, equations)


def _reduce_quotient(algebra):
    # Transfer the defining mathematical data; mutable local caches and native
    # converter closures belong to the process that computed them.
    return _restore_quotient, (
        algebra.variables,
        algebra.groebner_basis,
        algebra.domain,
        algebra.standard_exponents,
        algebra.normal_form_cache_size,
        algebra.defining_polynomials,
    )


def _serialize(value):
    """Preserve basis domains and orders without changing global pickle behavior."""
    import sympy as sp
    from sympy.polys.domains import AlgebraicField, FiniteField, FractionField

    from .quotient import QuotientAlgebra

    buffer = io.BytesIO()
    pickler = pickle.Pickler(buffer, protocol=pickle.HIGHEST_PROTOCOL)
    pickler.dispatch_table = {
        **copyreg.dispatch_table,
        sp.GroebnerBasis: _reduce_basis,
        sp.Poly: _reduce_poly,
        FiniteField: _reduce_prime_field,
        FractionField: _reduce_fraction_field,
        QuotientAlgebra: _reduce_quotient,
        AlgebraicField: _reduce_algebraic_field,
    }
    pickler.dump(value)
    return buffer.getvalue()


def _worker(connection, payload, limits):
    try:
        if os.name == "posix":
            os.setsid()
        if limits.memory_mib is not None:
            import resource

            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            requested = limits.memory_mib * 1024**2
            bounded = [
                value for value in (soft, hard, requested) if value != resource.RLIM_INFINITY
            ]
            resource.setrlimit(resource.RLIMIT_AS, (min(bounded), hard))
        function, args, kwargs = pickle.loads(payload)
        value = function(*args, **kwargs)
        try:
            response = _serialize((True, value))
        except (pickle.PickleError, TypeError, AttributeError) as error:
            raise ExecutionError(f"worker result must be pickleable: {error}") from error
        connection.send_bytes(response)
    except Exception as error:
        # This boundary accepts arbitrary user callables. Preserve their error
        # type when transferable, including failures during result pickling.
        with suppress(OSError, TypeError, AttributeError, MemoryError, pickle.PickleError):
            connection.send_bytes(_serialize((False, error)))
    finally:
        connection.close()


def _stop(process):
    if process.is_alive():
        if os.name == "posix":
            try:
                # setsid may not yet have run if startup exceeded the deadline.
                if os.getpgid(process.pid) == process.pid:
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            except ProcessLookupError:
                pass
        else:
            process.kill()
    process.join()


def run_supervised(function, /, *args, execution, **kwargs):
    """Return a callable's result or raise before accepting an incomplete result.

    Uses multiprocessing's spawn context, so functions, inputs, outputs and
    exceptions must be pickleable. Use module-level callables and a script's
    ``if __name__ == '__main__'`` guard. On POSIX, timeout terminates the worker
    process group, including external CAS children. On Windows it terminates
    the worker itself; descendant processes are not supervised.
    """
    if not isinstance(execution, ExecutionLimits):
        raise TypeError("execution must be ExecutionLimits")
    if not callable(function):
        raise TypeError("function must be callable")
    if execution.memory_mib is not None and not sys.platform.startswith("linux"):
        raise NotImplementedError(
            "memory_mib requires Linux RLIMIT_AS; use timeout alone on this platform"
        )
    try:
        payload = _serialize((function, args, kwargs))
    except (pickle.PickleError, TypeError, AttributeError) as error:
        raise TypeError(
            f"supervised functions and inputs must be pickleable; use a module-level callable and materialized inputs: {error}"
        ) from error
    context = mp.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(sender, payload, execution))
    deadline = monotonic() + execution.timeout
    packet = []
    ready = threading.Event()

    def receive():
        try:
            packet.append(pickle.loads(receiver.recv_bytes()))
        except Exception as error:
            # Deserialization can invoke constructors of arbitrary result types.
            packet.append(error)
        finally:
            ready.set()

    thread = threading.Thread(target=receive, daemon=True)
    started = False
    try:
        process.start()
        started = True
        sender.close()
        thread.start()
        if not ready.wait(max(0, deadline - monotonic())) or monotonic() > deadline:
            raise ExecutionTimeout(
                f"calculation exceeded {execution.timeout:g} seconds; no result was returned"
            )
        message = packet[0]
        if isinstance(message, Exception):
            process.join(max(0, deadline - monotonic()))
            raise ExecutionError(
                f"worker exited without a complete result (exit code {process.exitcode})"
            ) from message
        successful, value = message
        if successful:
            return value
        raise value
    finally:
        if started:
            _stop(process)
        sender.close()
        receiver.close()
        if thread.ident is not None:
            thread.join(timeout=1)
        process.close()
