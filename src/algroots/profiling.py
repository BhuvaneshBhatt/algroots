"""Opt-in profiling of verified Gröbner calls inside larger exact algorithms."""

from contextvars import ContextVar
from functools import wraps
from time import perf_counter

_active_profiler = ContextVar("algroots_groebner_profiler", default=None)
_current_record = ContextVar("algroots_groebner_record", default=None)


class GroebnerProfiler:
    """Collect candidate, conversion, verification, and native image costs.

    Use as a context manager around basis, quotient, RUR, or adapter calls.
    ``records`` contains one dictionary for each public ``groebner_basis`` call;
    cache hits that perform no basis construction create no record. Native field
    records include their modular/interpolation/reconstruction statistics. Set
    ``collect_batch_metrics=True`` in native limits for full image profiles.
    Timers are nested and must not be added as independent costs. Supervised
    child processes have their own contexts and do not publish records here.
    """

    def __init__(self):
        self.records = []
        self._token = None

    def __enter__(self):
        if self._token is not None:
            raise RuntimeError("a GroebnerProfiler instance cannot be entered twice")
        self._token = _active_profiler.set(self)
        return self

    def __exit__(self, exc_type, exc, traceback):
        _active_profiler.reset(self._token)
        self._token = None


def current_record():
    """Return the active verified-basis profiling record, if profiling is enabled."""
    return _current_record.get()


def record_native_stats(stats):
    """Attach native image statistics to the current profiling record."""
    record = _current_record.get()
    if record is not None:
        record.setdefault("native", []).append(stats)


def profiled_groebner(function):
    """Instrument an exact API only when a caller has selected a profiling scope."""

    @wraps(function)
    def wrapped(*args, **kwargs):
        profiler = _active_profiler.get()
        if profiler is None:
            return function(*args, **kwargs)
        record = {
            "backend": kwargs.get("backend", "native"),
            "order": kwargs.get("order", "grevlex"),
            "domain": str(kwargs.get("domain")),
            "conversion_seconds": 0.0,
            "candidate_seconds": 0.0,
            "verification_seconds": 0.0,
            "output_seconds": 0.0,
            "completed": False,
        }
        profiler.records.append(record)
        token = _current_record.set(record)
        started = perf_counter()
        try:
            result = function(*args, **kwargs)
            record["basis_size"] = len(result.polys)
            record["completed"] = True
            return result
        finally:
            record["seconds"] = perf_counter() - started
            _current_record.reset(token)

    return wrapped
