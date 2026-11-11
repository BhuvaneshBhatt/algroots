"""Coordinate the global mpmath and FLINT precision contexts used by algroots."""

from contextlib import contextmanager
from threading import RLock

import mpmath as mp

precision_lock = RLock()


@contextmanager
def working_precision(digits):
    """Set mpmath precision without interleaving another algroots precision scope.

    The same reentrant lock protects FLINT precision, so callbacks and nested
    numerical helpers can change either context and restore it in stack order.
    Native exact arithmetic does not use this lock.
    """
    with precision_lock, mp.workdps(digits):
        yield
