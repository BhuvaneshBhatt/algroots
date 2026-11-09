"""Failures shared by the Gröbner API and its optional kernels."""


class GroebnerLimitError(RuntimeError):
    """A kernel exhausted a budget without producing a complete basis."""


class GroebnerVerificationError(ArithmeticError):
    """An accelerated basis disagreed with an independent exact calculation."""


class GroebnerBackendUnavailable(ImportError):
    """The selected native kernel is not installed."""
