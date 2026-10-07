"""High-level solving API, advanced tools live in public submodules."""

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version

from .algebraic import AlgebraicSystemRoots, algsolve
from .errors import (
    NotZeroDimensionalError,
    PolynomialSystemError,
    PolynomialSystemInputError,
    SystemSolveLimitError,
)
from .recovery import HomotopyRecoveryOptions
from .solver import CompletenessEvidence, PolynomialSystemRoots, polysolve

try:
    __version__ = _version("algroots")
except _PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "polysolve",
    "algsolve",
    "PolynomialSystemRoots",
    "AlgebraicSystemRoots",
    "CompletenessEvidence",
    "HomotopyRecoveryOptions",
    "PolynomialSystemError",
    "PolynomialSystemInputError",
    "NotZeroDimensionalError",
    "SystemSolveLimitError",
    "__version__",
]
