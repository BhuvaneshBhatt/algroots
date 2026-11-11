"""High-level solving API, advanced tools live in public submodules."""

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version

from .algebraic import AlgebraicSystemRoots, algsolve
from .algebraic_function_fields import AlgebraicFunctionField, FieldSpecialization, TowerCompression
from .backends import GroebnerBackendInfo, groebner_backends
from .basis import groebner_basis
from .basis_certificates import GroebnerCertificate, certify_groebner_basis
from .errors import (
    NotZeroDimensionalError,
    PolynomialSystemError,
    PolynomialSystemInputError,
    SystemSolveLimitError,
)
from .execution import ExecutionError, ExecutionLimits, ExecutionTimeout, run_supervised
from .groebner_errors import (
    GroebnerBackendUnavailable,
    GroebnerLimitError,
    GroebnerVerificationError,
)
from .native_f4 import NativeF4Limits
from .options import PrecisionOptions, SolverLimits, SolverOptions
from .parametric_groebner import GroebnerSpecialization, parametric_groebner_basis
from .quotient import QuotientAlgebra
from .recovery import HomotopyRecoveryOptions
from .rur_certificates import (
    RationalUnivariateCertificate,
    certify_rational_univariate_representation,
)
from .solver import CompletenessEvidence, PolynomialSystemRoots, polysolve
from .verified_families import VerifiedGroebnerFamily, VerifiedRationalUnivariateFamily

try:
    __version__ = _version("algroots")
except _PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "AlgebraicFunctionField",
    "AlgebraicSystemRoots",
    "CompletenessEvidence",
    "ExecutionError",
    "ExecutionLimits",
    "ExecutionTimeout",
    "FieldSpecialization",
    "GroebnerBackendInfo",
    "GroebnerBackendUnavailable",
    "GroebnerCertificate",
    "GroebnerLimitError",
    "GroebnerSpecialization",
    "GroebnerVerificationError",
    "HomotopyRecoveryOptions",
    "NativeF4Limits",
    "NotZeroDimensionalError",
    "PolynomialSystemError",
    "PolynomialSystemInputError",
    "PolynomialSystemRoots",
    "PrecisionOptions",
    "QuotientAlgebra",
    "RationalUnivariateCertificate",
    "SolverLimits",
    "SolverOptions",
    "SystemSolveLimitError",
    "TowerCompression",
    "VerifiedGroebnerFamily",
    "VerifiedRationalUnivariateFamily",
    "__version__",
    "algsolve",
    "certify_groebner_basis",
    "certify_rational_univariate_representation",
    "groebner_backends",
    "groebner_basis",
    "parametric_groebner_basis",
    "polysolve",
    "run_supervised",
]
