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
    "VerifiedGroebnerFamily",
    "VerifiedRationalUnivariateFamily",
    "RationalUnivariateCertificate",
    "certify_rational_univariate_representation",
    "AlgebraicFunctionField",
    "FieldSpecialization",
    "TowerCompression",
    "GroebnerCertificate",
    "certify_groebner_basis",
    "parametric_groebner_basis",
    "GroebnerSpecialization",
    "groebner_backends",
    "GroebnerBackendInfo",
    "ExecutionLimits",
    "ExecutionError",
    "ExecutionTimeout",
    "run_supervised",
    "PrecisionOptions",
    "SolverLimits",
    "SolverOptions",
    "groebner_basis",
    "QuotientAlgebra",
    "NativeF4Limits",
    "GroebnerLimitError",
    "GroebnerVerificationError",
    "GroebnerBackendUnavailable",
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
