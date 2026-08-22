"""Numerical and exact roots of zero-dimensional algebraic equation systems."""

from importlib.metadata import PackageNotFoundError, version

from .algebraic import AlgebraicSystemRoots, algsolve
from .algebraization import AlgebraizedSystem, algebraize_system
from .border_basis import (
    BorderBasisDiagnostics,
    BorderBasisError,
    BorderBasisResult,
    compute_border_basis,
    compute_border_basis_linear,
)
from .continuation import (
    HomotopySystem,
    PathResult,
    PathStep,
    PathStepError,
    PathTrackerOptions,
    PathTrackingError,
    SympyHomotopy,
    track_path,
)
from .errors import (
    ActionMatrixError,
    HomotopySolveError,
    NotZeroDimensionalError,
    NumericalRootError,
    PolynomialSystemError,
    PolynomialSystemInputError,
    ShapePositionError,
    SystemSolveLimitError,
    TriangularSolveError,
)
from .monodromy import (
    MonodromyLoop,
    MonodromyOrbitResult,
    MonodromyPermutation,
    MonodromyRootInfo,
    closed_additive_loop,
    discover_monodromy_orbit,
    monodromy_permutation,
    track_loop,
)
from .monodromy_stopping import (
    CaptureRecaptureEstimate,
    capture_recapture_estimate,
    second_order_trace_test,
)
from .rational_univariate import (
    RationalUnivariateError,
    RationalUnivariatePoint,
    RationalUnivariateRepresentation,
    compute_rational_univariate_representation,
    solve_rur_points,
    solve_rur_representation,
    solve_zero_dimensional_system_with_rur,
)
from .recognition import (
    ExactCertificationError,
    RecognizedSystemRoot,
    recognize_system_roots,
)
from .solver import PolynomialSystemRoots, RootDiagnostics, polysolve

try:
    __version__ = version("algroots")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "HomotopySystem",
    "SympyHomotopy",
    "PathTrackerOptions",
    "PathStep",
    "PathResult",
    "PathTrackingError",
    "PathStepError",
    "track_path",
    "CaptureRecaptureEstimate",
    "capture_recapture_estimate",
    "second_order_trace_test",
    "MonodromyLoop",
    "MonodromyPermutation",
    "MonodromyOrbitResult",
    "MonodromyRootInfo",
    "closed_additive_loop",
    "track_loop",
    "monodromy_permutation",
    "discover_monodromy_orbit",
    "algsolve",
    "polysolve",
    "algebraize_system",
    "AlgebraicSystemRoots",
    "AlgebraizedSystem",
    "recognize_system_roots",
    "PolynomialSystemRoots",
    "RootDiagnostics",
    "RecognizedSystemRoot",
    "ExactCertificationError",
    "PolynomialSystemError",
    "PolynomialSystemInputError",
    "NotZeroDimensionalError",
    "ActionMatrixError",
    "HomotopySolveError",
    "ShapePositionError",
    "TriangularSolveError",
    "NumericalRootError",
    "SystemSolveLimitError",
    "RationalUnivariateError",
    "RationalUnivariateRepresentation",
    "RationalUnivariatePoint",
    "compute_rational_univariate_representation",
    "solve_zero_dimensional_system_with_rur",
    "solve_rur_representation",
    "solve_rur_points",
    "BorderBasisDiagnostics",
    "BorderBasisError",
    "BorderBasisResult",
    "compute_border_basis",
    "compute_border_basis_linear",
    "__version__",
]
