"""Public exception types for algroots solvers."""


class PolynomialSystemError(ValueError):
    """Base class for polynomial-system solver errors."""


class PolynomialSystemInputError(PolynomialSystemError):
    """The supplied equations or variables are outside the supported domain."""


class NotZeroDimensionalError(PolynomialSystemError):
    """The polynomial system does not have a zero-dimensional solution set."""


class ShapePositionError(PolynomialSystemError):
    """The lexicographic Gröbner basis is not in supported shape position."""


class ActionMatrixError(PolynomialSystemError):
    """The quotient-algebra action-matrix backend could not certify all roots."""


class TriangularSolveError(PolynomialSystemError):
    """The system could not be resolved by triangular back-substitution."""


class HomotopySolveError(PolynomialSystemError):
    """The total-degree homotopy backend could not track all required paths."""


class NumericalRootError(PolynomialSystemError):
    """Numerical root computation or residual verification failed."""


class SystemSolveLimitError(PolynomialSystemError):
    """A solver safety limit was exceeded."""
