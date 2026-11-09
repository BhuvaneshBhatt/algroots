"""Reusable precision, resource, and recovery settings for the public solvers."""

from dataclasses import dataclass, field, fields, replace
from inspect import signature

from ._validation import validate_integer
from .execution import ExecutionLimits, run_supervised
from .recovery import HomotopyRecoveryOptions


@dataclass(frozen=True)
class PrecisionOptions:
    """Working precision and root-recognition settings shared by both solvers."""

    digits: int = 50
    verification_digits: int | None = None
    guard_digits: int = 10
    max_precision_digits: int | None = None
    recognize: bool = True
    recognition_max_degree: int = 8

    def __post_init__(self):
        validate_integer(self.digits, "digits", minimum=15)
        validate_integer(self.guard_digits, "guard_digits", minimum=0)
        if self.verification_digits is not None:
            validate_integer(self.verification_digits, "verification_digits")
            if self.verification_digits >= self.digits:
                raise ValueError("verification_digits must be smaller than digits")
        if self.max_precision_digits is not None:
            validate_integer(self.max_precision_digits, "max_precision_digits", minimum=self.digits)
        if not isinstance(self.recognize, bool):
            raise TypeError("recognize must be Boolean")
        validate_integer(self.recognition_max_degree, "recognition_max_degree")


@dataclass(frozen=True)
class SolverLimits:
    """Operation and output bounds, separate from an overall execution deadline."""

    maxsteps: int = 200
    max_solutions: int = 10000
    max_action_dimension: int = 256
    max_homotopy_paths: int = 10000
    max_returned_roots: int = 10000
    ordering_max_refinements: int = 64

    def __post_init__(self):
        for setting in fields(self):
            validate_integer(getattr(self, setting.name), setting.name)


@dataclass(frozen=True)
class SolverOptions:
    """Group advanced precision, resource, recovery, and worker settings.

    Supply this through ``options=`` to either solver. Flat keywords still
    work on their own. A nondefault flat setting that duplicates a grouped
    setting is rejected, so no advanced setting is overwritten unnoticed.
    Recovery requires ``polysolve(method='homotopy')``; algebraic branch
    projection does not expose homotopy recovery.
    """

    precision: PrecisionOptions = field(default_factory=PrecisionOptions)
    limits: SolverLimits = field(default_factory=SolverLimits)
    recovery: HomotopyRecoveryOptions | None = None
    execution: ExecutionLimits | None = None

    def __post_init__(self):
        for name, expected in (
            ("precision", PrecisionOptions),
            ("limits", SolverLimits),
            ("recovery", HomotopyRecoveryOptions),
            ("execution", ExecutionLimits),
        ):
            value = getattr(self, name)
            if value is None and name in ("recovery", "execution"):
                continue
            if not isinstance(value, expected):
                raise TypeError(f"{name} must be {expected.__name__}")


def _solver_settings(options, current):
    """Resolve grouped settings once, rejecting conflicting flat input."""
    if not isinstance(options, SolverOptions):
        raise TypeError("options must be SolverOptions")
    result = {}
    for group in (options.precision, options.limits):
        for setting in fields(group):
            if current[setting.name] != setting.default:
                raise ValueError(
                    f"options cannot be combined with nondefault {setting.name}; set it inside SolverOptions"
                )
            result[setting.name] = getattr(group, setting.name)
    return result


def _supervise_solver(function, equations, variables, arguments):
    """Forward only declared public parameters, excluding temporary solver locals."""
    options = arguments["options"]
    keywords = {
        name: arguments[name]
        for name in signature(function).parameters
        if name not in ("equations", "variables")
    }
    keywords["options"] = replace(options, execution=None)
    return run_supervised(function, equations, variables, execution=options.execution, **keywords)
