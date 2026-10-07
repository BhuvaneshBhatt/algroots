"""Total-degree homotopy construction and path orchestration."""

from __future__ import annotations

import itertools
import multiprocessing
import random
from collections.abc import Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

import sympy as sp

from .continuation import PathResult, PathTrackerOptions, SympyHomotopy, track_path
from .errors import HomotopySolveError, SystemSolveLimitError


@dataclass(frozen=True)
class TotalDegreeHomotopyOptions:
    """Validated orchestration options for total-degree path tracking."""

    max_paths: int = 10_000
    seed: int = 0
    parallel: bool = False
    max_workers: int | None = None
    gamma_attempts: int = 4

    def __post_init__(self) -> None:
        if not isinstance(self.max_paths, int) or self.max_paths <= 0:
            raise ValueError("max_homotopy_paths must be a positive integer")
        if not isinstance(self.seed, int):
            raise ValueError("homotopy_seed must be an integer")
        if not isinstance(self.parallel, bool):
            raise ValueError("homotopy_parallel must be a boolean")
        if self.max_workers is not None and (
            not isinstance(self.max_workers, int) or self.max_workers <= 0
        ):
            raise ValueError("homotopy_max_workers must be None or a positive integer")
        if not isinstance(self.gamma_attempts, int) or self.gamma_attempts <= 0:
            raise ValueError("homotopy_gamma_attempts must be a positive integer")


@dataclass(frozen=True)
class TotalDegreeHomotopy:
    """A total-degree start-to-target homotopy for a square polynomial system."""

    equations: tuple[Any, ...]
    variables: tuple[sp.Symbol, ...]
    degrees: tuple[int, ...]
    gamma: Any

    @property
    def path_count(self) -> int:
        count = 1
        for degree in self.degrees:
            count *= degree
        return count

    def _expressions(self, parameter: sp.Symbol) -> tuple[Any, ...]:
        return tuple(
            sp.expand((1 - parameter) * self.gamma * (variable**degree - 1) + parameter * equation)
            for equation, variable, degree in zip(
                self.equations, self.variables, self.degrees, strict=True
            )
        )

    def expressions(self) -> tuple[Any, ...]:
        parameter = sp.Symbol("_algroots_homotopy_t", real=True)
        return self._expressions(parameter)

    def system(self) -> SympyHomotopy:
        parameter = sp.Symbol("_algroots_homotopy_t", real=True)
        return SympyHomotopy(self._expressions(parameter), self.variables, parameter)

    def with_gamma(self, gamma: Any) -> TotalDegreeHomotopy:
        return TotalDegreeHomotopy(self.equations, self.variables, self.degrees, gamma)

    def iter_start_points(self) -> Iterator[tuple[Any, ...]]:
        """Yield total-degree start roots without materializing the Cartesian product."""
        root_sets = []
        for degree in self.degrees:
            roots = tuple(sp.exp(2 * sp.pi * sp.I * index / degree) for index in range(degree))
            root_sets.append(roots)
        yield from itertools.product(*root_sets)

    def start_points(self) -> tuple[tuple[Any, ...], ...]:
        """Return all start roots; prefer :meth:`iter_start_points` for large systems."""
        return tuple(self.iter_start_points())


def gamma_from_seed(seed: int) -> Any:
    """Return a deterministic exact complex gamma for the gamma trick."""
    rng = random.Random(seed)
    real = rng.randint(101, 997)
    imag = rng.randint(101, 997)
    if rng.randrange(2):
        real = -real
    if rng.randrange(2):
        imag = -imag
    scale = max(abs(real), abs(imag))
    return sp.Rational(real, scale) + sp.I * sp.Rational(imag, scale)


def gamma_candidates(seed: int, attempts: int) -> tuple[Any, ...]:
    """Return a deterministic sequence of distinct gamma-trick candidates."""
    candidates: list[Any] = []
    offset = 0
    while len(candidates) < attempts:
        gamma = gamma_from_seed(seed + offset)
        if gamma not in candidates:
            candidates.append(gamma)
        offset += 1
    return tuple(candidates)


def build_total_degree_homotopy(
    equations: Sequence[Any],
    variables: Sequence[sp.Symbol],
    *,
    seed: int = 0,
    max_paths: int = 10_000,
    gamma: Any | None = None,
) -> TotalDegreeHomotopy:
    """Build a total-degree homotopy without computing a Gröbner basis."""
    equations = tuple(sp.sympify(expr) for expr in equations)
    variables = tuple(variables)
    if not isinstance(seed, int):
        raise ValueError("homotopy_seed must be an integer")
    if not isinstance(max_paths, int) or max_paths <= 0:
        raise ValueError("max_homotopy_paths must be a positive integer")
    if not variables or len(equations) != len(variables):
        raise HomotopySolveError(
            "total-degree homotopy requires a nonempty square polynomial system"
        )

    degrees: list[int] = []
    for equation in equations:
        try:
            polynomial = sp.Poly(equation, *variables, extension=True)
        except (sp.PolynomialError, TypeError, ValueError) as exc:
            raise HomotopySolveError("total-degree homotopy requires polynomial equations") from exc
        if polynomial.is_zero:
            raise HomotopySolveError(
                "total-degree homotopy does not support an identically zero equation"
            )
        degree = int(polynomial.total_degree())
        degrees.append(max(0, degree))

    path_count = 1
    for degree in degrees:
        if degree == 0:
            path_count = 0
            break
        path_count *= degree
        if path_count > max_paths:
            raise SystemSolveLimitError(
                f"total-degree homotopy Bézout path count exceeds max_homotopy_paths={max_paths}"
            )

    return TotalDegreeHomotopy(
        equations=equations,
        variables=variables,
        degrees=tuple(degrees),
        gamma=gamma_from_seed(seed) if gamma is None else sp.sympify(gamma),
    )


_WORKER_SYSTEM: SympyHomotopy | None = None
_WORKER_OPTIONS: PathTrackerOptions | None = None


def _init_track_worker(
    equations: tuple[Any, ...],
    variables: tuple[sp.Symbol, ...],
    degrees: tuple[int, ...],
    gamma: Any,
    options: PathTrackerOptions,
) -> None:
    global _WORKER_SYSTEM, _WORKER_OPTIONS
    _WORKER_SYSTEM = TotalDegreeHomotopy(equations, variables, degrees, gamma).system()
    _WORKER_OPTIONS = options


def _track_worker(start: tuple[Any, ...]) -> PathResult:
    if _WORKER_SYSTEM is None or _WORKER_OPTIONS is None:  # pragma: no cover - defensive
        raise RuntimeError("homotopy worker was not initialized")
    return track_path(_WORKER_SYSTEM, start, options=_WORKER_OPTIONS)


def track_total_degree_paths(
    homotopy: TotalDegreeHomotopy,
    *,
    options: PathTrackerOptions,
    parallel: bool = False,
    max_workers: int | None = None,
) -> tuple[PathResult, ...]:
    """Track every start solution of a total-degree homotopy."""
    if not parallel or homotopy.path_count <= 1:
        system = homotopy.system()
        return tuple(
            track_path(system, start, options=options) for start in homotopy.iter_start_points()
        )

    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_init_track_worker,
        initargs=(
            homotopy.equations,
            homotopy.variables,
            homotopy.degrees,
            homotopy.gamma,
            options,
        ),
    ) as executor:
        return tuple(executor.map(_track_worker, homotopy.iter_start_points(), chunksize=1))
