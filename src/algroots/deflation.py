"""Targeted exact determinantal deflation without auxiliary variables."""

from dataclasses import dataclass

import sympy as sp

from .certification import (
    RootCertificationError,
    _evaluate,
    _jacobian_data,
    _positive_integer,
    certify_isolated_root,
)
from .numerical import (
    compile_polynomials,
    compiled_jacobian,
    compiled_residual,
    newton_refine,
    sympy_to_mpc,
)


@dataclass(frozen=True)
class DeflationStage:
    rank_before: int
    pivot_rows: tuple
    pivot_columns: tuple
    pivot_value: object
    added_equations: tuple
    rank_after: int


@dataclass(frozen=True)
class DeflatedRefinement:
    root: tuple
    converged: bool
    deflated_relative_residual: object
    original_relative_residual: object
    target_distance: object
    status: str = "numerical"


@dataclass(frozen=True)
class DeflationResult:
    certificate: object
    equations: tuple
    variables: tuple
    point: tuple
    stages: tuple
    final_rank: int
    stopping_reason: str
    max_stages: int
    max_added_equations: int

    @property
    def regular(self):
        return self.final_rank == len(self.variables)

    def refine(self, approximate, *, digits=50, maxsteps=50):
        """Use the overdetermined deflated system for numerical Newton refinement.

        The target's exact certificate is separate from the returned numerical
        point. Convergence here does not certify association with that target.
        """
        _positive_integer(maxsteps, "maxsteps")
        if isinstance(digits, bool) or not isinstance(digits, int) or digits < 15:
            raise ValueError("digits must be an integer >=15")
        approximate = tuple(approximate)
        if len(approximate) != len(self.variables):
            raise ValueError("point dimension does not match")
        values = tuple(sympy_to_mpc(v, digits + 10) for v in approximate)
        polynomials = compile_polynomials(self.equations, self.variables)
        refined, converged = newton_refine(
            polynomials, compiled_jacobian(self.equations, self.variables), values, digits, maxsteps
        )
        return DeflatedRefinement(
            refined,
            converged,
            compiled_residual(polynomials, refined, digits),
            compiled_residual(
                compile_polynomials(self.certificate.equations, self.variables), refined, digits
            ),
            max(
                abs(v - sympy_to_mpc(p, digits + 10))
                for v, p in zip(refined, self.point, strict=True)
            ),
        )

    def verify(self, *, max_quotient_dimension=256, max_refinements=128):
        """Replay the exact certificate, pivot choices and every deflation equation."""
        if not self.certificate.verify(
            max_quotient_dimension=max_quotient_dimension, max_refinements=max_refinements
        ):
            return False
        try:
            replay = _deflate_certificate(
                self.certificate,
                max_stages=self.max_stages,
                max_added_equations=self.max_added_equations,
            )
            return replay == self
        except (ValueError, TypeError, NotImplementedError):
            return False


def deflate_isolated_root(
    equations,
    variables,
    point,
    *,
    max_stages=8,
    max_added_equations=1024,
    max_quotient_dimension=256,
):
    """Deflate one exact isolated root using bordered Jacobian minors.

    At each stage an exact nonzero rank-r pivot minor chooses a local Jacobian
    chart. The (r+1) bordered minors impose rank <=r in that chart and vanish at
    the target. Original equations are retained. Other original roots may be
    removed: this is local deflation, not an equivalent global solving system.
    Full column Jacobian rank is proved ezactly before reporting regularity.
    Stage/equation limits return an unresolved result with an explicit reason.
    """
    _positive_integer(max_stages, "max_stages")
    _positive_integer(max_added_equations, "max_added_equations")
    certificate = certify_isolated_root(
        equations, variables, point, max_quotient_dimension=max_quotient_dimension
    )
    return _deflate_certificate(
        certificate, max_stages=max_stages, max_added_equations=max_added_equations
    )


def _deflate_certificate(certificate, *, max_stages=8, max_added_equations=1024):
    """Internal reuse path for an already proved original-system endpoint."""
    _positive_integer(max_stages, "max_stages")
    _positive_integer(max_added_equations, "max_added_equations")
    variables, point = certificate.variables, certificate.point
    domain = certificate.characteristic_polynomial.domain
    extensions = [v for v in point if v.is_Rational is not True]
    if domain.is_AlgebraicField:
        extensions.insert(0, domain.ext.as_expr())
    field = sp.QQ.algebraic_field(*extensions) if extensions else sp.QQ
    rows = list(certificate.equations)
    stages = []
    total_added = 0
    reason = "stage_limit"
    for _ in range(max_stages):
        jacobian, evaluated = _jacobian_data(tuple(rows), variables, point, field)
        rank = evaluated.rank()
        if rank == len(variables):
            reason = "regular"
            break
        pivot_columns = tuple(evaluated.rref()[1])
        pivot_rows = tuple(evaluated.transpose().rref()[1])
        pivot = jacobian.extract(pivot_rows, pivot_columns).det() if rank else sp.Integer(1)
        pivot_value = _evaluate(pivot, variables, point, field)
        if pivot_value == field.zero:
            raise RootCertificationError(
                "chosen Jacobian pivot does not define a valid local chart"
            )
        candidates = []
        for row in range(len(rows)):
            if row in pivot_rows:
                continue
            for column in range(len(variables)):
                if column in pivot_columns:
                    continue
                minor = sp.expand(
                    jacobian.extract((*pivot_rows, row), (*pivot_columns, column)).det()
                )
                if minor != 0:
                    minor = sp.Poly(minor, *variables, extension=True).monic().as_expr()
                    if minor not in rows and minor not in candidates:
                        candidates.append(minor)
                if len(candidates) + total_added > max_added_equations:
                    reason = "equation_limit"
                    break
            if reason == "equation_limit":
                break
        if reason == "equation_limit":
            break
        if not candidates:
            reason = "no_progress"
            break
        if any(_evaluate(e, variables, point, field) != field.zero for e in candidates):
            raise RootCertificationError("deflation equation failed exact target preservation")
        rows.extend(candidates)
        total_added += len(candidates)
        _, after = _jacobian_data(tuple(rows), variables, point, field)
        stages.append(
            DeflationStage(
                rank,
                pivot_rows,
                pivot_columns,
                field.to_sympy(pivot_value),
                tuple(candidates),
                after.rank(),
            )
        )
        if after.rank() == len(variables):
            reason = "regular"
            break
    _, evaluated = _jacobian_data(tuple(rows), variables, point, field)
    return DeflationResult(
        certificate,
        tuple(rows),
        variables,
        point,
        tuple(stages),
        evaluated.rank(),
        reason,
        max_stages,
        max_added_equations,
    )
