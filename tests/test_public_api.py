import importlib

import algroots

EXPECTED = [
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
MOVED = {
    "PathRecoveryRecord": "recovery",
    "SolveCostDiagnostics": "solver",
    "RootCertificationAttempt": "certification",
    "certify_numerical_roots": "certification",
    "IsolatedRootCertificate": "certification",
    "RationalComplexBox": "certification",
    "RootCertificationError": "errors",
    "certify_isolated_root": "certification",
    "certify_root_box": "certification",
    "DeflatedRefinement": "deflation",
    "DeflationResult": "deflation",
    "DeflationStage": "deflation",
    "deflate_isolated_root": "deflation",
    "ChartSwitch": "projective_tracking",
    "ProjectivePathResult": "projective_tracking",
    "track_projective_path": "projective_tracking",
    "EndgameResult": "endgames",
    "ProjectiveHomotopy": "endgames",
    "cauchy_endgame": "endgames",
    "projective_homotopy": "endgames",
    "HomotopySystem": "continuation",
    "SympyHomotopy": "continuation",
    "PathTrackerOptions": "continuation",
    "PathStep": "continuation",
    "PathResult": "continuation",
    "PathTrackingError": "errors",
    "PathStepError": "errors",
    "track_path": "continuation",
    "CaptureRecaptureEstimate": "monodromy_stopping",
    "capture_recapture_estimate": "monodromy_stopping",
    "second_order_trace_test": "monodromy_stopping",
    "MonodromyLoop": "monodromy",
    "MonodromyPermutation": "monodromy",
    "MonodromyOrbitResult": "monodromy",
    "MonodromyRootInfo": "monodromy",
    "closed_additive_loop": "monodromy",
    "track_loop": "monodromy",
    "monodromy_permutation": "monodromy",
    "discover_monodromy_orbit": "monodromy",
    "algebraize_system": "algebraization",
    "AlgebraizedSystem": "algebraization",
    "recognize_system_roots": "recognition",
    "RootDiagnostics": "solver",
    "RecognizedSystemRoot": "recognition",
    "ExactCertificationError": "errors",
    "QuotientAlgebraError": "errors",
    "ActionMatrixError": "errors",
    "HomotopySolveError": "errors",
    "ShapePositionError": "errors",
    "TriangularSolveError": "errors",
    "NumericalRootError": "errors",
    "QuotientAlgebra": "quotient",
    "SeparatingElement": "quotient",
    "RationalUnivariateError": "errors",
    "RationalUnivariateRepresentation": "rational_univariate",
    "RationalUnivariatePoint": "rational_univariate",
    "compute_rational_univariate_representation": "rational_univariate",
    "solve_zero_dimensional_system_with_rur": "rational_univariate",
    "solve_rur_representation": "rational_univariate",
    "solve_rur_points": "rational_univariate",
    "BorderBasisDiagnostics": "border_basis",
    "BorderBasisError": "errors",
    "BorderBasisResult": "border_basis",
    "compute_border_basis": "border_basis",
    "compute_border_basis_linear": "border_basis",
}


def test_public_api():
    assert set(algroots.__all__) == set(EXPECTED)
    assert all(hasattr(algroots, name) for name in EXPECTED)
    assert not hasattr(algroots, "version")
    assert not hasattr(algroots, "PackageNotFoundError")


def test_specialized_api_is_only_in_namespaces():
    for name, module in MOVED.items():
        assert not hasattr(algroots, name), name
        assert hasattr(importlib.import_module("algroots." + module), name), name


def test_root_aliases_match_defining_modules():
    for name in EXPECTED:
        obj = getattr(algroots, name)
        if name != "__version__":
            assert obj is getattr(importlib.import_module(obj.__module__), name)


def test_error_aliases_preserve_identity_and_catch_behavior():
    from algroots import errors

    for name, module in errors._SPECIALIZED_ERRORS.items():
        assert getattr(errors, name) is getattr(importlib.import_module("algroots." + module), name)
    assert issubclass(errors.RootCertificationError, errors.PolynomialSystemError)
    assert issubclass(errors.ExactCertificationError, ValueError)
    assert issubclass(errors.PathTrackingError, RuntimeError)
