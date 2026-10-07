"""Structural checks that keep Markdown navigation and the public API aligned."""

import re
from pathlib import Path

import sympy as sp

import algroots

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def test_every_public_export_is_documented_in_api_reference():
    api_text = (DOCS / "api.md").read_text(encoding="utf-8")
    missing = [name for name in algroots.__all__ if f"`{name}`" not in api_text]
    assert missing == []


def test_markdown_relative_links_resolve():
    markdown_files = [ROOT / "README.md", *DOCS.glob("*.md"), *(ROOT / "examples").glob("*.md")]
    missing: list[tuple[str, str]] = []
    link_pattern = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
    for path in markdown_files:
        text = path.read_text(encoding="utf-8")
        for target in link_pattern.findall(text):
            target = target.split("#", 1)[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            resolved = (path.parent / target).resolve()
            if not resolved.exists():
                missing.append((str(path.relative_to(ROOT)), target))
    assert missing == []


def test_docs_index_links_every_topic_page():
    index = (DOCS / "index.md").read_text(encoding="utf-8")
    unlinked = [
        path.name
        for path in DOCS.glob("*.md")
        if path.name != "index.md" and f"({path.name})" not in index
    ]
    assert unlinked == []


def _documented_parameters(function_name: str):
    import ast

    api_text = (DOCS / "api.md").read_text(encoding="utf-8")
    pattern = re.compile(
        rf"```python\s*{re.escape(function_name)}\((.*?)\)\s*```",
        flags=re.DOTALL,
    )
    match = pattern.search(api_text)
    assert match is not None, f"missing documented signature for {function_name}"
    declaration = ast.parse("def _documented(" + match.group(1) + "): pass").body[0].args
    assert not declaration.posonlyargs and declaration.vararg is None and declaration.kwarg is None

    def default_value(node):
        return sp.QQ if isinstance(node, ast.Name) and node.id == "QQ" else ast.literal_eval(node)

    parameters = []
    first_default = len(declaration.args) - len(declaration.defaults)
    for index, argument in enumerate(declaration.args):
        required = index < first_default
        value = None if required else default_value(declaration.defaults[index - first_default])
        parameters.append((argument.arg, False, required, value))
    for argument, node in zip(declaration.kwonlyargs, declaration.kw_defaults, strict=True):
        parameters.append(
            (argument.arg, True, node is None, None if node is None else default_value(node))
        )
    return parameters


def test_documented_public_function_signatures_match_runtime_api():
    import inspect

    function_names = (
        "algsolve",
        "polysolve",
        "algebraize_system",
        "recognize_system_roots",
        "discover_monodromy_orbit",
        "compute_rational_univariate_representation",
        "solve_rur_representation",
        "solve_rur_points",
        "solve_zero_dimensional_system_with_rur",
        "compute_border_basis",
        "compute_border_basis_linear",
        "certify_isolated_root",
        "certify_root_box",
        "certify_numerical_roots",
        "deflate_isolated_root",
        "cauchy_endgame",
        "track_projective_path",
    )
    for function_name in function_names:
        documented = _documented_parameters(function_name)
        signature = inspect.signature(
            getattr(
                __import__("algroots." + FUNCTION_MODULES[function_name], fromlist=[function_name]),
                function_name,
            )
        )
        actual = []
        for parameter in signature.parameters.values():
            required = parameter.default is inspect.Parameter.empty
            default = None if required else parameter.default
            actual.append(
                (
                    parameter.name,
                    parameter.kind is inspect.Parameter.KEYWORD_ONLY,
                    required,
                    default,
                )
            )
        assert documented == actual, function_name


def test_recovery_defaults_match_workflow_budget_table():
    from dataclasses import fields

    from algroots import HomotopyRecoveryOptions

    text = (DOCS / "recovery-workflow.md").read_text()
    rows = dict(re.findall(r"\| `([a-z_]+)` \| (\d+) \|", text))
    assert {field.name: field.default for field in fields(HomotopyRecoveryOptions)} == {
        name: int(default) for name, default in rows.items()
    }


def test_local_markdown_anchors_resolve():
    """Check fragments as well as files for the README, guides and example index."""
    paths = [ROOT / "README.md", *DOCS.glob("*.md"), *(ROOT / "examples").glob("*.md")]
    missing = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for target in re.findall(r"(?<!!)\[[^\]]+\]\(([^)]+)\)", text):
            if "://" in target or "#" not in target:
                continue
            filename, fragment = target.split("#", 1)
            destination = (path.parent / filename).resolve() if filename else path
            if destination.suffix != ".md" or not destination.exists() or not fragment:
                continue
            content = re.sub(r"```.*?```", "", destination.read_text(), flags=re.DOTALL)
            anchors = set(re.findall(r'<a\s+(?:id|name)="([^"]+)"', content))
            counts = {}
            for heading in re.findall(r"^#{1,6}\s+(.+)$", content, flags=re.MULTILINE):
                slug = re.sub(r"[^\w\s-]", "", heading.lower()).strip().replace(" ", "-")
                count = counts.get(slug, 0)
                anchors.add(slug if count == 0 else f"{slug}-{count}")
                counts[slug] = count + 1
            if fragment not in anchors:
                missing.append((str(path.relative_to(ROOT)), target))
    assert missing == []


def test_documented_result_evidence_attributes_exist():
    from dataclasses import fields

    from algroots import CompletenessEvidence

    text = (DOCS / "reading-results.md").read_text()
    for field in fields(CompletenessEvidence):
        assert f"`{field.name}`" in text, field.name


def test_citation_version_matches_project_metadata():
    import tomllib

    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    citation = (ROOT / "CITATION.cff").read_text()
    match = re.search(r'^version:\s*"([^"]+)"$', citation, flags=re.MULTILINE)
    assert match is not None and match.group(1) == version


FUNCTION_MODULES = {
    "AlgebraicSystemRoots": "algebraic",
    "algsolve": "algebraic",
    "AlgebraizedSystem": "algebraization",
    "algebraize_system": "algebraization",
    "BorderBasisDiagnostics": "border_basis",
    "BorderBasisError": "border_basis",
    "BorderBasisResult": "border_basis",
    "compute_border_basis": "border_basis",
    "compute_border_basis_linear": "border_basis",
    "IsolatedRootCertificate": "certification",
    "RationalComplexBox": "certification",
    "RootCertificationAttempt": "certification",
    "RootCertificationError": "certification",
    "certify_isolated_root": "certification",
    "certify_numerical_roots": "certification",
    "certify_root_box": "certification",
    "HomotopySystem": "continuation",
    "PathResult": "continuation",
    "PathStep": "continuation",
    "PathStepError": "continuation",
    "PathTrackerOptions": "continuation",
    "PathTrackingError": "continuation",
    "SympyHomotopy": "continuation",
    "track_path": "continuation",
    "DeflatedRefinement": "deflation",
    "DeflationResult": "deflation",
    "DeflationStage": "deflation",
    "deflate_isolated_root": "deflation",
    "EndgameResult": "endgames",
    "ProjectiveHomotopy": "endgames",
    "cauchy_endgame": "endgames",
    "projective_homotopy": "endgames",
    "ActionMatrixError": "errors",
    "HomotopySolveError": "errors",
    "NotZeroDimensionalError": "errors",
    "NumericalRootError": "errors",
    "PolynomialSystemError": "errors",
    "PolynomialSystemInputError": "errors",
    "QuotientAlgebraError": "errors",
    "ShapePositionError": "errors",
    "SystemSolveLimitError": "errors",
    "TriangularSolveError": "errors",
    "MonodromyLoop": "monodromy",
    "MonodromyOrbitResult": "monodromy",
    "MonodromyPermutation": "monodromy",
    "MonodromyRootInfo": "monodromy",
    "closed_additive_loop": "monodromy",
    "discover_monodromy_orbit": "monodromy",
    "monodromy_permutation": "monodromy",
    "track_loop": "monodromy",
    "CaptureRecaptureEstimate": "monodromy_stopping",
    "capture_recapture_estimate": "monodromy_stopping",
    "second_order_trace_test": "monodromy_stopping",
    "ChartSwitch": "projective_tracking",
    "ProjectivePathResult": "projective_tracking",
    "track_projective_path": "projective_tracking",
    "QuotientAlgebra": "quotient",
    "SeparatingElement": "quotient",
    "RationalUnivariateError": "rational_univariate",
    "RationalUnivariatePoint": "rational_univariate",
    "RationalUnivariateRepresentation": "rational_univariate",
    "compute_rational_univariate_representation": "rational_univariate",
    "solve_rur_points": "rational_univariate",
    "solve_rur_representation": "rational_univariate",
    "solve_zero_dimensional_system_with_rur": "rational_univariate",
    "ExactCertificationError": "recognition",
    "RecognizedSystemRoot": "recognition",
    "recognize_system_roots": "recognition",
    "HomotopyRecoveryOptions": "recovery",
    "PathRecoveryRecord": "recovery",
    "CompletenessEvidence": "solver",
    "PolynomialSystemRoots": "solver",
    "RootDiagnostics": "solver",
    "SolveCostDiagnostics": "solver",
    "polysolve": "solver",
}
