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
    markdown_files = [ROOT / "README.md", *DOCS.glob("*.md")]
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
        rf"```python\s*{re.escape(function_name)}\(\n(.*?)\n\)\s*```",
        flags=re.DOTALL,
    )
    match = pattern.search(api_text)
    assert match is not None, f"missing documented signature for {function_name}"

    parameters = []
    keyword_only = False
    for raw_line in match.group(1).splitlines():
        token = raw_line.strip().rstrip(",")
        if not token:
            continue
        if token == "*":
            keyword_only = True
            continue
        if "=" in token:
            name, default_text = token.split("=", 1)
            default = sp.QQ if default_text == "QQ" else ast.literal_eval(default_text)
            required = False
        else:
            name = token
            default = None
            required = True
        parameters.append((name, keyword_only, required, default))
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
    )
    for function_name in function_names:
        documented = _documented_parameters(function_name)
        signature = inspect.signature(getattr(algroots, function_name))
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
