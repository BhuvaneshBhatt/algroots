"""Keep executable user-facing documentation synchronized with the package."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXECUTE_MARKER = "<!-- algroots: execute -->"


def _python_blocks(relative_path: str) -> list[str]:
    text = (ROOT / relative_path).read_text(encoding="utf-8")
    return re.findall(r"```python\n(.*?)```", text, flags=re.DOTALL)


def _executable_blocks(relative_path: str) -> list[str]:
    text = (ROOT / relative_path).read_text(encoding="utf-8")
    pattern = re.compile(
        rf"{re.escape(EXECUTE_MARKER)}\s*```python\n(.*?)```",
        flags=re.DOTALL,
    )
    return pattern.findall(text)


def _marked_documents() -> list[str]:
    paths = [ROOT / "README.md", *(ROOT / "docs").glob("*.md")]
    return sorted(
        str(path.relative_to(ROOT))
        for path in paths
        if EXECUTE_MARKER in path.read_text(encoding="utf-8")
    )


@pytest.mark.parametrize("relative_path", _marked_documents())
@pytest.mark.slow
def test_marked_documented_python_examples_execute(relative_path):
    blocks = _executable_blocks(relative_path)
    assert blocks, f"{relative_path} contains an execution marker but no marked block"
    namespace = {"__name__": "__algroots_documentation_example__"}
    for index, code in enumerate(blocks, start=1):
        if "recognize_system_roots(" in code:
            pytest.importorskip("flint")
            pytest.importorskip("algrecognize")
        exec(compile(code, f"{relative_path}:marked-block-{index}", "exec"), namespace)


def test_execution_markers_are_immediately_followed_by_python_fences():
    for path in [ROOT / "README.md", *(ROOT / "docs").glob("*.md")]:
        text = path.read_text(encoding="utf-8")
        marker_count = text.count(EXECUTE_MARKER)
        executable_count = len(
            re.findall(
                rf"{re.escape(EXECUTE_MARKER)}\s*```python\n",
                text,
            )
        )
        assert executable_count == marker_count, path.relative_to(ROOT)


def test_all_python_fences_are_syntactically_valid_or_explicit_signatures():
    """Catch stale/garbled code fences without executing deliberate error examples."""
    for path in [ROOT / "README.md", *(ROOT / "docs").glob("*.md")]:
        for index, code in enumerate(_python_blocks(str(path.relative_to(ROOT)))):
            # api.md contains signature-display blocks, not executable statements.
            if path.name == "api.md":
                continue
            compile(code, f"{path.name}:block-{index + 1}", "exec")
