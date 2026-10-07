"""Run every public example outside the source working directory."""

import ast
import importlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
CATALOG = json.loads((EXAMPLES / "catalog.json").read_text())


def _run(path, tmp_path, *arguments):
    environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONHASHSEED="0")
    environment.pop("PYTHONOPTIMIZE", None)
    return subprocess.run(
        [sys.executable, str(path), *arguments],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


@pytest.mark.parametrize("entry", CATALOG, ids=lambda entry: entry["file"])
def test_catalog_example_runs_standalone(entry, tmp_path):
    process = _run(EXAMPLES / entry["file"], tmp_path)
    assert process.returncode == 0, process.stdout + process.stderr
    assert process.stdout.strip(), entry["file"]
    assert list(tmp_path.iterdir()) == [], "examples should not create output files"


def test_catalog_is_complete_and_uses_public_imports():
    names = [entry["file"] for entry in CATALOG]
    assert len(set(names)) == len(names)
    assert set(names) == {path.name for path in EXAMPLES.glob("[0-9][0-9]_*.py")}
    documentation = (EXAMPLES / "README.md").read_text()
    for entry in CATALOG:
        assert re.fullmatch(r"[0-9]{2}_[a-z_]+\.py", entry["file"])
        assert all(isinstance(value, str) and value for value in entry.values())
        assert f"({entry['file']})" in documentation
        tree = ast.parse((EXAMPLES / entry["file"]).read_text())
        assert any(isinstance(node, ast.FunctionDef) and node.name == "main" for node in tree.body)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.startswith("algroots")
            ):
                assert all(not part.startswith("_") for part in node.module.split("."))
                module = importlib.import_module(node.module)
                assert all(hasattr(module, alias.name) for alias in node.names)
                assert all(not alias.name.startswith("_") for alias in node.names)


def test_runner_lists_entire_catalog_without_execution(tmp_path):
    process = _run(EXAMPLES / "run_all.py", tmp_path, "--list")
    assert process.returncode == 0
    assert len(process.stdout.splitlines()) == len(CATALOG)
    assert all(entry["file"] in process.stdout for entry in CATALOG)


def test_runner_executes_selected_examples(tmp_path):
    process = _run(
        EXAMPLES / "run_all.py", tmp_path, "01_polynomial_system.py", "15_limits_and_failures.py"
    )
    assert process.returncode == 0, process.stdout + process.stderr
    assert "Positive-dimensional input refused" in process.stdout
    assert "02_branches_and_poles.py" not in process.stdout


@pytest.mark.parametrize("name", ["missing.py", "../src/algroots/__init__.py", "/tmp/unlisted.py"])
def test_runner_refuses_unlisted_paths(name, tmp_path):
    process = _run(EXAMPLES / "run_all.py", tmp_path, name)
    assert process.returncode == 2 and "unknown example" in process.stderr
