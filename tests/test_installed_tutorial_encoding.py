"""Packaged tutorials use UTF-8 even when the host text default is cp1252."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace


def test_tutorial_encoding(monkeypatch, tmp_path, capsys):
    source = Path(__file__).resolve().parents[1] / "tools/check_installed_examples.py"
    spec = importlib.util.spec_from_file_location("installed_examples", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = tmp_path / "share/algroots"
    (data / "examples").mkdir(parents=True)
    (data / "docs").mkdir()
    (data / "examples/catalog.json").write_text(
        json.dumps([{"file": "unicode.py"}]), encoding="utf-8"
    )
    (data / "examples/unicode.py").write_text("print('ℚ')\n", encoding="utf-8")
    (data / "README.md").write_text(
        "# “Gröbner” over ℚ\n\n<!-- algroots: execute -->\n```python\nassert 'ℚ' == chr(0x211A)\n```\n",
        encoding="utf-8",
    )
    read_text, write_text = Path.read_text, Path.write_text

    def read(path, encoding=None, errors=None):
        return read_text(path, encoding=encoding or "cp1252", errors=errors)

    def write(path, text, encoding=None, errors=None, **kwargs):
        return write_text(path, text, encoding=encoding or "cp1252", errors=errors, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(Path, "write_text", write)
    monkeypatch.setattr(module.sysconfig, "get_path", lambda name: str(tmp_path))
    monkeypatch.setattr(
        module, "algroots", SimpleNamespace(__file__=str(Path(sys.prefix) / "algroots/__init__.py"))
    )
    monkeypatch.setattr(sys, "argv", ["check_installed_examples.py"])
    module.main()
    result = json.loads(capsys.readouterr().out)
    assert result["catalog_examples"] == 1
    assert result["marked_documents"] == ["README.md"]
