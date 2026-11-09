"""Execute packaged tutorials in fresh processes outside the source checkout."""

import argparse
import json
import os
import re
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path

import sympy as sp

import algroots


def _run(script, directory, environment, timeout):
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=directory,
        env=environment,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"tutorial {script.name} failed:\n{result.stdout}\n{result.stderr}")


def main():
    """Check installed origins, all catalog programs, and marked Markdown examples."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optional-backends", action="store_true")
    parser.add_argument("--portable", action="store_true", help="run explicit fallback tutorials")
    parser.add_argument("--timeout", type=int, default=180, help="seconds per tutorial process")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    if not Path(algroots.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError(
            "algroots must be imported from the installed environment, outside the checkout"
        )
    data = Path(sysconfig.get_path("data")) / "share/algroots"
    catalog = json.loads((data / "examples/catalog.json").read_text())
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1")
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONOPTIMIZE", None)
    documents = []
    with tempfile.TemporaryDirectory(prefix="algroots-tutorials-") as temporary:
        directory = Path(temporary)
        for entry in catalog:
            if args.portable and entry["file"] not in {
                "25_groebner_operations.py",
                "26_solver_settings.py",
                "27_coefficient_fields.py",
            }:
                continue
            _run(data / "examples" / entry["file"], directory, environment, args.timeout)
        for document in [data / "README.md", *sorted((data / "docs").glob("*.md"))]:
            if args.portable:
                continue
            blocks = re.findall(
                r"<!-- algroots: execute -->\s*```python\n(.*?)```",
                document.read_text(),
                flags=re.S,
            )
            if blocks:
                script = directory / "tutorial.py"
                script.write_text(
                    "namespace = {'__name__': '__algroots_tutorial__'}\n"
                    + "\n".join(
                        f"exec(compile({code!r}, {document.name!r}, 'exec'), namespace)"
                        for code in blocks
                    )
                )
                _run(script, directory, environment, args.timeout)
                documents.append(document.name)
    engines = {}
    if args.optional_backends:
        x, y = sp.symbols("x y")
        equations = [x * x - y, y * y - 1]
        for name, info in algroots.groebner_backends().items():
            if not info.available:
                engines[name] = {"status": "unavailable", "reason": info.reason}
                continue
            domains = (sp.QQ, sp.GF(101)) if "prime fields" in info.domains else (sp.QQ,)
            for domain in domains:
                basis = algroots.groebner_basis(equations, (x, y), domain=domain, backend=name)
                if basis != sp.groebner(equations, x, y, domain=domain, order="grevlex"):
                    raise ArithmeticError(
                        f"installed backend {name} returned an incorrect basis over {domain}"
                    )
            engines[name] = {"status": "passed", "domains": list(map(str, domains))}
    print(
        json.dumps(
            {
                "origin": algroots.__file__,
                "catalog_examples": 3 if args.portable else len(catalog),
                "profile": "portable" if args.portable else "native",
                "marked_documents": documents,
                "backends": engines,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
