"""Fetch the exact upstream source commits selected for stack validation."""

import argparse
import json
import re
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    """Materialize a candidate stack without moving branch references."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lock", type=Path, default=ROOT / "ci/release-sources.json")
    args = parser.parse_args()
    output = args.output.resolve()
    if output == ROOT or ROOT.is_relative_to(output) or output.is_relative_to(ROOT):
        parser.error("use an output directory outside this checkout")
    if output.exists() and any(output.iterdir()):
        parser.error("output must be empty")
    output.mkdir(parents=True, exist_ok=True)
    sources = json.loads(args.lock.read_text())
    with tempfile.TemporaryDirectory(prefix="release-sources-") as temp:
        cache = Path(temp)
        for name, source in sources.items():
            repository, commit = source["repository"], source["commit"]
            if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository) or not re.fullmatch(
                r"[0-9a-f]{40}", commit
            ):
                raise ValueError("invalid pinned repository or commit")
            if name not in {"semialg", "alggeom", "funcprops", "symbopt", "probstats"}:
                raise ValueError("unexpected release source: " + name)
            archive = cache / (name + ".tar.gz")
            url = f"https://codeload.github.com/{repository}/tar.gz/{commit}"
            with urllib.request.urlopen(url, timeout=120) as response:
                archive.write_bytes(response.read())
            with tarfile.open(archive) as contents:
                contents.extractall(cache / name, filter="data")
            folders = list((cache / name).iterdir())
            if len(folders) != 1:
                raise ValueError("unexpected repository archive")
            shutil.move(str(folders[0]), output / name)
    shutil.copytree(
        ROOT,
        output / "algroots",
        ignore=shutil.ignore_patterns(
            ".git",
            "__pycache__",
            ".pytest_cache",
            ".ruff_cache",
            ".ci-backends",
            "build",
            "dist",
            "*.egg-info",
            "*.so",
            ".dylibs",
        ),
    )
    print("Materialized the pinned candidate source stack:", output)


if __name__ == "__main__":
    main()
