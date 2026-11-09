"""Verify candidate Python sources, sdist payloads and native wheel dependencies."""

import argparse
import hashlib
import json
import platform
import re
import subprocess
import tarfile
import tempfile
import zipfile
from pathlib import Path


def main():
    """Check each artifact against the exact source tree selected for release."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--macos-target", type=float)
    args = parser.parse_args()
    reports = []
    for artifact in sorted(args.directory.rglob("*")):
        if not artifact.is_file() or not (
            artifact.suffix == ".whl" or artifact.name.endswith(".tar.gz")
        ):
            continue
        name = artifact.name.split("-", 1)[0]
        source = args.source_root / name
        if not source.is_dir():
            raise ValueError("unknown release artifact: " + artifact.name)
        expected = {
            p.relative_to(source / "src").as_posix(): p.read_bytes()
            for p in (source / "src" / name).rglob("*")
            if p.is_file() and p.suffix in {".py", ".json"}
        }
        row = {
            "artifact": str(artifact.resolve()),
            "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        }
        if artifact.suffix == ".whl":
            with zipfile.ZipFile(artifact) as wheel:
                if wheel.testzip() is not None:
                    raise ValueError("wheel CRC failure")
                actual = {
                    p: wheel.read(p)
                    for p in wheel.namelist()
                    if p.startswith(name + "/") and p.endswith((".py", ".json"))
                }
                binaries = [
                    p for p in wheel.namelist() if p.endswith((".so", ".pyd", ".dylib", ".dll"))
                ]
                if artifact.name.endswith("none-any.whl") and binaries:
                    raise ValueError("fallback wheel contains compiled modules")
                if name == "algroots" and binaries:
                    for kernel in ("_native_f4", "_native_extension", "_ff_sparse"):
                        if not any(Path(p).name.startswith(kernel + ".") for p in binaries):
                            raise ValueError("native wheel is missing " + kernel)
                row["binaries"] = binaries
                if platform.system() == "Darwin" and "macosx" in artifact.name:
                    libraries = []
                    with tempfile.TemporaryDirectory(prefix="wheel-audit-") as temp:
                        for member in binaries:
                            path = Path(wheel.extract(member, temp))
                            dependencies = subprocess.run(
                                ["otool", "-L", str(path)],
                                capture_output=True,
                                text=True,
                                check=True,
                            ).stdout
                            build = subprocess.run(
                                ["vtool", "-show-build", str(path)],
                                capture_output=True,
                                text=True,
                                check=True,
                            ).stdout
                            identifiers = subprocess.run(
                                ["otool", "-D", str(path)],
                                capture_output=True,
                                text=True,
                                check=True,
                            ).stdout.splitlines()[1:]
                            own_ids = {value.strip() for value in identifiers}
                            for line in dependencies.splitlines()[1:]:
                                dependency = line.strip().split(" ", 1)[0]
                                if dependency in own_ids:
                                    continue
                                if dependency.startswith("/") and not dependency.startswith(
                                    ("/usr/lib/", "/System/Library/")
                                ):
                                    raise ValueError("unbundled native dependency: " + dependency)
                            targets = [float(v) for v in re.findall(r"minos\s+(\d+\.\d+)", build)]
                            if args.macos_target is not None and (
                                not targets or any(v > args.macos_target for v in targets)
                            ):
                                raise ValueError(
                                    "native binary exceeds requested deployment target: " + member
                                )
                            libraries.append(
                                {"member": member, "dependencies": dependencies, "build": build}
                            )
                    row["native_audit"] = libraries
        else:
            with tarfile.open(artifact) as archive:
                actual = {
                    m.name.partition("/src/")[2]: archive.extractfile(m).read()
                    for m in archive.getmembers()
                    if m.isfile()
                    and "/src/" in m.name
                    and m.name.endswith((".py", ".json"))
                    and m.name.partition("/src/")[2].startswith(name + "/")
                }
                names = [m.name.partition("/")[2] for m in archive.getmembers() if m.isfile()]
                if any(
                    "__pycache__" in p or p.endswith((".pyc", ".so", ".pyd", ".dylib"))
                    for p in names
                ):
                    raise ValueError("source distribution contains generated binaries or caches")
        mismatches = sorted(
            n for n in expected.keys() | actual.keys() if expected.get(n) != actual.get(n)
        )
        if mismatches:
            raise ValueError(f"{artifact.name}: Python source mismatches: {mismatches}")
        row["runtime_files"] = len(expected)
        row["mismatches"] = mismatches
        reports.append(row)
    if not reports:
        parser.error("no release artifacts found")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(reports, indent=2) + "\n")
    print(f"Verified {len(reports)} release artifacts")


if __name__ == "__main__":
    main()
