"""Build and validate matching distributions without editable installations."""

import argparse
import importlib.util
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

PACKAGES = ("algroots", "semialg", "alggeom", "funcprops", "symbopt", "probstats")


def main():
    """Build candidates together, check installed consumers, and retain raw logs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--tests", action="store_true")
    args = parser.parse_args()
    source = args.source_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    wheels = output / "distributions"
    wheels.mkdir(exist_ok=True)
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    env["MPLBACKEND"] = "Agg"
    env["ALGROOTS_BUILD_NATIVE_F4"] = "1" if args.native else "0"
    env["ALGROOTS_BUILD_FF_SPARSE"] = env["ALGROOTS_BUILD_NATIVE_F4"]
    env["ALGROOTS_BUILD_NATIVE_EXTENSION"] = env["ALGROOTS_BUILD_NATIVE_F4"]
    results = []

    def run(label, command, *, cwd):
        with (output / (label + ".txt")).open("w") as log:
            result = subprocess.run(
                command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, check=False
            )
        results.append({"check": label, "exit_code": result.returncode})
        (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        if result.returncode:
            raise RuntimeError(f"{label} failed; see {output / (label + '.txt')}")

    with tempfile.TemporaryDirectory(prefix="algebra-stack-") as temp:
        outside = Path(temp)
        for name in PACKAGES:
            run(
                "build-" + name,
                [sys.executable, "-m", "build", "--outdir", str(wheels), str(source / name)],
                cwd=outside,
            )
        if args.native:
            raw = output / "raw-native"
            raw.mkdir(exist_ok=True)
            native = next(wheels.glob("algroots-*.whl"))
            native.rename(raw / native.name)
            run(
                "repair-native",
                [
                    sys.executable,
                    str(source / "algroots/tools/repair_extension_wheels.py"),
                    str(raw),
                    "--output",
                    str(wheels),
                ],
                cwd=outside,
            )
        audit = [
            sys.executable,
            str(source / "algroots/tools/check_release_artifacts.py"),
            "--source-root",
            str(source),
            "--directory",
            str(wheels),
            "--report",
            str(output / "artifact-audit.json"),
        ]
        if platform.system() == "Darwin":
            audit.extend(["--macos-target", "14.0"])
        run("artifact-audit", audit, cwd=outside)
        candidates = sorted(wheels.glob("*.whl"))
        if len(candidates) != len(PACKAGES):
            raise RuntimeError("expected exactly one candidate wheel for each package")
        run(
            "install",
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--force-reinstall",
                *[str(p) for p in candidates],
            ],
            cwd=outside,
        )
        run("dependencies", [sys.executable, "-m", "pip", "check"], cwd=outside)
        command = [
            sys.executable,
            str(source / "algroots/tools/check_release_stack.py"),
            "--report",
            str(output / "installed-contract.json"),
        ]
        if args.native:
            command.append("--native")
        run("installed-contract", command, cwd=outside)
        for name in PACKAGES:
            rebuilt = output / "rebuilt" / name
            rebuilt.mkdir(parents=True, exist_ok=True)
            archives = list(wheels.glob(name + "-*.tar.gz"))
            if len(archives) != 1:
                raise RuntimeError("expected one source distribution for " + name)
            run(
                "rebuild-" + name,
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "wheel",
                    "--no-deps",
                    str(archives[0]),
                    "--wheel-dir",
                    str(rebuilt),
                ],
                cwd=outside,
            )
        run(
            "rebuilt-source-audit",
            [
                sys.executable,
                str(source / "algroots/tools/check_release_artifacts.py"),
                "--source-root",
                str(source),
                "--directory",
                str(output / "rebuilt"),
                "--report",
                str(output / "rebuilt-source-audit.json"),
            ],
            cwd=outside,
        )
        if args.tests:
            if not args.native:
                run(
                    "portable-contracts",
                    [
                        sys.executable,
                        str(source / "algroots/tools/run_test_profile.py"),
                        "--junitxml=" + str(output / "portable-contracts.xml"),
                    ],
                    cwd=source / "algroots",
                )
                (output / "unavailable-suites.json").write_text(
                    json.dumps(
                        {
                            "reason": "Default Gröbner construction requires native binaries; the pure profile checks explicit SymPy contracts.",
                            "full_suites": list(PACKAGES),
                        },
                        indent=2,
                    )
                    + "\n"
                )
                print(
                    "Portable contracts passed; full default-native suites require native binaries"
                )
                return
            for name in PACKAGES:
                if name == "funcprops":
                    parent = Path(importlib.util.find_spec(name).origin).parent.parent
                    command = [
                        sys.executable,
                        "-m",
                        "tools.run_test_corpus",
                        "--evidence",
                        "--require-replay-types",
                        "--package-root",
                        str(parent),
                        "--report-dir",
                        str(output / "funcprops-corpus"),
                    ]
                else:
                    command = [
                        sys.executable,
                        "-m",
                        "pytest",
                        "-q",
                        "-o",
                        "pythonpath=",
                        "--junitxml=" + str(output / (name + ".xml")),
                    ]
                    if name == "symbopt":
                        command.extend(["-m", "not unsupported and not capability_probe"])
                run("tests-" + name, command, cwd=source / name)
            run(
                "semialg-slow",
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "-o",
                    "pythonpath=",
                    "-m",
                    "slow",
                    "--junitxml=" + str(output / "semialg-slow.xml"),
                ],
                cwd=source / "semialg",
            )
    print("Built, installed and checked the coordinated stack")


if __name__ == "__main__":
    main()
