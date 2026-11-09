"""Bundle external FLINT dependencies into host-built extension-field wheels."""

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    """Repair every input wheel with the platform's dependency bundler."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    wheels = sorted(args.directory.glob("*.whl"))
    if not wheels:
        parser.error("no wheels found")
    for wheel in wheels:
        if sys.platform == "darwin":
            executable = shutil.which("delocate-wheel")
            if executable is None:
                raise RuntimeError("install delocate before repairing a macOS wheel")
            command = [
                executable,
                "-w",
                str(args.output),
                "--require-archs",
                platform.machine(),
                str(wheel),
            ]
        elif os.name == "nt":
            prefix = os.environ.get("ALGROOTS_FLINT_PREFIX")
            if not prefix:
                raise RuntimeError("set ALGROOTS_FLINT_PREFIX to the FLINT installation")
            command = [
                sys.executable,
                "-m",
                "delvewheel",
                "repair",
                "--add-path",
                str(Path(prefix) / "bin"),
                "-w",
                str(args.output),
                str(wheel),
            ]
        else:
            command = [
                sys.executable,
                "-m",
                "auditwheel",
                "repair",
                "-w",
                str(args.output),
                str(wheel),
            ]
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
