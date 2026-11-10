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
    parser.add_argument("path", type=Path, help="a wheel file or a directory containing wheels")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    wheels = [args.path] if args.path.is_file() else sorted(args.path.glob("*.whl"))
    if not wheels:
        parser.error("no wheels found")
    environment = os.environ.copy()
    prefix = environment.get("ALGROOTS_FLINT_PREFIX")
    if prefix and os.name != "nt":
        # Link-time -L flags do not make private shared libraries discoverable
        # to the wheel repairer. Retain Python's own loader search path too.
        variable = "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"
        libraries = [str(Path(prefix) / name) for name in ("lib", "lib64")]
        if environment.get(variable):
            libraries.append(environment[variable])
        environment[variable] = os.pathsep.join(libraries)
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
            # manylinux installs auditwheel separately from each wheel's Python.
            executable = shutil.which("auditwheel")
            if executable is None:
                raise RuntimeError("install auditwheel before repairing a Linux wheel")
            command = [
                executable,
                "repair",
                "-w",
                str(args.output),
                str(wheel),
            ]
        subprocess.run(command, check=True, env=environment)


if __name__ == "__main__":
    main()
