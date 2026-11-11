"""Validate native cp311-abi3 wheel payloads and audit their imported symbols."""

import argparse
import subprocess
import sys
from email.parser import Parser
from pathlib import Path
from zipfile import ZipFile

from packaging.tags import parse_tag
from packaging.utils import parse_wheel_filename

NATIVE_MODULES = {"_native_f4", "_native_extension", "_ff_sparse"}


def inspect_wheel(path, *, require_all=False):
    """Check the ABI floor, metadata agreement and extension-module filenames."""
    _, _, _, tags = parse_wheel_filename(path.name)
    if {(tag.interpreter, tag.abi) for tag in tags} != {("cp311", "abi3")}:
        raise ValueError(f"{path.name}: expected cp311-abi3 tags")
    with ZipFile(path) as archive:
        metadata = [name for name in archive.namelist() if name.endswith(".dist-info/WHEEL")]
        if len(metadata) != 1:
            raise ValueError(f"{path.name}: expected one WHEEL metadata file")
        headers = Parser().parsestr(archive.read(metadata[0]).decode("utf-8"))
        recorded = set().union(*(parse_tag(tag) for tag in headers.get_all("Tag", [])))
        if recorded != tags or headers.get("Root-Is-Purelib", "").strip().lower() != "false":
            raise ValueError(f"{path.name}: binary wheel metadata disagrees with filename")
        modules = set()
        for name in archive.namelist():
            if not name.startswith("algroots/") or not name.endswith((".so", ".pyd")):
                continue
            filename = Path(name).name
            module = filename.split(".", 1)[0]
            if module not in NATIVE_MODULES:
                raise ValueError(f"{path.name}: unexpected native module {name}")
            valid_suffix = filename == module + ".abi3.so" or filename == module + ".pyd"
            if not valid_suffix or module in modules:
                raise ValueError(f"{path.name}: invalid or duplicate extension {name}")
            modules.add(module)
        if not modules or (require_all and modules != NATIVE_MODULES):
            raise ValueError(f"{path.name}: native module set is {sorted(modules)}")
    return modules


def main():
    """Require valid native payloads and fail on any Stable-ABI audit violation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="a native wheel or wheel directory")
    parser.add_argument(
        "--require-all", action="store_true", help="require all three native modules"
    )
    args = parser.parse_args()
    wheels = [args.path] if args.path.is_file() else sorted(args.path.glob("*.whl"))
    if not wheels:
        parser.error("no wheels found")
    for path in wheels:
        modules = inspect_wheel(path, require_all=args.require_all)
        subprocess.run(
            [sys.executable, "-m", "abi3audit", "--strict", "--summary", str(path)], check=True
        )
        print(f"Verified {path.name}: cp311-abi3, {', '.join(sorted(modules))}")


if __name__ == "__main__":
    main()
