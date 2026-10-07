"""List or run the catalog without changing precision or skipping failed examples."""

import argparse
import json
import runpy
from pathlib import Path


def main():
    directory = Path(__file__).resolve().parent
    catalog = json.loads((directory / "catalog.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", help="catalog filenames; omit to run all")
    parser.add_argument("--list", action="store_true", help="list examples without running")
    args = parser.parse_args()
    known = {entry["file"]: entry for entry in catalog}
    for name in args.names:
        if name not in known:
            parser.error(f"unknown example: {name}")
    selected = args.names or list(known)
    for name in selected:
        entry = known[name]
        print(f"{name}: {entry['title']} [{entry['guarantee']}]", flush=True)
        if not args.list:
            runpy.run_path(str(directory / name), run_name="__main__")


if __name__ == "__main__":
    main()
