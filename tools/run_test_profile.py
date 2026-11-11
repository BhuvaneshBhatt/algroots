"""Run the full native suite or explicit portable-fallback contracts."""

import argparse
import subprocess
import sys
from pathlib import Path

PORTABLE_TESTS = (
    "test_precision_contexts.py",
    "test_coefficient_state.py",
    "test_native_defaults.py",
    "test_groebner_api.py",
    "test_groebner_generated.py",
    "test_finite_fields.py",
    "test_native_finite_fields.py",
    "test_rational_basis_verification.py",
    "test_streaming_rational_certificates.py",
    "test_proof_performance.py",
)


def main():
    """Keep native-default requirements separate from explicit SymPy portability."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--junitxml")
    parser.add_argument("--verbose", action="store_true", help="show each test as it runs")
    args = parser.parse_args()
    from algroots.native_f4 import compiled_available

    if compiled_available() != args.native:
        parser.error("installed native availability differs from the selected test profile")
    root = Path(__file__).resolve().parents[1]
    tests = [root / "tests"] if args.native else [root / "tests" / p for p in PORTABLE_TESTS]
    command = [sys.executable, "-m", "pytest", "-v" if args.verbose else "-q", "-o", "pythonpath="]
    if args.junitxml:
        command.append("--junitxml=" + args.junitxml)
    command.extend(map(str, tests))
    print("Full native suite" if args.native else "Explicit SymPy fallback contracts", flush=True)
    raise SystemExit(subprocess.call(command, cwd=root))


if __name__ == "__main__":
    main()
