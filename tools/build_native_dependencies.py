"""Build pinned shared FLINT dependencies for portable Unix wheels."""

import argparse
import hashlib
import http.client
import json
import os
import platform
import subprocess
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

SOURCES = Path(__file__).with_name("native_sources.json")


def download_archive(url: str, destination: Path, *, timeout: int = 180, attempts: int = 6) -> None:
    """Download an archive with retries for transient network failures."""
    if attempts < 1:
        raise ValueError("attempts must be positive")
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                destination.write_bytes(response.read())
            return
        except (urllib.error.URLError, TimeoutError, http.client.IncompleteRead) as exc:
            if attempt >= attempts:
                raise
            wait = min(float(2**attempt), 30.0)
            print(
                f"Download failed for {url}: {exc}. Retrying in {wait:.1f}s "
                f"(attempt {attempt + 1} of {attempts})"
            )
            time.sleep(wait)


def main():
    """Download checked archives and build all libraries at one deployment target."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--source-cache", type=Path)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    if os.name == "nt":
        parser.error("Windows builds use the pinned vcpkg installation in the wheel workflow")
    if args.jobs < 1:
        parser.error("jobs must be positive")
    prefix = args.prefix.resolve()
    sources = json.loads(SOURCES.read_text())
    identity = {
        "sources": sources,
        "machine": platform.machine(),
        "target": os.environ.get("MACOSX_DEPLOYMENT_TARGET"),
    }
    stamp = prefix / "algroots-native-build.json"
    if stamp.exists() and json.loads(stamp.read_text()) == identity:
        print("Reusing checked native dependency build:", prefix)
        return
    cache = (args.source_cache or prefix.parent / "native-source-cache").resolve()
    cache.mkdir(parents=True, exist_ok=True)
    prefix.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    target = env.get("MACOSX_DEPLOYMENT_TARGET")
    flags = "-O2 -fPIC"
    if target and platform.system() == "Darwin":
        flags += " -mmacosx-version-min=" + target
    env["CFLAGS"] = flags
    env["CXXFLAGS"] = flags
    env["LDFLAGS"] = "-L" + str(prefix / "lib") + " -Wl,-rpath," + str(prefix / "lib")
    env["CPPFLAGS"] = "-I" + str(prefix / "include")
    for name in ("gmp", "mpfr", "flint"):
        spec = sources[name]
        archive = cache / spec["url"].rsplit("/", 1)[1]
        if not archive.exists():
            download_archive(spec["url"], archive)
        if hashlib.sha256(archive.read_bytes()).hexdigest() != spec["sha256"]:
            raise ValueError("native source checksum mismatch: " + str(archive))
        build = prefix.parent / ("native-build-" + name)
        build.mkdir(exist_ok=True)
        with tarfile.open(archive) as contents:
            contents.extractall(build, filter="data")
        source = build / (name + "-" + spec["version"])
        command = [
            str(source / "configure"),
            "--prefix=" + str(prefix),
            "--enable-shared",
            "--disable-static",
        ]
        if name == "gmp":
            command.append("--enable-fat")
        else:
            command.append("--with-gmp=" + str(prefix))
        if name == "flint":
            command.extend(["--with-mpfr=" + str(prefix), "--disable-avx2", "--disable-avx512"])
        subprocess.run(command, cwd=source, env=env, check=True)
        subprocess.run(["make", "-j" + str(args.jobs)], cwd=source, env=env, check=True)
        subprocess.run(["make", "install"], cwd=source, env=env, check=True)
    stamp.write_text(json.dumps(identity, indent=2) + "\n")
    print("Built checked shared native dependencies:", prefix)


if __name__ == "__main__":
    main()
