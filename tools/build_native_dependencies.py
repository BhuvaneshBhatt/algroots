"""Build pinned shared FLINT dependencies for portable Unix wheels."""

import argparse
import hashlib
import http.client
import json
import os
import platform
import shutil
import ssl
import subprocess
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

SOURCES = Path(__file__).with_name("native_sources.json")


class SourceChecksumError(ValueError):
    """A downloaded native source archive differs from its pinned digest."""


def _archive_matches(path, expected):
    if not path.is_file():
        return False
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() == expected


def download_archive(urls, destination, sha256, *, timeout=60, attempts=3):
    """Publish a checksum-verified archive after bounded retries and mirror fallback.

    Partial responses and bad cached files must never become accepted source
    archives. Stage each response beside the cache and replace the destination
    only after its complete contents match the pinned release digest.
    """
    urls = (urls,) if isinstance(urls, str) else tuple(urls)
    if not urls or attempts < 1:
        raise ValueError("at least one source URL and a positive attempt count are required")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if _archive_matches(destination, sha256):
        print("Reusing verified source archive:", destination, flush=True)
        return destination
    destination.unlink(missing_ok=True)
    last_error = None
    with tempfile.TemporaryDirectory(prefix="native-download-", dir=destination.parent) as folder:
        staged = Path(folder) / destination.name
        for url in urls:
            for attempt in range(1, attempts + 1):
                print(f"Downloading {url} (attempt {attempt}/{attempts})", flush=True)
                try:
                    with (
                        urllib.request.urlopen(url, timeout=timeout) as response,
                        staged.open("wb") as stream,
                    ):
                        shutil.copyfileobj(response, stream)
                    if not _archive_matches(staged, sha256):
                        raise SourceChecksumError(f"native source checksum mismatch: {url}")
                    staged.replace(destination)
                    return destination
                except (
                    urllib.error.URLError,
                    TimeoutError,
                    ConnectionError,
                    ssl.SSLError,
                    http.client.IncompleteRead,
                    SourceChecksumError,
                ) as error:
                    last_error = error
                    print(f"Source download failed: {error}", flush=True)
                    if attempt < attempts:
                        time.sleep(min(2 ** (attempt - 1), 8))
    raise RuntimeError(
        f"could not obtain a verified native source archive: {destination.name}"
    ) from last_error


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
    sources = json.loads(SOURCES.read_text(encoding="utf-8"))
    identity = {
        "sources": sources,
        "machine": platform.machine(),
        "target": os.environ.get("MACOSX_DEPLOYMENT_TARGET"),
    }
    stamp = prefix / "algroots-native-build.json"
    if stamp.exists() and json.loads(stamp.read_text(encoding="utf-8")) == identity:
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
        download_archive((spec["url"], *spec.get("mirrors", ())), archive, spec["sha256"])
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
    stamp.write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
    print("Built checked shared native dependencies:", prefix)


if __name__ == "__main__":
    main()
