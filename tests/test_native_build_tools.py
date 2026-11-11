"""Regression checks for native builds, verified downloads and wheel repair."""

import hashlib
import importlib.util
import io
import os
import shutil
import subprocess
import sys
import textwrap
from http.client import IncompleteRead
from pathlib import Path
from types import SimpleNamespace
from urllib.error import URLError

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAYLOAD = b"verified source archive"
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()


def _tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("architecture", ["arm64", "x86_64"])
def test_macos_build_architecture(tmp_path, architecture):
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("requires Bash to execute the composite action")
    action = (ROOT / ".github/actions/native-dependencies/action.yml").read_text(encoding="utf-8")
    script = textwrap.dedent(action.split("run: |\n", 1)[1].split("\n    - ", 1)[0])
    environment_file = tmp_path / "github-env"
    environment = dict(
        os.environ,
        RUNNER_OS="macOS",
        RUNNER_TEMP=tmp_path.as_posix(),
        GITHUB_ENV=environment_file.as_posix(),
        ARCHFLAGS="-arch arm64 -arch x86_64",
        _PYTHON_HOST_PLATFORM="macosx-10.9-universal2",
    )
    # Model setup-python's universal framework defaults while running the actual action shell.
    stubs = (
        f"uname() {{ printf '%s\\n' {architecture}; }}\n"
        'python() { printf \'%s\\n\' "$ARCHFLAGS" "$_PYTHON_HOST_PLATFORM" '
        '"$MACOSX_DEPLOYMENT_TARGET" > "$RUNNER_TEMP/builder-env"; }\n'
    )
    subprocess.run([bash, "-e", "-c", stubs + script], check=True, env=environment)
    assert (tmp_path / "builder-env").read_text(encoding="utf-8").splitlines() == [
        f"-arch {architecture}",
        f"macosx-14.0-{architecture}",
        "14.0",
    ]
    exported = dict(
        line.split("=", 1) for line in environment_file.read_text(encoding="utf-8").splitlines()
    )
    assert exported == {
        "ARCHFLAGS": f"-arch {architecture}",
        "_PYTHON_HOST_PLATFORM": f"macosx-14.0-{architecture}",
        "MACOSX_DEPLOYMENT_TARGET": "14.0",
        "ALGROOTS_FLINT_PREFIX": f"{tmp_path.as_posix()}/algroots-libs",
    }


@pytest.fixture
def downloader(monkeypatch):
    module = _tool("build_native_dependencies")
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    return module


def test_verified_cache(downloader, monkeypatch, tmp_path):
    archive = tmp_path / "source.tar.gz"
    archive.write_bytes(PAYLOAD)

    def unexpected_request(*args, **kwargs):
        raise AssertionError("verified cached data must not be downloaded again")

    monkeypatch.setattr(downloader.urllib.request, "urlopen", unexpected_request)
    assert downloader.download_archive("https://source.test/archive", archive, DIGEST) == archive
    assert archive.read_bytes() == PAYLOAD


@pytest.mark.parametrize(
    "failure", [URLError("timeout"), TimeoutError(), IncompleteRead(b"part", 8)]
)
def test_download_retry(downloader, monkeypatch, tmp_path, failure):
    requests = []

    def request(url, *, timeout):
        requests.append((url, timeout))
        if len(requests) == 1:
            raise failure
        return io.BytesIO(PAYLOAD)

    monkeypatch.setattr(downloader.urllib.request, "urlopen", request)
    archive = tmp_path / "source.tar.gz"
    downloader.download_archive("https://source.test/archive", archive, DIGEST, timeout=12)
    assert archive.read_bytes() == PAYLOAD
    assert requests == [("https://source.test/archive", 12)] * 2
    assert list(tmp_path.iterdir()) == [archive]


def test_corrupt_cache_and_mirror(downloader, monkeypatch, tmp_path):
    archive = tmp_path / "source.tar.gz"
    archive.write_bytes(b"old incomplete cache")
    requests = []

    def request(url, **kwargs):
        requests.append(url)
        return io.BytesIO(b"corrupt" if url.endswith("primary") else PAYLOAD)

    monkeypatch.setattr(downloader.urllib.request, "urlopen", request)
    downloader.download_archive(
        ["https://source.test/primary", "https://source.test/mirror"], archive, DIGEST, attempts=2
    )
    assert archive.read_bytes() == PAYLOAD
    assert requests == ["https://source.test/primary"] * 2 + ["https://source.test/mirror"]
    assert list(tmp_path.iterdir()) == [archive]


def test_partial_response_is_not_published(downloader, monkeypatch, tmp_path):
    archive = tmp_path / "source.tar.gz"

    class Interrupted(io.BytesIO):
        def read(self, size=-1):
            if self.tell():
                assert not archive.exists()
                raise IncompleteRead(b"partial", 10)
            return super().read(4)

    monkeypatch.setattr(downloader.urllib.request, "urlopen", lambda *a, **k: Interrupted(PAYLOAD))
    with pytest.raises(RuntimeError, match="verified native source archive") as error:
        downloader.download_archive("https://source.test/archive", archive, DIGEST, attempts=2)
    assert isinstance(error.value.__cause__, IncompleteRead)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("failure", ["network", "checksum"])
def test_download_failure_is_bounded(downloader, monkeypatch, tmp_path, failure):
    requests = []

    def request(url, **kwargs):
        requests.append(url)
        if failure == "network":
            raise URLError("unreachable")
        return io.BytesIO(b"unverified bytes")

    monkeypatch.setattr(downloader.urllib.request, "urlopen", request)
    urls = ["https://source.test/primary", "https://source.test/mirror"]
    with pytest.raises(RuntimeError, match="verified native source archive"):
        downloader.download_archive(urls, tmp_path / "archive", DIGEST, attempts=2)
    assert requests == [urls[0], urls[0], urls[1], urls[1]]
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("single", [False, True])
def test_linux_repair_library_path(monkeypatch, tmp_path, single):
    module = _tool("repair_extension_wheels")
    source = tmp_path / "raw"
    source.mkdir()
    wheel = source / "algroots-0.4.1-cp311-cp311-linux_x86_64.whl"
    wheel.touch()
    prefix = tmp_path / "flint"
    (prefix / "lib").mkdir(parents=True)
    (prefix / "lib/libflint.so.24").touch()
    monkeypatch.setenv("ALGROOTS_FLINT_PREFIX", str(prefix))
    monkeypatch.setenv("LD_LIBRARY_PATH", "/python/lib")
    monkeypatch.setattr(
        module, "os", SimpleNamespace(name="posix", environ=os.environ, pathsep=os.pathsep)
    )
    monkeypatch.setattr(module, "sys", SimpleNamespace(platform="linux", executable=sys.executable))
    executable = str(tmp_path / "repair-tools" / "auditwheel")
    monkeypatch.setattr(
        module.shutil, "which", lambda name: executable if name == "auditwheel" else None
    )
    calls = []

    def repair(command, *, check, env):
        paths = env["LD_LIBRARY_PATH"].split(os.pathsep)
        assert paths == [str(prefix / "lib"), str(prefix / "lib64"), "/python/lib"]
        assert any((Path(path) / "libflint.so.24").is_file() for path in paths)
        calls.append(command)

    monkeypatch.setattr(module.subprocess, "run", repair)
    output = tmp_path / "repaired"
    monkeypatch.setattr(
        sys, "argv", ["repair", str(wheel if single else source), "--output", str(output)]
    )
    module.main()
    assert calls == [[executable, "repair", "-w", str(output), str(wheel)]]


def test_windows_repair_library_path(monkeypatch, tmp_path):
    module = _tool("repair_extension_wheels")
    wheel = tmp_path / "algroots-0.4.1-cp311-cp311-win_amd64.whl"
    wheel.touch()
    prefix = tmp_path / "flint"
    monkeypatch.setattr(
        module, "os", SimpleNamespace(name="nt", environ={"ALGROOTS_FLINT_PREFIX": str(prefix)})
    )
    monkeypatch.setattr(module, "sys", SimpleNamespace(platform="win32", executable="python.exe"))
    calls = []
    monkeypatch.setattr(module.subprocess, "run", lambda command, **kwargs: calls.append(command))
    output = tmp_path / "repaired"
    monkeypatch.setattr(sys, "argv", ["repair", str(wheel), "--output", str(output)])
    module.main()
    assert calls == [
        [
            "python.exe",
            "-m",
            "delvewheel",
            "repair",
            "--add-path",
            str(prefix / "bin"),
            "-w",
            str(output),
            str(wheel),
        ]
    ]
