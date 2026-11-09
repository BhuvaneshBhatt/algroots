"""Tests for native dependency source downloads."""

import importlib.util
from pathlib import Path
from urllib.error import URLError

import pytest


class _Response:
    def __init__(self, payload: bytes):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return self._payload


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "build_native_dependencies.py"
MODULE_SPEC = importlib.util.spec_from_file_location("build_native_dependencies", MODULE_PATH)
assert MODULE_SPEC and MODULE_SPEC.loader
BUILD_NATIVE_DEPENDENCIES = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(BUILD_NATIVE_DEPENDENCIES)


def test_download_archive_retries_and_succeeds(monkeypatch, tmp_path):
    calls = {"count": 0}

    def flaky_urlopen(_url, *, timeout):
        assert timeout == 120
        calls["count"] += 1
        if calls["count"] < 3:
            raise URLError(TimeoutError("timed out"))
        return _Response(b"archive-bytes")

    sleeps = []
    monkeypatch.setattr(BUILD_NATIVE_DEPENDENCIES.urllib.request, "urlopen", flaky_urlopen)
    monkeypatch.setattr(BUILD_NATIVE_DEPENDENCIES.time, "sleep", sleeps.append)

    archive = tmp_path / "native.tar.gz"
    BUILD_NATIVE_DEPENDENCIES.download_archive("https://example.test/native.tar.gz", archive)

    assert archive.read_bytes() == b"archive-bytes"
    assert calls["count"] == 3
    assert sleeps == [2.0, 4.0]


def test_download_archive_raises_after_last_retry(monkeypatch, tmp_path):
    calls = {"count": 0}

    def always_fails(_url, *, timeout):
        assert timeout == 120
        calls["count"] += 1
        raise URLError(TimeoutError("timed out"))

    sleeps = []
    monkeypatch.setattr(BUILD_NATIVE_DEPENDENCIES.urllib.request, "urlopen", always_fails)
    monkeypatch.setattr(BUILD_NATIVE_DEPENDENCIES.time, "sleep", sleeps.append)

    archive = tmp_path / "native.tar.gz"
    with pytest.raises(URLError):
        BUILD_NATIVE_DEPENDENCIES.download_archive(
            "https://example.test/native.tar.gz",
            archive,
            attempts=4,
        )

    assert calls["count"] == 4
    assert sleeps == [2.0, 4.0, 8.0]
