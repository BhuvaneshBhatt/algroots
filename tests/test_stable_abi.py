"""Contracts for Stable-ABI native builds and platform wheel payloads."""

import importlib.util
import os
import runpy
import sysconfig
from pathlib import Path
from zipfile import ZipFile

import pytest
import setuptools
from setuptools.command.build_ext import build_ext

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("wheel_abi", ROOT / "tools/check_wheel_abi.py")
wheel_abi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wheel_abi)


def build_configuration(monkeypatch, *, native):
    captured = {}
    for flag in (
        "ALGROOTS_BUILD_NATIVE_F4",
        "ALGROOTS_BUILD_FF_SPARSE",
        "ALGROOTS_BUILD_NATIVE_EXTENSION",
    ):
        monkeypatch.setenv(flag, "1" if native else "0")
    monkeypatch.setenv("ALGROOTS_FLINT_PREFIX", "/build/flint")
    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: captured.update(kwargs))
    runpy.run_path(str(ROOT / "setup.py"))
    return captured


def test_native_configuration(monkeypatch):
    config = build_configuration(monkeypatch, native=True)
    assert config["options"]["bdist_wheel"] == {"py_limited_api": "cp311"}
    assert {extension.name for extension in config["ext_modules"]} == {
        "algroots._native_f4",
        "algroots._native_extension",
        "algroots._ff_sparse",
    }
    for extension in config["ext_modules"]:
        assert extension.py_limited_api
        assert extension.define_macros == [("Py_LIMITED_API", "0x030B0000")]
        assert "src/algroots/_python_abi.h" in extension.depends


def test_portable_configuration(monkeypatch):
    config = build_configuration(monkeypatch, native=False)
    assert config["ext_modules"] == []
    assert "bdist_wheel" not in config["options"]


def test_extension_filename(monkeypatch):
    config = build_configuration(monkeypatch, native=True)
    command = config["cmdclass"]["build_ext"](setuptools.Distribution())
    command.ext_map = {extension.name: extension for extension in config["ext_modules"]}
    monkeypatch.setattr(build_ext, "get_ext_filename", lambda self, name: name + ".abi3-darwin.so")
    suffix = ".pyd" if os.name == "nt" else ".abi3.so"
    for name in command.ext_map:
        assert command.get_ext_filename(name) == os.path.join(*name.split(".")) + suffix


def test_free_threaded_build(monkeypatch):
    original = sysconfig.get_config_var
    monkeypatch.setattr(
        sysconfig, "get_config_var", lambda key: 1 if key == "Py_GIL_DISABLED" else original(key)
    )
    with pytest.raises(RuntimeError, match="GIL-enabled CPython"):
        build_configuration(monkeypatch, native=True)


def wheel_fixture(
    tmp_path, *, platform="manylinux_2_28_x86_64", abi="abi3", payload=None, metadata_python="cp311"
):
    path = tmp_path / f"algroots-0.4.0-cp311-{abi}-{platform}.whl"
    suffix = ".pyd" if platform.startswith("win") else ".abi3.so"
    if payload is None:
        payload = [f"algroots/{name}{suffix}" for name in sorted(wheel_abi.NATIVE_MODULES)]
    with ZipFile(path, "w") as archive:
        archive.writestr(
            "algroots-0.4.0.dist-info/WHEEL",
            f"Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: {metadata_python}-{abi}-{platform}\n",
        )
        for name in payload:
            archive.writestr(name, b"fixture payload; symbol auditing is tested on built wheels")
    return path


@pytest.mark.parametrize("platform", ["manylinux_2_28_x86_64", "macosx_14_0_arm64", "win_amd64"])
def test_platform_metadata(tmp_path, platform):
    path = wheel_fixture(tmp_path, platform=platform)
    assert wheel_abi.inspect_wheel(path, require_all=True) == wheel_abi.NATIVE_MODULES


def test_specific_abi_rejected(tmp_path):
    path = wheel_fixture(tmp_path, abi="cp311")
    with pytest.raises(ValueError, match="expected cp311-abi3"):
        wheel_abi.inspect_wheel(path)


def test_mistagged_extension(tmp_path):
    path = wheel_fixture(tmp_path, payload=["algroots/_native_f4.cpython-311-x86_64-linux-gnu.so"])
    with pytest.raises(ValueError, match="invalid or duplicate extension"):
        wheel_abi.inspect_wheel(path)


def test_incomplete_native_payload(tmp_path):
    path = wheel_fixture(tmp_path, payload=["algroots/_native_f4.abi3.so"])
    assert wheel_abi.inspect_wheel(path) == {"_native_f4"}
    with pytest.raises(ValueError, match="native module set"):
        wheel_abi.inspect_wheel(path, require_all=True)


def test_metadata_tag_mismatch(tmp_path):
    path = wheel_fixture(tmp_path, metadata_python="cp312")
    with pytest.raises(ValueError, match="binary wheel metadata disagrees"):
        wheel_abi.inspect_wheel(path)
