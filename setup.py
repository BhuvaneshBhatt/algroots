"""Optional compiled Gröbner kernels, with pure-Python fallback available."""

import hashlib
import json
import os
import shlex
import subprocess
import sys
import sysconfig
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class StableABIBuild(build_ext):
    """Keep extension filenames loadable by every interpreter from Python 3.11."""

    def get_ext_filename(self, fullname):
        filename = super().get_ext_filename(fullname)
        extension = self.ext_map.get(fullname)
        if extension is not None and extension.py_limited_api:
            # Python 3.15 adds platform details to abi3 suffixes that older
            # importers do not recognize. The original suffix works on both.
            suffix = ".pyd" if os.name == "nt" else ".abi3.so"
            return os.path.join(*fullname.split(".")) + suffix
        return filename


limited_api = {
    "define_macros": [("Py_LIMITED_API", "0x030B0000")],
    "py_limited_api": True,
}
extensions = []
if os.environ.get("ALGROOTS_BUILD_NATIVE_EXTENSION") == "1":
    prefix = os.environ.get("ALGROOTS_FLINT_PREFIX")
    if prefix:
        include_dirs = [str(Path(prefix) / "include")]
        library_dirs = [str(Path(prefix) / "lib")]
        libraries = ["flint", "mpfr", "gmp"]
        link_args = []
    else:
        try:
            process = subprocess.run(
                ["pkg-config", "--cflags", "--libs", "flint"],
                capture_output=True,
                text=True,
                check=True,
                timeout=15,
            )
        except (
            FileNotFoundError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
        ) as error:
            raise RuntimeError(
                "native extension fields require FLINT development files and pkg-config, or ALGROOTS_FLINT_PREFIX"
            ) from error
        flags = shlex.split(process.stdout)
        include_dirs = [flag[2:] for flag in flags if flag.startswith("-I")]
        library_dirs = [flag[2:] for flag in flags if flag.startswith("-L")]
        libraries = [flag[2:] for flag in flags if flag.startswith("-l")]
        link_args = [flag for flag in flags if not flag.startswith(("-I", "-L", "-l"))]
    extensions.append(
        Extension(
            "algroots._native_extension",
            ["src/algroots/_native_extension.cpp"],
            language="c++",
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=libraries,
            extra_link_args=link_args,
            extra_compile_args=["/std:c++17", "/O2"] if os.name == "nt" else ["-O2", "-std=c++17"],
            depends=["src/algroots/_native_f4_python.h", "src/algroots/_python_abi.h"],
            **limited_api,
        )
    )
if os.environ.get("ALGROOTS_BUILD_FF_SPARSE") == "1":
    extensions.append(
        Extension(
            "algroots._ff_sparse",
            ["src/algroots/_ff_sparse.c"],
            depends=["src/algroots/_python_abi.h"],
            **limited_api,
        )
    )
if os.environ.get("ALGROOTS_BUILD_NATIVE_F4") == "1":
    native_opt = os.environ.get("ALGROOTS_NATIVE_F4_OPT_LEVEL", "-O2")
    if native_opt not in ("-O0", "-O1", "-O2", "-O3"):
        raise ValueError("ALGROOTS_NATIVE_F4_OPT_LEVEL must be -O0, -O1, -O2 or -O3")
    extensions.append(
        Extension(
            "algroots._native_f4",
            ["src/algroots/_native_f4.cpp"],
            language="c++",
            extra_compile_args=(
                ["/std:c++17", "/Od" if native_opt == "-O0" else "/O2"]
                if os.name == "nt"
                else [native_opt, "-std=c++17"]
            ),
            depends=[
                "src/algroots/_native_f4_python.h",
                "src/algroots/_native_f4_bindings.inc",
                "src/algroots/_native_f4_stats.inc",
                "src/algroots/_python_abi.h",
            ],
            **limited_api,
        )
    )
if extensions and (
    sys.implementation.name != "cpython" or sysconfig.get_config_var("Py_GIL_DISABLED")
):
    raise RuntimeError("native cp311-abi3 builds require GIL-enabled CPython 3.11 or later")

# Setuptools otherwise reuses the same build directory across feature and
# library changes, allowing a wheel to retain disabled kernels or old links.
configuration = {
    name: os.environ.get(name, "")
    for name in (
        "ALGROOTS_BUILD_NATIVE_F4",
        "ALGROOTS_BUILD_FF_SPARSE",
        "ALGROOTS_BUILD_NATIVE_EXTENSION",
        "ALGROOTS_FLINT_PREFIX",
        "ALGROOTS_NATIVE_F4_OPT_LEVEL",
        "CXX",
        "CC",
        "CFLAGS",
        "CXXFLAGS",
        "LDFLAGS",
        "MACOSX_DEPLOYMENT_TARGET",
        "_PYTHON_HOST_PLATFORM",
    )
}
configuration["limited_api"] = "0x030B0000"
configuration["build_python"] = sys.implementation.cache_tag
build_key = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()[:12]
build_profile = "native" if extensions else "python"
options = {"build": {"build_base": f"build/{build_profile}-{build_key}"}}
if extensions:
    options["bdist_wheel"] = {"py_limited_api": "cp311"}
setup(ext_modules=extensions, options=options, cmdclass={"build_ext": StableABIBuild})
