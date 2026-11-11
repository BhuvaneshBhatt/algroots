# Native wheels and Python compatibility

Use a native wheel to run the default Gröbner, quotient and solver workflows without a compiler. It includes the F4 kernel, finite-field sparse arithmetic and the FLINT coefficient-field module, together with their required numerical libraries. The portable wheel provides explicit SymPy construction through `backend="sympy"` or `groebner_backend="sympy"`.

## Supported interpreter and platform combinations

Native extensions use CPython's Python 3.11 Limited API and Stable ABI. Each wheel is tagged `cp311-abi3`, so the same binary supports GIL-enabled CPython 3.11 and later Python 3.x releases on its declared operating system and architecture. Python dependencies and mathematical functionality are still tested on the supported interpreters. The ABI tag does not guarantee that unrelated dependencies support every future Python release.

| Wheel family | Architectures | Platform requirement |
| --- | --- | --- |
| Linux manylinux | x86-64 and ARM64 | The published manylinux tag's glibc floor |
| macOS | Intel and Apple silicon | macOS 14 or later |
| Windows | x86-64 | A compatible 64-bit Windows installation |

The `abi3` wheels target ordinary GIL-enabled CPython. Free-threaded builds and PyPy have different binary interfaces and are excluded from this wheel configuration. The CPython ABI does not remove operating-system, architecture or bundled-library requirements.

The wheel build selector is open-ended (`cp3*-*`) and uses `requires-python = ">=3.11"` to enforce the lower bound. cibuildwheel reuses the compatible `cp311-abi3` artifact when checking subsequent CPython versions. Updating cibuildwheel makes newly supported stable interpreters available to that build/test selection.

## Build locally

A source installation can enable all three native modules:

```sh
ALGROOTS_BUILD_NATIVE_F4=1 ALGROOTS_BUILD_FF_SPARSE=1 ALGROOTS_BUILD_NATIVE_EXTENSION=1 python -m pip install .
```

The FLINT module needs development headers and libraries. Use `pkg-config flint`, or set `ALGROOTS_FLINT_PREFIX` to a prefix containing `include/flint` and the FLINT, GMP and MPFR libraries in `lib`. See [coefficient-field builds](coefficient-fields.md#build-the-flint-extension) for details. Native builds define `Py_LIMITED_API=0x030B0000`; the extension filenames and wheel tags are configured together. A source build on a newer supported CPython still targets the Python 3.11 ABI floor.

A locally linked wheel can depend on libraries installed on that machine. Before distributing it, bundle dependencies with `tools/repair_extension_wheels.py`. Its platform repairers preserve the Stable-ABI tag and determine the applicable platform tag from the native binaries. The portable build keeps its `py3-none-any` tag.

## Validate artifacts

Run these checks on the repaired native wheel:

```sh
python -m pip install abi3audit
python tools/check_wheel_abi.py repaired --require-all
python -m pip install --force-reinstall --no-deps repaired/algroots-0.4.1-cp311-abi3-macosx_14_0_arm64.whl
python tools/wheel_smoke.py --native --extension-fields --stable-abi
```

Use the wheel filename for the target platform. The checker compares filename and metadata tags, requires all three modules, rejects interpreter-specific extension filenames, and invokes `abi3audit --strict` to detect private symbols or Stable-ABI symbols introduced after Python 3.11. The installation smoke test imports all three modules and verifies exact rational, finite-field, algebraic, parameter and tower computations. It can run unchanged on multiple interpreters using the same wheel.

The wheel workflow validates repaired artifacts for Linux x86-64/ARM64, macOS Intel/ARM64 and Windows x86-64. The regular native test jobs also run the ABI checks after repair. These checks establish binary compatibility and preserve the package's independent exact-verification requirements.
