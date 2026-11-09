# Bundled native libraries

FLINT-enabled binary wheels can bundle FLINT, GMP and MPFR as shared libraries. These are independent upstream projects; library versions are recorded in the build evidence and wheel contents. Their notices and complete license texts are included here.

- FLINT: copyright the FLINT contributors; GNU Lesser General Public License version 3 or later. Source: https://flintlib.org/downloads.html and https://github.com/flintlib/flint.
- GMP: copyright the Free Software Foundation and GMP contributors; distributed here under GNU Lesser General Public License version 3 or later. Source: https://gmplib.org/download/gmp/.
- MPFR: copyright the Free Software Foundation and MPFR contributors; GNU Lesser General Public License version 3 or later. Source: https://www.mpfr.org/mpfr-current/.

The libraries are linked dynamically. Rebuilding with ALGROOTS_FLINT_PREFIX selects a locally installed compatible FLINT/GMP/MPFR build. The algroots source distribution contains its native extension sources and build instructions. Pure Python wheels do not bundle these libraries.
