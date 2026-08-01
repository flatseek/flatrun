"""Single source of truth for the flatrun build.

Both the Python package (``src/flatrun``) and the optional C++
extension (``src/flatrun_native/_C.cpp``) are built by this
``setup.py``. The extension is mandatory from setuptools' point
of view — it will be built whenever ``pip install -e .`` is run
with pybind11 present in the build environment.

If pybind11 is missing, the extension is excluded and the build
falls back to the pure-Python install: the runtime detects
``flatrun_native._C`` is unavailable and dispatches to the numpy
path.
"""

from __future__ import annotations

import os
import sys

from setuptools import setup, Extension
from pybind11.setup_helpers import build_ext

HERE = os.path.dirname(os.path.abspath(__file__))


def _has_pybind11() -> bool:
    try:
        import pybind11  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Compiler flags
# ---------------------------------------------------------------------------

extra_compile_args = ["-O3", "-fPIC", "-std=c++17"]
extra_link_args = []
if sys.platform.startswith("darwin"):
    # Always target arm64 on Apple Silicon. The default clang on PATH
    # may be the x86_64 toolchain under Rosetta, but the running
    # Python is arm64 native and the extension must match.
    extra_compile_args += ["-arch", "arm64"]
    extra_link_args += ["-arch", "arm64"]
elif sys.platform.startswith("linux"):
    if os.uname().machine in ("arm64", "aarch64"):
        extra_compile_args += ["-march=armv8-a"]


# ---------------------------------------------------------------------------
# Extension (only declared when pybind11 is available)
# ---------------------------------------------------------------------------

ext_modules = []
if _has_pybind11():
    import pybind11
    pybind11_include = pybind11.get_include()
    ext_modules = [
        Extension(
            "flatrun_native._C",
            sources=["src/flatrun_native/_C.cpp"],
            include_dirs=["src/flatrun_native", pybind11_include],
            extra_compile_args=extra_compile_args,
            language="c++",
            extra_link_args=extra_link_args,
        ),
    ]
else:
    sys.stderr.write(
        "[flatrun] pybind11 not installed; the C++ extension will not be "
        "built. Run 'pip install \"pybind11>=2.10\"' and reinstall to "
        "enable the native backend.\n"
    )


setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
