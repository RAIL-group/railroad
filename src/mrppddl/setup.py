from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

import pybind11  # type: ignore[import-not-found]  # pyright: ignore[reportMissingImports]


class get_pybind_include:
    def __str__(self):
        return pybind11.get_include()


class build_ext_and_stubgen(build_ext):
    """
    Build the C++ extension, then generate .pyi stubs via pybind11-stubgen.

    Generates: src/mrppddl/_bindings.pyi
    """

    def run(self):
        super().run()
        self._run_stubgen()

    def _run_stubgen(self):
        # The compiled extension ends up under self.build_lib (temporary build dir).
        build_lib = Path(self.build_lib).resolve()

        module = "mrppddl._bindings"

        # Where to write the final stub inside your source tree/package.
        out_pyi = Path("src/mrppddl/_bindings.pyi").resolve()

        # Optional: skip if stub is newer than built extension
        ext_path = Path(self.get_ext_fullpath(module)).resolve()
        if out_pyi.exists() and ext_path.exists():
            if out_pyi.stat().st_mtime >= ext_path.stat().st_mtime:
                return

        # Make sure the freshly built extension is importable:
        # - add build_lib to PYTHONPATH
        env = os.environ.copy()
        env["PYTHONPATH"] = str(build_lib) + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )

        # Generate stubs into a temp dir under build/
        stub_out_dir = build_lib / "_stubgen"
        stub_out_dir.mkdir(parents=True, exist_ok=True)

        # Run pybind11-stubgen using the current Python executable
        subprocess.check_call(
            [sys.executable, "-m", "pybind11_stubgen", module, "-o", str(stub_out_dir)],
            env=env,
        )

        # pybind11-stubgen writes a package-style tree:
        #   <out>/mrppddl/_bindings.pyi
        generated = stub_out_dir / "mrppddl" / "_bindings.pyi"
        if not generated.exists():
            raise RuntimeError(f"Stubgen did not produce expected file: {generated}")

        out_pyi.parent.mkdir(parents=True, exist_ok=True)

        # Post-process: pybind11-stubgen sometimes emits fully-qualified names
        # like `mrppddl._bindings.Action` instead of just `Action`. Since we're
        # inside the _bindings module, strip the prefix to avoid import errors.
        stub_content = generated.read_text(encoding="utf-8")
        stub_content = stub_content.replace("mrppddl._bindings.", "")

        out_pyi.write_text(stub_content, encoding="utf-8")


ext_modules = [
    Extension(
        "mrppddl._bindings",
        sources=["src/mrppddl/_bindings.cpp"],
        include_dirs=[
            "include",
            get_pybind_include(),  # pyright: ignore[reportArgumentType]
        ],
        language="c++",
        extra_compile_args=["-std=c++17"],
    )
]

setup(
    name="mrppddl",
    version="0.2.0",
    package_dir={"": "src"},
    packages=["mrppddl"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext_and_stubgen},
    include_package_data=True,
    package_data={"mrppddl": ["*.pyi"]},
    zip_safe=False,
)
