from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11  # pyright: ignore[reportMissingImports]

class get_pybind_include:
    def __str__(self):
        return pybind11.get_include()

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
    version="0.1.0",
    package_dir={"": "src"},
    packages=["mrppddl"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
