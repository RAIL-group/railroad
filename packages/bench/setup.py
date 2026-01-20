from setuptools import setup, find_packages

setup(
    name="bench",
    version="0.1.0",
    package_dir={"": "src"},
    description="Benchmark harness for PDDL planning system",
    packages=find_packages(),
    python_requires=">=3.13",
    install_requires=[
        "mlflow",
        "rich",
        "pandas",
    ],
    entry_points={
        "console_scripts": [
            "benchmarks-dashboard=bench.dashboard.app:main",
            "benchmarks-run=bench.cli:main",
        ],
    },
)
