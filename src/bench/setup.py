from setuptools import setup, find_packages

setup(
    name="bench",
    version="0.1.0",
    description="Benchmark harness for PDDL planning system",
    packages=find_packages(),
    python_requires=">=3.13",
    install_requires=[
        "mlflow",
        "rich",
        "pandas",
    ],
)
