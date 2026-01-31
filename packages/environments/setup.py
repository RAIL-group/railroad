from setuptools import setup, find_packages


setup(name='environments',
      version='1.0.0',
      package_dir={"": "src"},
      description='Environments',
      license="MIT",
      author='Gregory J. Stein',
      author_email='gjstein@gmu.edu',
      packages=find_packages(),
      install_requires=[],
      entry_points={
          "railroad.benchmarks": [
              "environments=environments.benchmarks",
          ],
      })
