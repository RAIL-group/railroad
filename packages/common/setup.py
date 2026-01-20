from setuptools import setup, find_packages


setup(name='common',
      version='1.0.0',
      package_dir={"": "src"},
      description='Some shared resources for RAIL-core',
      license="MIT",
      author='Gregory J. Stein',
      author_email='gjstein@gmu.edu',
      packages=find_packages(),
      install_requires=[])
