from setuptools import setup, find_packages

setup(name='resilient_mrp',
      version='1.0.0',
      package_dir={"": "src"},
      description='Resilient Multi-Robot Planning Package',
      author='Morayo Ogunsina',
      author_email='mogunsin@gmu.edu',
      packages=find_packages(where="src"),
      install_requires=[])