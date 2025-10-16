from setuptools import setup, find_packages
from os import path

# Copy readme to long description
this_dir = path.abspath(path.dirname(__file__))
with open(path.join(this_dir, "README.md")) as f:
    long_description = f.read()

# always required packages
required_pkgs = ['numpy',
                 'scipy',
                 'matplotlib',
                 'datetime',
                 'sympy',
                 'tqdm',
                 'dask',
                 'scikit-image',
                 'zarr',
                 'ndtiff',
                 'localize_psf @ git+https://git@github.com/qi2lab/localize-psf@master#egg=localize_psf'
                 ]

extras = None

setup(
      name='SPIM_model',
      version='0.1.0',
      description="Python package for simulating SPIM excitation light sheet",
      long_description=long_description,
      author='qi2lab, Steven J Sheppard, Peter Brown',
      packages=find_packages(include=['model_tools']),
      python_requires='>=3.10',
      install_requires=required_pkgs
      )