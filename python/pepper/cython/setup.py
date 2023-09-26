from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

"""
```sh
python setup.py build_ext --inplace
```
"""

# Define the Cython extension module
extensions = [
    Extension("ufuncs", ["ufuncs.pyx"], include_dirs=[np.get_include()])
]

setup(
    ext_modules=cythonize(extensions),
    include_dirs=[np.get_include()],
)
