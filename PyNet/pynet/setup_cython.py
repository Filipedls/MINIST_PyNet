from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
  Extension('im2col_cy', ['im2col_cy.pyx'],
    include_dirs = [numpy.get_include()]),
]

setup(
    ext_modules = cythonize(extensions),
)